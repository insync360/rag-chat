"""Metadata enricher — LLM-powered chunk enrichment via GPT-4o-mini.

Single API call per chunk produces summary, keywords, hypothetical questions,
and entity tags. Chunk type and freshness score are computed deterministically.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from enum import Enum

from openai import AsyncOpenAI

from app.config import settings
from app.ingestion.chunker import Chunk, _token_count
from app.ingestion.version_tracker import DocumentRecord

logger = logging.getLogger(__name__)


class ChunkType(str, Enum):
    PARAGRAPH = "PARAGRAPH"
    TABLE = "TABLE"
    CODE = "CODE"
    HEADING = "HEADING"
    LIST = "LIST"


_LIST_RE = re.compile(r"^[\s]*([-*+]|\d+[.)]) ", re.MULTILINE)


def _classify_chunk_type(chunk: Chunk) -> ChunkType:
    if chunk.has_table:
        return ChunkType.TABLE
    if chunk.has_code:
        return ChunkType.CODE
    content = chunk.content.strip()
    if content.startswith("#") and _token_count(content) < 20:
        return ChunkType.HEADING
    lines = content.split("\n")
    list_lines = sum(1 for line in lines if _LIST_RE.match(line))
    if lines and list_lines / len(lines) > 0.5:
        return ChunkType.LIST
    return ChunkType.PARAGRAPH


def _compute_freshness_score(ingested_at: datetime) -> float:
    now = datetime.now(timezone.utc)
    if ingested_at.tzinfo is None:
        ingested_at = ingested_at.replace(tzinfo=timezone.utc)
    days_since = (now - ingested_at).total_seconds() / 86400
    return round(max(0.0, 1.0 - (days_since / 365)), 4)


_SYSTEM_PROMPT = """You are a metadata extraction assistant. Given a document chunk, return a JSON object with exactly these keys:
- "summary": 2-3 sentence summary of the chunk content
- "keywords": array of 5-10 important keywords/phrases
- "hypothetical_questions": array of 3-5 questions this chunk could answer
- "entities": object with keys "people", "organizations", "dates", "money" — each an array of strings found in the chunk

Return ONLY valid JSON, no markdown fences."""


def _build_messages(content: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"<chunk>\n{content}\n</chunk>"},
    ]


async def _enrich_single(
    client: AsyncOpenAI,
    chunk: Chunk,
    semaphore: asyncio.Semaphore,
    doc_record: DocumentRecord,
) -> Chunk:
    now_iso = datetime.now(timezone.utc).isoformat()
    chunk_type = _classify_chunk_type(chunk)
    freshness = _compute_freshness_score(doc_record.ingested_at)

    base_metadata = {
        "chunk_type": chunk_type.value,
        "freshness_score": freshness,
        "document_id": doc_record.id,
        "section_path": chunk.section_path,
        "version": doc_record.version,
        "content_hash": doc_record.content_hash,
        "ingested_at": doc_record.ingested_at.isoformat(),
        "enriched_at": now_iso,
    }

    # Skip LLM for tiny chunks
    if _token_count(chunk.content) < 10:
        chunk.metadata = base_metadata
        return chunk

    llm_data = {}
    last_error = None
    async with semaphore:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=settings.ENRICHMENT_MODEL,
                    messages=_build_messages(chunk.content),
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=1024,
                )
                llm_data = json.loads(resp.choices[0].message.content)
                break
            except (json.JSONDecodeError, Exception) as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)

    if not llm_data and last_error:
        logger.warning(
            "Enrichment failed for chunk %d of doc %s after 3 retries: %s",
            chunk.chunk_index, doc_record.id[:12], last_error,
        )

    chunk.metadata = {
        "summary": llm_data.get("summary", ""),
        "keywords": llm_data.get("keywords", []),
        "hypothetical_questions": llm_data.get("hypothetical_questions", []),
        "entities": llm_data.get("entities", {"people": [], "organizations": [], "dates": [], "money": []}),
        **base_metadata,
    }
    return chunk


async def enrich_chunks(chunks: list[Chunk], doc_record: DocumentRecord) -> list[Chunk]:
    """Enrich all chunks with LLM-generated metadata. Never raises — failures get fallback metadata."""
    if not chunks:
        return chunks

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    semaphore = asyncio.Semaphore(settings.ENRICHMENT_CONCURRENCY)

    enriched = await asyncio.gather(*(
        _enrich_single(client, chunk, semaphore, doc_record)
        for chunk in chunks
    ))

    llm_count = sum(1 for c in enriched if c.metadata.get("summary"))
    logger.info(
        "Enriched %d/%d chunks for document %s (LLM: %d, fallback: %d)",
        len(enriched), len(chunks), doc_record.id[:12],
        llm_count, len(enriched) - llm_count,
    )
    return list(enriched)
