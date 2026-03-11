"""Chunk embedder — generates text-embedding-3-large vectors and stores in pgvector.

Batched OpenAI API calls, upsert to chunk_embeddings table. Never raises.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from app.config import settings
from app.database import get_pool

if TYPE_CHECKING:
    from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    total: int
    embedded: int
    skipped: bool = False
    error: str | None = None


def _build_embedding_text(chunk: Chunk) -> str:
    """Combine chunk content with enriched metadata for richer embeddings.

    Includes section_path, summary, keywords, and hypothetical questions
    to improve semantic density and query-space coverage.
    """
    meta = getattr(chunk, "metadata", None) or {}
    parts: list[str] = []

    if chunk.section_path:
        parts.append(f"Section: {chunk.section_path}")

    summary = meta.get("summary")
    if summary:
        parts.append(f"Summary: {summary}")

    keywords = meta.get("keywords")
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}")

    questions = meta.get("hypothetical_questions")
    if questions:
        parts.append(f"Questions: {' '.join(questions)}")

    parts.append(chunk.content)
    return "\n\n".join(parts)


async def embed_chunks(chunk_ids: list[str], chunks: list[Chunk]) -> EmbeddingResult:
    """Generate embeddings for chunks and upsert to chunk_embeddings. Never raises."""
    total = len(chunk_ids)
    if total == 0:
        return EmbeddingResult(total=0, embedded=0)

    try:
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        pool = await get_pool()

        texts = [_build_embedding_text(c) for c in chunks]

        all_embeddings = []
        batch_size = settings.CHUNK_EMBEDDING_BATCH_SIZE
        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            resp = await client.embeddings.create(
                model=settings.CHUNK_EMBEDDING_MODEL,
                input=batch,
                dimensions=settings.CHUNK_EMBEDDING_DIMENSIONS,
            )
            all_embeddings.extend([d.embedding for d in resp.data])

        rows = [
            (chunk_ids[i], str(all_embeddings[i]), settings.CHUNK_EMBEDDING_MODEL)
            for i in range(total)
        ]
        await pool.executemany(
            "INSERT INTO chunk_embeddings (chunk_id, embedding, model) "
            "VALUES ($1::uuid, $2::vector, $3) "
            "ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding, "
            "model = EXCLUDED.model",
            rows,
        )

        logger.info("Embedded %d chunks with %s", total, settings.CHUNK_EMBEDDING_MODEL)
        return EmbeddingResult(total=total, embedded=total)

    except Exception as exc:
        logger.warning("Chunk embedding failed: %s", exc)
        return EmbeddingResult(total=total, embedded=0, skipped=True, error=str(exc))
