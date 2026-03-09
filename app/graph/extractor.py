"""GPT-4o entity + relationship extraction from chunks."""

import asyncio
import json
import logging

from openai import AsyncOpenAI

from app.config import settings
from app.graph.models import Entity, Relationship
from app.ingestion.chunker import Chunk, _token_count

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a knowledge graph extraction assistant. Given a document chunk, extract entities and relationships.

Return a JSON object with exactly these keys:
- "entities": array of objects with keys:
  - "name": canonical name (e.g. "Acme Corporation", not "Acme" or "acme corp")
  - "type": one of Person, Organization, Product, Policy, Date, Location, Metric, Event, Technology, Role, Document, Regulation (or other appropriate type)
  - "confidence": float 0.0-1.0
  - "properties": object with any additional attributes
- "relationships": array of objects with keys:
  - "source": name of source entity (must match an entity name above)
  - "target": name of target entity (must match an entity name above)
  - "type": relationship type in UPPER_SNAKE_CASE (e.g. REPORTS_TO, SUPERSEDES, REFERENCES, CAUSED_BY, APPLIES_TO, DEFINED_IN, MEMBER_OF, LOCATED_IN, PRODUCES, EMPLOYS)
  - "confidence": float 0.0-1.0
  - "properties": object with any additional attributes

Rules:
- Use canonical, full names for entities
- Only extract relationships where BOTH entities appear in the chunk
- Assign confidence based on how explicitly the entity/relationship is stated
- Return ONLY valid JSON, no markdown fences"""


def _build_messages(content: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"<chunk>\n{content}\n</chunk>"},
    ]


async def _extract_single(
    client: AsyncOpenAI,
    chunk: Chunk,
    document_id: str,
    semaphore: asyncio.Semaphore,
) -> tuple[list[Entity], list[Relationship]]:
    """Extract entities and relationships from a single chunk. Never raises."""
    if _token_count(chunk.content) < 10:
        return [], []

    entities: list[Entity] = []
    relationships: list[Relationship] = []
    last_error = None

    async with semaphore:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=settings.GRAPH_EXTRACTION_MODEL,
                    messages=_build_messages(chunk.content),
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=2048,
                )
                data = json.loads(resp.choices[0].message.content)

                for e in data.get("entities", []):
                    entities.append(Entity(
                        name=e["name"],
                        type=e.get("type", "Unknown"),
                        source_chunk_index=chunk.chunk_index,
                        source_document_id=document_id,
                        properties=e.get("properties", {}),
                        confidence=float(e.get("confidence", 1.0)),
                    ))

                for r in data.get("relationships", []):
                    relationships.append(Relationship(
                        source_entity=r["source"],
                        target_entity=r["target"],
                        type=r.get("type", "RELATED_TO"),
                        source_chunk_index=chunk.chunk_index,
                        source_document_id=document_id,
                        confidence=float(r.get("confidence", 1.0)),
                        properties=r.get("properties", {}),
                    ))
                break

            except (json.JSONDecodeError, KeyError, Exception) as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)

    if not entities and last_error:
        logger.warning(
            "Graph extraction failed for chunk %d of doc %s: %s",
            chunk.chunk_index, document_id[:12], last_error,
        )

    return entities, relationships


async def extract_from_chunks(
    chunks: list[Chunk], document_id: str,
) -> tuple[list[Entity], list[Relationship]]:
    """Extract entities and relationships from all chunks. Never raises."""
    if not chunks:
        return [], []

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    semaphore = asyncio.Semaphore(settings.GRAPH_EXTRACTION_CONCURRENCY)

    results = await asyncio.gather(*(
        _extract_single(client, chunk, document_id, semaphore)
        for chunk in chunks
    ))

    all_entities: list[Entity] = []
    all_relationships: list[Relationship] = []
    for ents, rels in results:
        all_entities.extend(ents)
        all_relationships.extend(rels)

    logger.info(
        "Extracted %d entities, %d relationships from %d chunks (doc %s)",
        len(all_entities), len(all_relationships), len(chunks), document_id[:12],
    )
    return all_entities, all_relationships
