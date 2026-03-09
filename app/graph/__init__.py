"""Graph extraction integration point — single function for pipeline use.

NOTE: coref and ingestion imports are deferred to avoid circular import
(app.graph → app.ingestion.chunker → app.ingestion.__init__ → app.ingestion.pipeline → app.graph).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.config import settings
from app.graph.dedup import (
    deduplicate_entities,
    deduplicate_entities_enhanced,
    deduplicate_relationships,
)
from app.graph.extractor import extract_from_chunks
from app.graph.models import Entity, GraphExtractionResult
from app.graph.schema import ensure_schema
from app.graph.store import (
    clear_document_graph,
    deprecate_chunk_entities,
    get_document_entities,
    store_graph,
)

if TYPE_CHECKING:
    from app.ingestion.chunker import Chunk
    from app.ingestion.version_tracker import DocumentRecord

logger = logging.getLogger(__name__)


async def extract_and_store_graph(
    chunks: list[Chunk],
    doc_record: DocumentRecord,
    *,
    old_document_id: str | None = None,
    changed_indices: set[int] | None = None,
) -> GraphExtractionResult:
    """Extract entities/relationships from chunks and store in Neo4j.

    When changed_indices provided (re-ingestion) and INCREMENTAL_GRAPH_ENABLED:
    - Only extract from changed chunks (saves GPT-4o cost)
    - Fetch existing entities from Neo4j and merge via enhanced dedup
    - Deprecate old entities from changed chunks (never delete)

    Non-blocking: returns skipped=True on any failure so pipeline continues.
    """
    if not settings.GRAPH_EXTRACTION_ENABLED:
        return GraphExtractionResult(
            entities=[], relationships=[],
            entity_count=0, relationship_count=0,
            skipped=True, error=None,
        )

    try:
        incremental = (
            settings.INCREMENTAL_GRAPH_ENABLED
            and changed_indices is not None
            and old_document_id is not None
        )

        # Filter chunks for incremental mode
        extraction_chunks = chunks
        if incremental:
            extraction_chunks = [c for c in chunks if c.chunk_index in changed_indices]
            if not extraction_chunks:
                logger.info("No changed chunks for doc %s — skipping extraction", doc_record.id[:12])
                return GraphExtractionResult(
                    entities=[], relationships=[],
                    entity_count=0, relationship_count=0,
                    skipped=False, error=None,
                )

        # Coreference resolution on extraction chunks only
        resolved_texts = None
        if settings.COREF_ENABLED:
            try:
                from app.graph.coref import resolve_coreferences
                resolved_texts = await resolve_coreferences(extraction_chunks)
            except Exception as exc:
                logger.warning("Coref failed, using originals: %s", exc)

        entities, relationships = await extract_from_chunks(
            extraction_chunks, doc_record.id, resolved_texts,
        )

        # Dedup — enhanced (fuzzy+embedding) with optional existing entities
        existing_entities: list[Entity] | None = None
        if incremental:
            try:
                raw = await get_document_entities(old_document_id)
                existing_entities = [
                    Entity(
                        name=r["name"], type=r["type"],
                        source_chunk_index=r.get("source_chunk_index", 0),
                        source_document_id=old_document_id,
                        properties=r.get("properties") or {},
                        confidence=r.get("confidence", 1.0),
                    )
                    for r in raw
                ]
            except Exception as exc:
                logger.warning("Failed to fetch existing entities: %s", exc)

        entities = await deduplicate_entities_enhanced(entities, existing_entities)
        relationships = deduplicate_relationships(relationships, entities)

        await ensure_schema()

        # Deprecate old entities from changed chunks (incremental) or whole doc (full)
        if incremental:
            try:
                await deprecate_chunk_entities(old_document_id, changed_indices)
            except Exception as exc:
                logger.warning("Deprecation failed, continuing: %s", exc)
        else:
            await clear_document_graph(doc_record.id)

        await store_graph(entities, relationships, doc_record.id)

        return GraphExtractionResult(
            entities=entities,
            relationships=relationships,
            entity_count=len(entities),
            relationship_count=len(relationships),
            skipped=False,
            error=None,
        )

    except Exception as exc:
        logger.warning(
            "Graph extraction failed for doc %s, pipeline continues: %s",
            doc_record.id[:12], exc,
        )
        return GraphExtractionResult(
            entities=[], relationships=[],
            entity_count=0, relationship_count=0,
            skipped=True, error=str(exc),
        )
