"""Graph extraction integration point — single function for pipeline use."""

import logging

from app.config import settings
from app.graph.coref import resolve_coreferences
from app.graph.dedup import deduplicate_entities, deduplicate_relationships
from app.graph.extractor import extract_from_chunks
from app.graph.models import GraphExtractionResult
from app.graph.schema import ensure_schema
from app.graph.store import clear_document_graph, store_graph
from app.ingestion.chunker import Chunk
from app.ingestion.version_tracker import DocumentRecord

logger = logging.getLogger(__name__)


async def extract_and_store_graph(
    chunks: list[Chunk], doc_record: DocumentRecord,
) -> GraphExtractionResult:
    """Extract entities/relationships from chunks and store in Neo4j.

    Non-blocking: returns skipped=True on any failure so pipeline continues.
    """
    if not settings.GRAPH_EXTRACTION_ENABLED:
        return GraphExtractionResult(
            entities=[], relationships=[],
            entity_count=0, relationship_count=0,
            skipped=True, error=None,
        )

    try:
        resolved_texts = None
        if settings.COREF_ENABLED:
            try:
                resolved_texts = await resolve_coreferences(chunks)
            except Exception as exc:
                logger.warning("Coref failed, using originals: %s", exc)

        entities, relationships = await extract_from_chunks(
            chunks, doc_record.id, resolved_texts,
        )

        entities = deduplicate_entities(entities)
        relationships = deduplicate_relationships(relationships, entities)

        await ensure_schema()
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
