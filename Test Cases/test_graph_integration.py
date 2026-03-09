"""Integration tests for the full graph extraction flow."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.graph import extract_and_store_graph
from app.graph.models import Entity, GraphExtractionResult, Relationship
from app.ingestion.chunker import Chunk
from app.ingestion.version_tracker import DocumentRecord


def _doc_record(doc_id=None):
    return DocumentRecord(
        id=doc_id or str(uuid.uuid4()), filename="test.pdf",
        content_hash="abc123", version=1, status="active",
        page_count=1, ingested_at=datetime.now(timezone.utc), metadata={},
    )


def _chunks(doc_id, count=2):
    return [
        Chunk(
            document_id=doc_id, chunk_index=i,
            content=f"Acme Corporation hired person {i} as engineer.",
            token_count=20, section_path="Test",
            has_table=False, has_code=False,
            overlap_tokens=0, metadata={},
        )
        for i in range(count)
    ]


def _entities(doc_id):
    return [
        Entity(name="Acme Corporation", type="Organization",
               source_chunk_index=0, source_document_id=doc_id, confidence=0.9),
    ]


def _relationships(doc_id):
    return [
        Relationship(source_entity="Acme Corporation", target_entity="Person 0",
                      type="EMPLOYS", source_chunk_index=0,
                      source_document_id=doc_id, confidence=0.85),
    ]


# --- Tests ---


@pytest.mark.asyncio
async def test_full_flow_success():
    doc = _doc_record()
    chunks = _chunks(doc.id)
    ents = _entities(doc.id)
    rels = _relationships(doc.id)

    with patch("app.graph.extract_from_chunks", new_callable=AsyncMock, return_value=(ents, rels)), \
         patch("app.graph.deduplicate_entities", return_value=ents), \
         patch("app.graph.deduplicate_relationships", return_value=rels), \
         patch("app.graph.ensure_schema", new_callable=AsyncMock), \
         patch("app.graph.clear_document_graph", new_callable=AsyncMock), \
         patch("app.graph.store_graph", new_callable=AsyncMock) as mock_store, \
         patch("app.graph.settings") as mock_settings:
        mock_settings.GRAPH_EXTRACTION_ENABLED = True

        result = await extract_and_store_graph(chunks, doc)

    assert result.skipped is False
    assert result.entity_count == 1
    assert result.relationship_count == 1
    assert result.error is None
    mock_store.assert_called_once_with(ents, rels, doc.id)


@pytest.mark.asyncio
async def test_neo4j_failure_returns_skipped():
    doc = _doc_record()
    chunks = _chunks(doc.id)

    with patch("app.graph.extract_from_chunks", new_callable=AsyncMock, return_value=([], [])), \
         patch("app.graph.deduplicate_entities", return_value=[]), \
         patch("app.graph.deduplicate_relationships", return_value=[]), \
         patch("app.graph.ensure_schema", new_callable=AsyncMock, side_effect=Exception("Neo4j down")), \
         patch("app.graph.settings") as mock_settings:
        mock_settings.GRAPH_EXTRACTION_ENABLED = True

        result = await extract_and_store_graph(chunks, doc)

    assert result.skipped is True
    assert "Neo4j down" in result.error


@pytest.mark.asyncio
async def test_disabled_flag_skips():
    doc = _doc_record()
    chunks = _chunks(doc.id)

    with patch("app.graph.settings") as mock_settings:
        mock_settings.GRAPH_EXTRACTION_ENABLED = False

        result = await extract_and_store_graph(chunks, doc)

    assert result.skipped is True
    assert result.entity_count == 0
    assert result.error is None


@pytest.mark.asyncio
async def test_pipeline_continues_on_graph_failure():
    """Verify extract_and_store_graph never raises — always returns a result."""
    doc = _doc_record()
    chunks = _chunks(doc.id)

    with patch("app.graph.extract_from_chunks", new_callable=AsyncMock, side_effect=Exception("LLM crash")), \
         patch("app.graph.settings") as mock_settings:
        mock_settings.GRAPH_EXTRACTION_ENABLED = True

        result = await extract_and_store_graph(chunks, doc)

    assert isinstance(result, GraphExtractionResult)
    assert result.skipped is True
    assert "LLM crash" in result.error
