"""Unit tests for incremental graph updates — change detection, selective extraction, deprecation."""

import hashlib
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.graph import extract_and_store_graph
from app.graph.models import Entity, GraphExtractionResult, Relationship
from app.ingestion.chunker import Chunk, chunk_document
from app.ingestion.version_tracker import DocumentRecord


def _doc_record(doc_id=None):
    return DocumentRecord(
        id=doc_id or str(uuid.uuid4()), filename="test.pdf",
        content_hash="abc123", version=1, status="active",
        page_count=1, ingested_at=datetime.now(timezone.utc), metadata={},
    )


def _chunks(doc_id, contents=None, count=3):
    contents = contents or [f"Chunk {i} content" for i in range(count)]
    return [
        Chunk(
            document_id=doc_id, chunk_index=i,
            content=c, token_count=10,
            section_path="Test", has_table=False, has_code=False,
            overlap_tokens=0, metadata={},
            content_hash=hashlib.sha256(c.encode()).hexdigest(),
        )
        for i, c in enumerate(contents)
    ]


def _entities(doc_id, names=None):
    names = names or ["Acme Corporation"]
    return [
        Entity(name=n, type="Organization", source_chunk_index=0,
               source_document_id=doc_id, confidence=0.9)
        for n in names
    ]


# --- Chunk hashing ---


def test_chunk_content_hash_computed():
    """Chunks get non-empty content_hash after chunk_document."""
    chunks = chunk_document("# Test\nHello world paragraph.", "doc-1")
    assert all(c.content_hash for c in chunks)
    assert all(len(c.content_hash) == 64 for c in chunks)  # SHA-256


def test_identical_content_same_hash():
    """Deterministic — same content produces same hash."""
    chunks_a = chunk_document("# Test\nSame content here.", "doc-a")
    chunks_b = chunk_document("# Test\nSame content here.", "doc-b")
    assert chunks_a[0].content_hash == chunks_b[0].content_hash


# --- Change detection ---


def test_change_detection_changed():
    """Detects changed chunks by hash diff."""
    doc_id = "doc-1"
    old_hashes = {0: "hash_a", 1: "hash_b", 2: "hash_c"}
    new_chunks = _chunks(doc_id, ["Chunk 0 changed", "Chunk 1 content", "Chunk 2 content"])
    # chunk 0 has different hash (content changed)
    # chunk 1 & 2 may or may not match depending on content

    changed = {
        i for i, c in enumerate(new_chunks)
        if old_hashes.get(i) != c.content_hash
    }
    # All should be changed since old_hashes are fake strings
    assert 0 in changed


def test_change_detection_new_chunks():
    """Detects added chunks (indices not in old hashes)."""
    old_hashes = {0: "hash_a", 1: "hash_b"}
    new_chunks = _chunks("doc-1", count=4)

    new_idx = {c.chunk_index for c in new_chunks} - set(old_hashes.keys())
    assert new_idx == {2, 3}


# --- Selective extraction ---


@pytest.mark.asyncio
async def test_selective_extraction_only_changed():
    """GPT-4o called only for changed chunks in incremental mode."""
    doc = _doc_record()
    old_doc = _doc_record()
    all_chunks = _chunks(doc.id, count=5)
    changed = {1, 3}  # only chunks 1 and 3 changed
    ents = _entities(doc.id)

    with patch("app.graph.extract_from_chunks", new_callable=AsyncMock,
               return_value=(ents, [])) as mock_extract, \
         patch("app.graph.deduplicate_entities_enhanced", new_callable=AsyncMock,
               return_value=ents), \
         patch("app.graph.deduplicate_relationships", return_value=[]), \
         patch("app.graph.ensure_schema", new_callable=AsyncMock), \
         patch("app.graph.deprecate_chunk_entities", new_callable=AsyncMock), \
         patch("app.graph.get_document_entities", new_callable=AsyncMock, return_value=[]), \
         patch("app.graph.store_graph", new_callable=AsyncMock), \
         patch("app.graph.settings") as mock_settings:
        mock_settings.GRAPH_EXTRACTION_ENABLED = True
        mock_settings.INCREMENTAL_GRAPH_ENABLED = True
        mock_settings.COREF_ENABLED = False

        await extract_and_store_graph(
            all_chunks, doc,
            old_document_id=old_doc.id, changed_indices=changed,
        )

    # extract_from_chunks should receive only 2 chunks (indices 1 and 3)
    called_chunks = mock_extract.call_args[0][0]
    assert len(called_chunks) == 2
    assert {c.chunk_index for c in called_chunks} == {1, 3}


@pytest.mark.asyncio
async def test_first_ingestion_full_extraction():
    """No old_document_id → all chunks extracted."""
    doc = _doc_record()
    chunks = _chunks(doc.id, count=3)
    ents = _entities(doc.id)

    with patch("app.graph.extract_from_chunks", new_callable=AsyncMock,
               return_value=(ents, [])) as mock_extract, \
         patch("app.graph.deduplicate_entities_enhanced", new_callable=AsyncMock,
               return_value=ents), \
         patch("app.graph.deduplicate_relationships", return_value=[]), \
         patch("app.graph.ensure_schema", new_callable=AsyncMock), \
         patch("app.graph.clear_document_graph", new_callable=AsyncMock), \
         patch("app.graph.store_graph", new_callable=AsyncMock), \
         patch("app.graph.settings") as mock_settings:
        mock_settings.GRAPH_EXTRACTION_ENABLED = True
        mock_settings.INCREMENTAL_GRAPH_ENABLED = True
        mock_settings.COREF_ENABLED = False

        await extract_and_store_graph(chunks, doc)

    called_chunks = mock_extract.call_args[0][0]
    assert len(called_chunks) == 3


# --- Feature flag ---


@pytest.mark.asyncio
async def test_feature_flag_disabled():
    """INCREMENTAL_GRAPH_ENABLED=False → full extraction even with changed_indices."""
    doc = _doc_record()
    old_doc = _doc_record()
    chunks = _chunks(doc.id, count=3)
    ents = _entities(doc.id)

    with patch("app.graph.extract_from_chunks", new_callable=AsyncMock,
               return_value=(ents, [])) as mock_extract, \
         patch("app.graph.deduplicate_entities_enhanced", new_callable=AsyncMock,
               return_value=ents), \
         patch("app.graph.deduplicate_relationships", return_value=[]), \
         patch("app.graph.ensure_schema", new_callable=AsyncMock), \
         patch("app.graph.clear_document_graph", new_callable=AsyncMock) as mock_clear, \
         patch("app.graph.store_graph", new_callable=AsyncMock), \
         patch("app.graph.settings") as mock_settings:
        mock_settings.GRAPH_EXTRACTION_ENABLED = True
        mock_settings.INCREMENTAL_GRAPH_ENABLED = False
        mock_settings.COREF_ENABLED = False

        await extract_and_store_graph(
            chunks, doc,
            old_document_id=old_doc.id, changed_indices={0, 1},
        )

    # All chunks should be extracted (flag disabled)
    called_chunks = mock_extract.call_args[0][0]
    assert len(called_chunks) == 3
    # Full clear, not deprecation
    mock_clear.assert_called_once()


# --- Deprecation ---


@pytest.mark.asyncio
async def test_deprecate_marks_old_entities():
    """Old entities from changed chunks get deprecated."""
    doc = _doc_record()
    old_doc = _doc_record()
    chunks = _chunks(doc.id, count=2)
    changed = {0}
    ents = _entities(doc.id)

    with patch("app.graph.extract_from_chunks", new_callable=AsyncMock,
               return_value=(ents, [])), \
         patch("app.graph.deduplicate_entities_enhanced", new_callable=AsyncMock,
               return_value=ents), \
         patch("app.graph.deduplicate_relationships", return_value=[]), \
         patch("app.graph.ensure_schema", new_callable=AsyncMock), \
         patch("app.graph.deprecate_chunk_entities", new_callable=AsyncMock) as mock_deprecate, \
         patch("app.graph.get_document_entities", new_callable=AsyncMock, return_value=[]), \
         patch("app.graph.store_graph", new_callable=AsyncMock), \
         patch("app.graph.settings") as mock_settings:
        mock_settings.GRAPH_EXTRACTION_ENABLED = True
        mock_settings.INCREMENTAL_GRAPH_ENABLED = True
        mock_settings.COREF_ENABLED = False

        await extract_and_store_graph(
            chunks, doc,
            old_document_id=old_doc.id, changed_indices=changed,
        )

    mock_deprecate.assert_called_once_with(old_doc.id, changed)


@pytest.mark.asyncio
async def test_new_entities_stored_as_active():
    """New entities from store_graph have status='active' in the Cypher."""
    from app.graph.store import store_graph

    mock_session = AsyncMock()
    mock_session.run = AsyncMock()

    class _FakeSession:
        def __init__(self, s): self._s = s
        async def __aenter__(self): return self._s
        async def __aexit__(self, *a): return False

    mock_driver = AsyncMock()
    mock_driver.session = MagicMock(return_value=_FakeSession(mock_session))

    with patch("app.graph.store.get_driver", new_callable=AsyncMock,
               return_value=mock_driver):
        await store_graph(_entities("doc-1"), [], "doc-1")

    cypher = mock_session.run.call_args[0][0]
    assert "n.status = 'active'" in cypher


@pytest.mark.asyncio
async def test_source_document_ids_array():
    """store_graph uses source_document_ids array, not singular."""
    from app.graph.store import store_graph

    mock_session = AsyncMock()
    mock_session.run = AsyncMock()

    class _FakeSession:
        def __init__(self, s): self._s = s
        async def __aenter__(self): return self._s
        async def __aexit__(self, *a): return False

    mock_driver = AsyncMock()
    mock_driver.session = MagicMock(return_value=_FakeSession(mock_session))

    with patch("app.graph.store.get_driver", new_callable=AsyncMock,
               return_value=mock_driver):
        await store_graph(_entities("doc-1"), [], "doc-1")

    cypher = mock_session.run.call_args[0][0]
    assert "source_document_ids" in cypher
    assert "[e.doc_id]" in cypher


@pytest.mark.asyncio
async def test_deprecate_does_not_delete():
    """Deprecated entities still exist in graph — no DETACH DELETE."""
    from app.graph.store import deprecate_chunk_entities

    mock_session = AsyncMock()
    mock_session.run = AsyncMock()

    class _FakeSession:
        def __init__(self, s): self._s = s
        async def __aenter__(self): return self._s
        async def __aexit__(self, *a): return False

    mock_driver = AsyncMock()
    mock_driver.session = MagicMock(return_value=_FakeSession(mock_session))

    with patch("app.graph.store.get_driver", new_callable=AsyncMock,
               return_value=mock_driver):
        await deprecate_chunk_entities("doc-1", {0, 1})

    cypher = mock_session.run.call_args[0][0]
    assert "DETACH DELETE" not in cypher
    assert "deprecated" in cypher
    assert "SET" in cypher


@pytest.mark.asyncio
async def test_original_content_never_deleted():
    """clear_document_graph uses deprecation, not DETACH DELETE."""
    from app.graph.store import clear_document_graph

    mock_session = AsyncMock()
    mock_session.run = AsyncMock()

    class _FakeSession:
        def __init__(self, s): self._s = s
        async def __aenter__(self): return self._s
        async def __aexit__(self, *a): return False

    mock_driver = AsyncMock()
    mock_driver.session = MagicMock(return_value=_FakeSession(mock_session))

    with patch("app.graph.store.get_driver", new_callable=AsyncMock,
               return_value=mock_driver):
        await clear_document_graph("doc-1")

    cypher = mock_session.run.call_args[0][0]
    assert "DETACH DELETE" not in cypher
    assert "deprecated" in cypher
