"""Unit tests for Neo4j graph store."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.graph.models import Entity, Relationship
from app.graph.store import store_graph, clear_document_graph, _batch


def _entity(name="Acme Corp", type="Organization", confidence=0.9):
    return Entity(
        name=name, type=type, source_chunk_index=0,
        source_document_id="doc-123", confidence=confidence, properties={},
    )


def _rel(source="Acme Corp", target="John Smith", type="EMPLOYS", confidence=0.9):
    return Relationship(
        source_entity=source, target_entity=target, type=type,
        source_chunk_index=0, source_document_id="doc-123",
        confidence=confidence, properties={},
    )


class _FakeSession:
    """Async context manager that delegates to an AsyncMock session."""
    def __init__(self, session):
        self._session = session
    async def __aenter__(self):
        return self._session
    async def __aexit__(self, *args):
        return False


@pytest.fixture
def mock_driver():
    session = AsyncMock()
    session.run = AsyncMock()
    driver = AsyncMock()
    driver.session = MagicMock(return_value=_FakeSession(session))
    with patch("app.graph.store.get_driver", new_callable=AsyncMock, return_value=driver):
        yield driver, session


# --- Tests ---


@pytest.mark.asyncio
async def test_store_entities(mock_driver):
    driver, session = mock_driver
    entities = [_entity("Acme Corp"), _entity("John Smith", type="Person")]

    await store_graph(entities, [], "doc-123")

    session.run.assert_called()
    call_args = session.run.call_args_list[0]
    assert "MERGE (n:Entity {name: e.name, type: e.type})" in call_args[0][0]


@pytest.mark.asyncio
async def test_store_relationships(mock_driver):
    driver, session = mock_driver
    entities = [_entity("Acme Corp"), _entity("John Smith", type="Person")]
    rels = [_rel()]

    await store_graph(entities, rels, "doc-123")

    # 1 call for entities + 1 call for relationships
    assert session.run.call_count == 2
    rel_call = session.run.call_args_list[1]
    assert "EMPLOYS" in rel_call[0][0]


@pytest.mark.asyncio
async def test_clear_document_graph(mock_driver):
    driver, session = mock_driver

    await clear_document_graph("doc-123")

    session.run.assert_called_once()
    assert "DETACH DELETE" in session.run.call_args[0][0]


def test_batching():
    items = list(range(10))
    batches = _batch(items, 3)

    assert len(batches) == 4
    assert batches[0] == [0, 1, 2]
    assert batches[-1] == [9]


@pytest.mark.asyncio
async def test_neo4j_down():
    with patch("app.graph.store.get_driver", new_callable=AsyncMock, side_effect=Exception("Connection refused")):
        with pytest.raises(Exception, match="Connection refused"):
            await store_graph([_entity()], [], "doc-123")
