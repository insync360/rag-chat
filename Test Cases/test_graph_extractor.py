"""Unit tests for graph entity/relationship extraction."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.graph.extractor import extract_from_chunks, _extract_single
from app.graph.models import Entity, Relationship
from app.ingestion.chunker import Chunk


def _chunk(index=0, content="Acme Corporation hired John Smith as CEO in New York.", token_count=20):
    return Chunk(
        document_id="doc-123", chunk_index=index,
        content=content, token_count=token_count,
        section_path="Test", has_table=False, has_code=False,
        overlap_tokens=0, metadata={},
    )


def _llm_response(entities=None, relationships=None):
    """Build a mock OpenAI response with graph extraction JSON."""
    data = {
        "entities": entities or [
            {"name": "Acme Corporation", "type": "Organization", "confidence": 0.95, "properties": {}},
            {"name": "John Smith", "type": "Person", "confidence": 0.9, "properties": {"role": "CEO"}},
            {"name": "New York", "type": "Location", "confidence": 0.85, "properties": {}},
        ],
        "relationships": relationships or [
            {"source": "Acme Corporation", "target": "John Smith", "type": "EMPLOYS", "confidence": 0.9, "properties": {}},
            {"source": "John Smith", "target": "New York", "type": "LOCATED_IN", "confidence": 0.8, "properties": {}},
        ],
    }
    message = MagicMock()
    message.content = json.dumps(data)
    choice = MagicMock()
    choice.message = message
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.fixture
def mock_openai():
    with patch("app.graph.extractor.AsyncOpenAI") as mock_cls:
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(return_value=_llm_response())
        mock_cls.return_value = client
        yield client


# --- Tests ---


@pytest.mark.asyncio
async def test_extract_success(mock_openai):
    chunks = [_chunk(0), _chunk(1)]
    entities, relationships = await extract_from_chunks(chunks, "doc-123")

    assert len(entities) == 6  # 3 per chunk x 2 chunks
    assert len(relationships) == 4  # 2 per chunk x 2 chunks
    assert all(isinstance(e, Entity) for e in entities)
    assert all(isinstance(r, Relationship) for r in relationships)
    assert entities[0].name == "Acme Corporation"
    assert entities[0].type == "Organization"
    assert relationships[0].type == "EMPLOYS"


@pytest.mark.asyncio
async def test_skip_tiny_chunks(mock_openai):
    tiny = _chunk(content="Hi", token_count=2)
    entities, relationships = await extract_from_chunks([tiny], "doc-123")

    assert entities == []
    assert relationships == []
    mock_openai.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_failure_returns_empty(mock_openai):
    mock_openai.chat.completions.create.side_effect = Exception("API error")

    with patch("app.graph.extractor.asyncio.sleep", new_callable=AsyncMock):
        entities, relationships = await extract_from_chunks([_chunk()], "doc-123")

    assert entities == []
    assert relationships == []


@pytest.mark.asyncio
async def test_invalid_json_retries(mock_openai):
    bad_msg = MagicMock()
    bad_msg.content = "not json"
    bad_choice = MagicMock()
    bad_choice.message = bad_msg
    bad_resp = MagicMock()
    bad_resp.choices = [bad_choice]

    mock_openai.chat.completions.create.side_effect = [
        bad_resp, bad_resp, _llm_response(),
    ]

    with patch("app.graph.extractor.asyncio.sleep", new_callable=AsyncMock):
        entities, relationships = await extract_from_chunks([_chunk()], "doc-123")

    assert len(entities) == 3
    assert mock_openai.chat.completions.create.call_count == 3


@pytest.mark.asyncio
async def test_concurrency_limit(mock_openai):
    """Verify semaphore limits concurrent calls."""
    call_times = []

    async def slow_create(**kwargs):
        call_times.append(asyncio.get_event_loop().time())
        await asyncio.sleep(0.01)
        return _llm_response()

    mock_openai.chat.completions.create.side_effect = slow_create

    chunks = [_chunk(i) for i in range(6)]
    with patch("app.graph.extractor.settings") as mock_settings:
        mock_settings.GRAPH_EXTRACTION_CONCURRENCY = 2
        mock_settings.GRAPH_EXTRACTION_MODEL = "gpt-4o"
        mock_settings.OPENAI_API_KEY = "test-key"
        entities, _ = await extract_from_chunks(chunks, "doc-123")

    assert len(entities) == 18  # 3 per chunk x 6 chunks


@pytest.mark.asyncio
async def test_freeform_entity_types(mock_openai):
    """Verify non-standard entity types pass through."""
    resp = _llm_response(
        entities=[{"name": "GDPR", "type": "Regulation", "confidence": 0.95, "properties": {}}],
        relationships=[],
    )
    mock_openai.chat.completions.create.return_value = resp

    entities, _ = await extract_from_chunks([_chunk()], "doc-123")

    assert entities[0].type == "Regulation"
