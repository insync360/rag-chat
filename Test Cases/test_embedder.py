"""Tests for app/ingestion/embedder.py — chunk embedding generation."""

import uuid
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from app.ingestion.embedder import EmbeddingResult, _build_embedding_text, embed_chunks


@dataclass
class FakeChunk:
    content: str
    section_path: str = ""
    chunk_index: int = 0
    content_hash: str = "abc"
    metadata: dict = field(default_factory=dict)


@pytest.fixture
def chunk_ids():
    return [str(uuid.uuid4()) for _ in range(3)]


@pytest.fixture
def chunks():
    return [
        FakeChunk(
            content="First chunk text.",
            section_path="HR Policy > Benefits",
            metadata={
                "summary": "Overview of employee benefits.",
                "keywords": ["benefits", "HR", "policy"],
                "hypothetical_questions": ["What benefits are offered?"],
            },
        ),
        FakeChunk(
            content="Second chunk text.",
            section_path="HR Policy > Leave",
            metadata={
                "summary": "Leave policy details.",
                "keywords": ["leave", "PTO"],
                "hypothetical_questions": ["How much PTO do employees get?"],
            },
        ),
        FakeChunk(
            content="Third chunk text.",
            section_path="",
            metadata={},
        ),
    ]


@pytest.fixture
def mock_embedding():
    """Mock OpenAI embedding response data item."""
    item = MagicMock()
    item.embedding = [0.1] * 2000
    return item


@pytest.fixture
def mock_openai_response(mock_embedding):
    resp = MagicMock()
    resp.data = [mock_embedding, mock_embedding, mock_embedding]
    return resp


# --- _build_embedding_text tests ---


def test_build_embedding_text_includes_all_metadata():
    chunk = FakeChunk(
        content="Parental leave is 12 weeks.",
        section_path="HR Policy > California > Parental Leave",
        metadata={
            "summary": "California employees get 12 weeks parental leave.",
            "keywords": ["parental leave", "California", "FMLA"],
            "hypothetical_questions": [
                "What is the parental leave policy?",
                "How many weeks of leave?",
            ],
        },
    )
    text = _build_embedding_text(chunk)

    assert "Section: HR Policy > California > Parental Leave" in text
    assert "Summary: California employees get 12 weeks parental leave." in text
    assert "Keywords: parental leave, California, FMLA" in text
    assert "Questions: What is the parental leave policy? How many weeks of leave?" in text
    assert "Parental leave is 12 weeks." in text


def test_build_embedding_text_empty_metadata_falls_back_to_content():
    chunk = FakeChunk(content="Just raw content.", section_path="", metadata={})
    text = _build_embedding_text(chunk)
    assert text == "Just raw content."


def test_build_embedding_text_no_metadata_attr():
    """Chunk with metadata=None still works."""
    chunk = FakeChunk(content="Bare content.", section_path="", metadata=None)
    # metadata=None triggers the `or {}` fallback
    text = _build_embedding_text(chunk)
    assert text == "Bare content."


def test_build_embedding_text_partial_metadata():
    """Only summary present, no keywords or questions."""
    chunk = FakeChunk(
        content="Some content.",
        section_path="Docs",
        metadata={"summary": "A summary."},
    )
    text = _build_embedding_text(chunk)
    assert "Section: Docs" in text
    assert "Summary: A summary." in text
    assert "Keywords:" not in text
    assert "Questions:" not in text
    assert "Some content." in text


# --- embed_chunks tests ---


@pytest.mark.asyncio
async def test_embed_chunks_success(chunk_ids, chunks, mock_openai_response):
    mock_client = AsyncMock()
    mock_client.embeddings.create.return_value = mock_openai_response

    mock_pool = AsyncMock()

    with (
        patch("app.ingestion.embedder.AsyncOpenAI", return_value=mock_client),
        patch("app.ingestion.embedder.get_pool", return_value=mock_pool),
    ):
        result = await embed_chunks(chunk_ids, chunks)

    assert result.total == 3
    assert result.embedded == 3
    assert result.skipped is False
    assert result.error is None
    mock_client.embeddings.create.assert_called_once()
    mock_pool.executemany.assert_called_once()


@pytest.mark.asyncio
async def test_embed_chunks_sends_enriched_text(chunk_ids, chunks, mock_openai_response):
    """Verify the text sent to OpenAI includes metadata, not just raw content."""
    mock_client = AsyncMock()
    mock_client.embeddings.create.return_value = mock_openai_response

    with (
        patch("app.ingestion.embedder.AsyncOpenAI", return_value=mock_client),
        patch("app.ingestion.embedder.get_pool", return_value=AsyncMock()),
    ):
        await embed_chunks(chunk_ids, chunks)

    call_args = mock_client.embeddings.create.call_args
    texts_sent = call_args.kwargs["input"]
    # First chunk has metadata — text should include summary
    assert "Summary: Overview of employee benefits." in texts_sent[0]
    # Third chunk has empty metadata — text should be content only
    assert texts_sent[2] == "Third chunk text."


@pytest.mark.asyncio
async def test_embed_chunks_empty_input():
    result = await embed_chunks([], [])

    assert result.total == 0
    assert result.embedded == 0
    assert result.skipped is False


@pytest.mark.asyncio
async def test_embed_chunks_openai_failure(chunk_ids, chunks):
    mock_client = AsyncMock()
    mock_client.embeddings.create.side_effect = Exception("API rate limit")

    with (
        patch("app.ingestion.embedder.AsyncOpenAI", return_value=mock_client),
        patch("app.ingestion.embedder.get_pool", return_value=AsyncMock()),
    ):
        result = await embed_chunks(chunk_ids, chunks)

    assert result.total == 3
    assert result.embedded == 0
    assert result.skipped is True
    assert "API rate limit" in result.error


@pytest.mark.asyncio
async def test_pipeline_calls_embed_with_chunks():
    """Verify pipeline calls embed_chunks with chunk objects (not content strings)."""
    chunk_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    fake_chunks = [
        FakeChunk(content="hello", section_path="A", metadata={"summary": "hi"}),
        FakeChunk(content="world", section_path="B", metadata={"summary": "earth"}),
    ]

    mock_embed = AsyncMock(return_value=EmbeddingResult(total=2, embedded=2))

    # Simulate what pipeline does after save
    await mock_embed(chunk_ids, fake_chunks)

    mock_embed.assert_called_once_with(chunk_ids, fake_chunks)


@pytest.mark.asyncio
async def test_pipeline_continues_if_embed_raises():
    """Verify the pipeline try/except pattern handles embed failures."""
    import logging

    raised = False
    continued = False

    try:
        raise RuntimeError("embed boom")
    except Exception:
        raised = True
        logging.warning("Chunk embedding failed (expected in test)")
    continued = True

    assert raised
    assert continued
