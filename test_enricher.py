"""Tests for metadata enricher — unit + integration."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.ingestion.chunker import Chunk
from app.ingestion.enricher import (
    ChunkType,
    _classify_chunk_type,
    _compute_freshness_score,
    enrich_chunks,
)
from app.ingestion.version_tracker import DocumentRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(**overrides) -> Chunk:
    defaults = dict(
        document_id="test-doc-id",
        chunk_index=0,
        content="This is a test paragraph with enough content to be meaningful.",
        token_count=15,
        section_path="Section A",
        has_table=False,
        has_code=False,
        overlap_tokens=0,
        metadata={},
    )
    defaults.update(overrides)
    return Chunk(**defaults)


def _make_doc_record(**overrides) -> DocumentRecord:
    defaults = dict(
        id="doc-uuid-1234",
        filename="test.pdf",
        content_hash="abc123def456",
        version=1,
        status="active",
        page_count=5,
        ingested_at=datetime.now(timezone.utc),
        metadata={},
    )
    defaults.update(overrides)
    return DocumentRecord(**defaults)


# ---------------------------------------------------------------------------
# Unit: _classify_chunk_type
# ---------------------------------------------------------------------------

class TestClassifyChunkType:
    def test_table(self):
        chunk = _make_chunk(has_table=True)
        assert _classify_chunk_type(chunk) == ChunkType.TABLE

    def test_code(self):
        chunk = _make_chunk(has_code=True)
        assert _classify_chunk_type(chunk) == ChunkType.CODE

    def test_heading(self):
        chunk = _make_chunk(content="# Introduction")
        assert _classify_chunk_type(chunk) == ChunkType.HEADING

    def test_list(self):
        chunk = _make_chunk(content="- Item one\n- Item two\n- Item three\n- Item four")
        assert _classify_chunk_type(chunk) == ChunkType.LIST

    def test_numbered_list(self):
        chunk = _make_chunk(content="1. First\n2. Second\n3. Third\n4. Fourth")
        assert _classify_chunk_type(chunk) == ChunkType.LIST

    def test_paragraph(self):
        chunk = _make_chunk(content="This is a regular paragraph with multiple sentences.")
        assert _classify_chunk_type(chunk) == ChunkType.PARAGRAPH

    def test_table_takes_priority_over_code(self):
        chunk = _make_chunk(has_table=True, has_code=True)
        assert _classify_chunk_type(chunk) == ChunkType.TABLE


# ---------------------------------------------------------------------------
# Unit: _compute_freshness_score
# ---------------------------------------------------------------------------

class TestComputeFreshnessScore:
    def test_just_now(self):
        score = _compute_freshness_score(datetime.now(timezone.utc))
        assert score >= 0.99

    def test_half_year(self):
        dt = datetime.now(timezone.utc) - timedelta(days=182)
        score = _compute_freshness_score(dt)
        assert 0.49 <= score <= 0.51

    def test_one_year(self):
        dt = datetime.now(timezone.utc) - timedelta(days=365)
        score = _compute_freshness_score(dt)
        assert score == 0.0

    def test_older_than_year(self):
        dt = datetime.now(timezone.utc) - timedelta(days=500)
        score = _compute_freshness_score(dt)
        assert score == 0.0

    def test_naive_datetime(self):
        dt = datetime.now() - timedelta(days=10)
        score = _compute_freshness_score(dt)
        assert 0.9 <= score <= 1.0


# ---------------------------------------------------------------------------
# Integration: enrich_chunks (mocked OpenAI)
# ---------------------------------------------------------------------------

_MOCK_LLM_RESPONSE = json.dumps({
    "summary": "This chunk discusses test content for validation purposes.",
    "keywords": ["test", "validation", "content"],
    "hypothetical_questions": ["What is test content?", "How is validation done?"],
    "entities": {
        "people": [],
        "organizations": ["Acme Corp"],
        "dates": ["2026-03-07"],
        "money": [],
    },
})


def _make_mock_response():
    mock_choice = MagicMock()
    mock_choice.message.content = _MOCK_LLM_RESPONSE
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    return mock_resp


def _mock_openai_client():
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(side_effect=lambda **kwargs: _make_mock_response())
    return client


class TestEnrichChunks:
    def test_enrich_populates_all_fields(self):
        chunk = _make_chunk()
        doc = _make_doc_record()

        with patch("app.ingestion.enricher.AsyncOpenAI", return_value=_mock_openai_client()):
            result = asyncio.run(enrich_chunks([chunk], doc))

        assert len(result) == 1
        meta = result[0].metadata
        assert meta["summary"] != ""
        assert len(meta["keywords"]) > 0
        assert len(meta["hypothetical_questions"]) > 0
        assert "entities" in meta
        assert meta["chunk_type"] == "PARAGRAPH"
        assert meta["document_id"] == doc.id
        assert meta["version"] == 1
        assert meta["content_hash"] == doc.content_hash
        assert "enriched_at" in meta
        assert "ingested_at" in meta
        assert 0.0 <= meta["freshness_score"] <= 1.0

    def test_tiny_chunk_skips_llm(self):
        chunk = _make_chunk(content="Hi", token_count=1)
        doc = _make_doc_record()
        mock_client = _mock_openai_client()

        with patch("app.ingestion.enricher.AsyncOpenAI", return_value=mock_client):
            result = asyncio.run(enrich_chunks([chunk], doc))

        mock_client.chat.completions.create.assert_not_called()
        meta = result[0].metadata
        assert meta["chunk_type"] == "PARAGRAPH"
        assert "summary" not in meta

    def test_empty_list(self):
        doc = _make_doc_record()
        result = asyncio.run(enrich_chunks([], doc))
        assert result == []

    def test_multiple_chunks(self):
        chunks = [
            _make_chunk(chunk_index=0, content="The first chunk contains detailed information about employee benefits and retirement plans for the fiscal year."),
            _make_chunk(chunk_index=1, content="The second chunk describes the company organizational structure including department hierarchies and reporting lines."),
        ]
        doc = _make_doc_record()

        with patch("app.ingestion.enricher.AsyncOpenAI", return_value=_mock_openai_client()):
            result = asyncio.run(enrich_chunks(chunks, doc))

        assert len(result) == 2
        assert all(r.metadata.get("summary") for r in result)

    def test_api_failure_uses_fallback(self):
        chunk = _make_chunk()
        doc = _make_doc_record()

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("API down"))

        with patch("app.ingestion.enricher.AsyncOpenAI", return_value=mock_client):
            result = asyncio.run(enrich_chunks([chunk], doc))

        meta = result[0].metadata
        # Fallback: lineage fields present, LLM fields empty/default
        assert meta["chunk_type"] == "PARAGRAPH"
        assert meta["document_id"] == doc.id
        assert meta["summary"] == ""
        assert meta["keywords"] == []
