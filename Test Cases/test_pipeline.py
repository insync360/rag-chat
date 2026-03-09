"""Unit tests for pipeline orchestrator."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.ingestion.chunker import Chunk
from app.ingestion.parser import ParsedDocument
from app.ingestion.pipeline import (
    PipelineResult,
    PipelineStatus,
    StepTimings,
    ingest_file,
    ingest_files,
)
from app.ingestion.version_tracker import DocumentRecord


# --- Helpers ---


def _parsed_doc(filename="test.pdf"):
    return ParsedDocument(
        filename=filename, content_hash="abc123hash", pages=[],
        full_markdown="# Test\nHello world", page_count=1,
        metadata={"tier": "agentic_plus"},
    )


def _doc_record(doc_id=None, version=1):
    return DocumentRecord(
        id=doc_id or str(uuid.uuid4()), filename="test.pdf",
        content_hash="abc123hash", version=version, status="active",
        page_count=1, ingested_at=datetime.now(timezone.utc), metadata={},
    )


def _chunks(document_id, count=3):
    return [
        Chunk(
            document_id=document_id, chunk_index=i,
            content=f"Chunk {i} content", token_count=10,
            section_path="Test", has_table=False, has_code=False,
            overlap_tokens=0, metadata={},
        )
        for i in range(count)
    ]


def _mock_pool():
    pool = AsyncMock()
    pool.fetchrow = AsyncMock(return_value={"id": uuid.uuid4()})
    pool.fetchval = AsyncMock(return_value=0)
    pool.execute = AsyncMock()
    return pool


# --- Fixtures ---


@pytest.fixture
def deps():
    """Mock all pipeline dependencies."""
    doc_id = str(uuid.uuid4())
    doc = _parsed_doc()
    record = _doc_record(doc_id)
    ch = _chunks(doc_id)
    chunk_ids = [str(uuid.uuid4()) for _ in ch]
    pool = _mock_pool()

    patches = {
        "parser": patch("app.ingestion.pipeline.LlamaParser"),
        "tracker": patch("app.ingestion.pipeline.VersionTracker"),
        "chunk_doc": patch("app.ingestion.pipeline.chunk_document", return_value=ch),
        "enrich": patch("app.ingestion.pipeline.enrich_chunks", new_callable=AsyncMock, return_value=ch),
        "save": patch("app.ingestion.pipeline.save_chunks", new_callable=AsyncMock, return_value=chunk_ids),
        "get_pool": patch("app.ingestion.pipeline.get_pool", new_callable=AsyncMock, return_value=pool),
        "sleep": patch("asyncio.sleep", new_callable=AsyncMock),
    }
    mocks = {name: p.start() for name, p in patches.items()}

    parser_inst = MagicMock()
    parser_inst.parse = AsyncMock(return_value=doc)
    mocks["parser"].return_value = parser_inst

    tracker_inst = MagicMock()
    tracker_inst.track = AsyncMock(return_value=(record, True))
    mocks["tracker"].return_value = tracker_inst

    yield {
        "mocks": mocks, "pool": pool, "doc_id": doc_id,
        "doc": doc, "record": record, "chunks": ch, "chunk_ids": chunk_ids,
        "parser": parser_inst, "tracker": tracker_inst,
    }

    for p in patches.values():
        p.stop()


# --- Tests ---


@pytest.mark.asyncio
async def test_success_full_pipeline(deps):
    result = await ingest_file("test.pdf")

    assert result.status == PipelineStatus.COMPLETED
    assert result.document_id == deps["doc_id"]
    assert result.chunk_count == len(deps["chunk_ids"])
    assert result.error is None
    assert result.step == "save"
    deps["parser"].parse.assert_called_once()
    deps["tracker"].track.assert_called_once()
    deps["mocks"]["chunk_doc"].assert_called_once()
    deps["mocks"]["enrich"].assert_called_once()
    deps["mocks"]["save"].assert_called_once()


@pytest.mark.asyncio
async def test_skipped_duplicate(deps):
    deps["tracker"].track.return_value = (deps["record"], False)
    deps["pool"].fetchval.return_value = 5  # chunks exist

    result = await ingest_file("test.pdf")

    assert result.status == PipelineStatus.SKIPPED
    assert result.chunk_count == 5
    deps["mocks"]["chunk_doc"].assert_not_called()
    deps["mocks"]["enrich"].assert_not_called()
    deps["mocks"]["save"].assert_not_called()


@pytest.mark.asyncio
async def test_orphan_resumes(deps):
    deps["tracker"].track.return_value = (deps["record"], False)
    deps["pool"].fetchval.return_value = 0  # no chunks — orphan

    result = await ingest_file("test.pdf")

    assert result.status == PipelineStatus.COMPLETED
    deps["mocks"]["chunk_doc"].assert_called_once()
    deps["mocks"]["enrich"].assert_called_once()
    deps["mocks"]["save"].assert_called_once()


@pytest.mark.asyncio
async def test_parse_failure_retries_then_succeeds(deps):
    deps["parser"].parse.side_effect = [
        Exception("timeout 1"),
        Exception("timeout 2"),
        deps["doc"],
    ]

    result = await ingest_file("test.pdf")

    assert result.status == PipelineStatus.COMPLETED
    assert deps["parser"].parse.call_count == 3
    # 3 log rows created (one per attempt)
    assert deps["pool"].fetchrow.call_count == 3


@pytest.mark.asyncio
async def test_all_retries_exhausted(deps):
    deps["parser"].parse.side_effect = Exception("always fails")

    result = await ingest_file("test.pdf")

    assert result.status == PipelineStatus.FAILED
    assert "always fails" in result.error
    assert result.step == "parse"
    assert deps["parser"].parse.call_count == 3


@pytest.mark.asyncio
async def test_batch_isolation(deps):
    """3 files: 1 completes, 1 skips, 1 fails — each independent."""
    records = [_doc_record(version=i + 1) for i in range(3)]

    async def mock_parse(path):
        if str(path).endswith("c.pdf"):
            raise Exception("file3 error")
        return deps["doc"]

    call_count = 0

    async def mock_track(doc):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            return records[1], False  # duplicate
        return records[0], True

    deps["parser"].parse.side_effect = mock_parse
    deps["tracker"].track.side_effect = mock_track
    deps["pool"].fetchval.return_value = 5  # chunks exist for duplicate

    result = await ingest_files(["a.pdf", "b.pdf", "c.pdf"])

    assert result.total == 3
    assert result.completed == 1
    assert result.skipped == 1
    assert result.failed == 1
    assert result.files[0].status == PipelineStatus.COMPLETED
    assert result.files[1].status == PipelineStatus.SKIPPED
    assert result.files[2].status == PipelineStatus.FAILED


@pytest.mark.asyncio
async def test_empty_list(deps):
    result = await ingest_files([])

    assert result.total == 0
    assert result.completed == 0
    assert result.skipped == 0
    assert result.failed == 0
    assert result.files == []
    assert result.total_duration_ms == 0


def test_step_timings():
    t = StepTimings(parse_ms=100.5, chunk_ms=50.2)
    d = t.to_dict()

    assert d["parse_ms"] == 100.5
    assert d["chunk_ms"] == 50.2
    assert "version_track_ms" not in d
    assert "enrich_ms" not in d
    assert "save_ms" not in d


def test_pipeline_result_counts():
    files = [
        MagicMock(status=PipelineStatus.COMPLETED),
        MagicMock(status=PipelineStatus.SKIPPED),
        MagicMock(status=PipelineStatus.FAILED),
        MagicMock(status=PipelineStatus.COMPLETED),
    ]
    result = PipelineResult(
        total=4, completed=2, skipped=1, failed=1,
        files=files, total_duration_ms=1000,
    )

    assert result.completed + result.skipped + result.failed == result.total
