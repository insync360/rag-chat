"""Pipeline orchestrator — single entry point for document ingestion."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from app.database import get_pool
from app.graph import extract_and_store_graph
from app.ingestion.chunker import chunk_document, save_chunks
from app.ingestion.enricher import enrich_chunks
from app.ingestion.parser import LlamaParser
from app.ingestion.version_tracker import VersionTracker

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


class PipelineStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepTimings:
    parse_ms: float = 0
    version_track_ms: float = 0
    chunk_ms: float = 0
    enrich_ms: float = 0
    graph_extract_ms: float = 0
    save_ms: float = 0

    def to_dict(self) -> dict:
        return {k: v for k, v in {
            "parse_ms": self.parse_ms,
            "version_track_ms": self.version_track_ms,
            "chunk_ms": self.chunk_ms,
            "enrich_ms": self.enrich_ms,
            "graph_extract_ms": self.graph_extract_ms,
            "save_ms": self.save_ms,
        }.items() if v > 0}


@dataclass
class FileResult:
    filename: str
    status: PipelineStatus
    document_id: str | None = None
    version: int | None = None
    chunk_count: int | None = None
    entity_count: int | None = None
    relationship_count: int | None = None
    error: str | None = None
    step: str | None = None
    timings: StepTimings = field(default_factory=StepTimings)
    log_id: str | None = None


@dataclass
class PipelineResult:
    total: int
    completed: int
    skipped: int
    failed: int
    files: list[FileResult]
    total_duration_ms: float


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


async def _log_create(pool, filename: str, attempt: int) -> str:
    row = await pool.fetchrow(
        "INSERT INTO ingestion_logs (filename, status, attempt) "
        "VALUES ($1, 'pending', $2) RETURNING id",
        filename, attempt,
    )
    return str(row["id"])


async def _log_update(pool, log_id: str, *, status: str, step: str | None = None,
                      document_id: str | None = None, chunk_count: int | None = None,
                      error_message: str | None = None, step_timings: dict | None = None,
                      completed: bool = False) -> None:
    await pool.execute(
        "UPDATE ingestion_logs SET status = $2, step = $3, "
        "document_id = $4::uuid, chunk_count = $5, error_message = $6, "
        "step_timings = $7::jsonb, "
        "completed_at = CASE WHEN $8 THEN now() ELSE completed_at END "
        "WHERE id = $1::uuid",
        log_id, status, step, document_id, chunk_count, error_message,
        json.dumps(step_timings or {}), completed,
    )


async def _ingest_one(
    file_path: str | Path, parser: LlamaParser,
    tracker: VersionTracker, pool,
) -> FileResult:
    """Process a single file through all pipeline steps with retry logic."""
    file_path = Path(file_path)
    filename = file_path.name
    last_error = None
    step = "parse"
    timings = StepTimings()
    log_id = None

    for attempt in range(1, MAX_RETRIES + 1):
        timings = StepTimings()
        step = "parse"
        log_id = await _log_create(pool, filename, attempt)

        try:
            # 1. Parse
            t = time.perf_counter()
            doc = await parser.parse(file_path)
            timings.parse_ms = _ms_since(t)

            # 2. Version track
            step = "version_track"
            t = time.perf_counter()
            record, is_new = await tracker.track(doc)
            timings.version_track_ms = _ms_since(t)

            if not is_new:
                chunk_count = await pool.fetchval(
                    "SELECT COUNT(*) FROM chunks WHERE document_id = $1::uuid",
                    record.id,
                )
                if chunk_count > 0:
                    # True duplicate — skip
                    await _log_update(
                        pool, log_id, status="skipped", step="version_track",
                        document_id=record.id, chunk_count=chunk_count,
                        step_timings=timings.to_dict(), completed=True,
                    )
                    return FileResult(
                        filename=filename, status=PipelineStatus.SKIPPED,
                        document_id=record.id, version=record.version,
                        chunk_count=chunk_count, step="version_track",
                        timings=timings, log_id=log_id,
                    )
                # Orphan — proceed (prior failure between version_track and save)
                logger.info("Orphan detected for %s, resuming pipeline", filename)

            # 3. Chunk
            step = "chunk"
            t = time.perf_counter()
            chunks = chunk_document(doc.full_markdown, record.id)
            timings.chunk_ms = _ms_since(t)

            # 4. Enrich
            step = "enrich"
            t = time.perf_counter()
            chunks = await enrich_chunks(chunks, record)
            timings.enrich_ms = _ms_since(t)

            # 5. Graph extract (non-blocking)
            step = "graph_extract"
            t = time.perf_counter()
            try:
                graph_result = await extract_and_store_graph(chunks, record)
            except Exception as graph_exc:
                logger.warning("Graph extraction failed for %s: %s", filename, graph_exc)
                graph_result = None
            timings.graph_extract_ms = _ms_since(t)

            entity_count = graph_result.entity_count if graph_result else 0
            relationship_count = graph_result.relationship_count if graph_result else 0

            # 6. Save
            step = "save"
            t = time.perf_counter()
            chunk_ids = await save_chunks(chunks)
            timings.save_ms = _ms_since(t)

            await _log_update(
                pool, log_id, status="completed", step="save",
                document_id=record.id, chunk_count=len(chunk_ids),
                step_timings=timings.to_dict(), completed=True,
            )
            return FileResult(
                filename=filename, status=PipelineStatus.COMPLETED,
                document_id=record.id, version=record.version,
                chunk_count=len(chunk_ids),
                entity_count=entity_count,
                relationship_count=relationship_count,
                step="save", timings=timings, log_id=log_id,
            )

        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "Attempt %d/%d failed for %s at step '%s': %s",
                attempt, MAX_RETRIES, filename, step, exc,
            )
            await _log_update(
                pool, log_id, status="failed", step=step,
                error_message=last_error, step_timings=timings.to_dict(),
                completed=True,
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2 ** attempt)

    return FileResult(
        filename=filename, status=PipelineStatus.FAILED,
        error=last_error, step=step,
        timings=timings, log_id=log_id,
    )


async def ingest_files(file_paths: list[str | Path]) -> PipelineResult:
    """Ingest multiple files sequentially. Returns aggregate result."""
    if not file_paths:
        return PipelineResult(
            total=0, completed=0, skipped=0, failed=0,
            files=[], total_duration_ms=0,
        )

    start = time.perf_counter()
    parser = LlamaParser()
    tracker = VersionTracker()
    pool = await get_pool()

    files: list[FileResult] = []
    for fp in file_paths:
        result = await _ingest_one(fp, parser, tracker, pool)
        files.append(result)

    total_ms = _ms_since(start)
    return PipelineResult(
        total=len(files),
        completed=sum(1 for f in files if f.status == PipelineStatus.COMPLETED),
        skipped=sum(1 for f in files if f.status == PipelineStatus.SKIPPED),
        failed=sum(1 for f in files if f.status == PipelineStatus.FAILED),
        files=files,
        total_duration_ms=total_ms,
    )


async def ingest_file(file_path: str | Path) -> FileResult:
    """Convenience wrapper: ingest a single file."""
    result = await ingest_files([file_path])
    return result.files[0]
