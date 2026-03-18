"""FastAPI server — HTTP layer for the RAG Chat backend."""

import dataclasses
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.database import close_pool, get_pool
from app.graph.neo4j_client import close_driver
from app.retrieval.models import QueryType

logger = logging.getLogger(__name__)

UPLOADS_DIR = Path("uploads")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB pool + ensure uploads dir. Shutdown: close connections."""
    UPLOADS_DIR.mkdir(exist_ok=True)
    await get_pool()
    logger.info("Database pool initialised")
    yield
    await close_pool()
    await close_driver()
    logger.info("Connections closed")


app = FastAPI(title="RAG Chat API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str


class ChunkOut(BaseModel):
    chunk_id: str
    content: str
    score: float
    section_path: str
    filename: str
    source: str


class QueryResponse(BaseModel):
    answer: str
    chunks_used: list[ChunkOut]
    query_type: str
    cached: bool
    step_timings: dict
    error: str | None


class DocumentOut(BaseModel):
    id: str
    filename: str
    version: int
    page_count: int | None
    ingested_at: str
    chunk_count: int


class FileResultOut(BaseModel):
    filename: str
    status: str
    error: str | None
    chunk_count: int | None


class UploadResponse(BaseModel):
    total: int
    completed: int
    skipped: int
    failed: int
    files: list[FileResultOut]


# ── Endpoints ────────────────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse)
async def api_query(body: QueryRequest):
    from app.retrieval import query

    result = await query(body.query)

    chunks = [
        ChunkOut(
            chunk_id=c.chunk_id,
            content=c.content,
            score=round(c.score, 4),
            section_path=c.section_path,
            filename=c.filename,
            source=c.source,
        )
        for c in result.chunks_used
    ]

    return QueryResponse(
        answer=result.answer,
        chunks_used=chunks,
        query_type=result.query_type.value,
        cached=result.cached,
        step_timings=result.step_timings,
        error=result.error,
    )


@app.post("/api/upload", response_model=UploadResponse)
async def api_upload(files: list[UploadFile]):
    from app.ingestion.pipeline import ingest_files

    if not files:
        raise HTTPException(400, "No files provided")

    saved_paths: list[Path] = []
    for f in files:
        if not f.filename or not f.filename.lower().endswith(".pdf"):
            raise HTTPException(400, f"Only PDF files are accepted: {f.filename}")

        dest = UPLOADS_DIR / f"{uuid.uuid4().hex[:8]}_{f.filename}"
        content = await f.read()
        dest.write_bytes(content)
        saved_paths.append(dest)

    try:
        result = await ingest_files(saved_paths)
    finally:
        for p in saved_paths:
            p.unlink(missing_ok=True)

    return UploadResponse(
        total=result.total,
        completed=result.completed,
        skipped=result.skipped,
        failed=result.failed,
        files=[
            FileResultOut(
                filename=fr.filename,
                status=fr.status.value,
                error=fr.error,
                chunk_count=fr.chunk_count,
            )
            for fr in result.files
        ],
    )


@app.get("/api/documents", response_model=list[DocumentOut])
async def api_documents():
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT d.id, d.filename, d.version, d.page_count, d.ingested_at,
               COUNT(c.id) AS chunk_count
        FROM documents d
        LEFT JOIN chunks c ON c.document_id = d.id
        WHERE d.status = 'active'
        GROUP BY d.id
        ORDER BY d.ingested_at DESC
        """
    )
    return [
        DocumentOut(
            id=str(r["id"]),
            filename=r["filename"],
            version=r["version"],
            page_count=r["page_count"],
            ingested_at=r["ingested_at"].isoformat() if r["ingested_at"] else "",
            chunk_count=r["chunk_count"],
        )
        for r in rows
    ]


@app.delete("/api/documents/{doc_id}", status_code=204)
async def api_delete_document(doc_id: str):
    pool = await get_pool()
    result = await pool.execute(
        """
        UPDATE documents
        SET status = 'deprecated', deprecated_at = now()
        WHERE id = $1::uuid AND status = 'active'
        """,
        uuid.UUID(doc_id),
    )
    if result == "UPDATE 0":
        raise HTTPException(404, "Document not found or already deprecated")
