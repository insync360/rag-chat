"""FastAPI server — HTTP layer for the RAG Chat backend."""

import dataclasses
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
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
    conversation_id: str | None = None


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


class ConversationOut(BaseModel):
    id: str
    title: str
    last_message: str
    message_count: int
    updated_at: str


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    metadata: dict | None
    created_at: str


class ConversationCreate(BaseModel):
    title: str | None = None


class ConversationUpdate(BaseModel):
    title: str


# ── Endpoints ────────────────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse)
async def api_query(body: QueryRequest):
    from app.retrieval import query

    pool = await get_pool()
    conversation_history: list[dict] | None = None

    # If conversation_id provided, persist user message and load history
    if body.conversation_id:
        cid = uuid.UUID(body.conversation_id)
        # Insert user message
        await pool.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES ($1, 'user', $2)",
            cid, body.query,
        )
        # Load prior messages (excluding the just-inserted one)
        rows = await pool.fetch(
            """SELECT role, content FROM messages
               WHERE conversation_id = $1
               ORDER BY created_at ASC""",
            cid,
        )
        # Exclude last row (the just-inserted user message)
        if len(rows) > 1:
            conversation_history = [{"role": r["role"], "content": r["content"]} for r in rows[:-1]]

    result = await query(body.query, conversation_history=conversation_history)

    # Persist assistant message if in a conversation
    if body.conversation_id:
        cid = uuid.UUID(body.conversation_id)
        meta = json.dumps({
            "query_type": result.query_type.value,
            "cached": result.cached,
            "chunk_count": len(result.chunks_used),
            "total_time": result.step_timings.get("total"),
        })
        await pool.execute(
            "INSERT INTO messages (conversation_id, role, content, metadata) VALUES ($1, 'assistant', $2, $3::jsonb)",
            cid, result.answer, meta,
        )
        await pool.execute(
            "UPDATE conversations SET updated_at = now() WHERE id = $1", cid,
        )

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


# ── Conversation endpoints ──────────────────────────────────────────

@app.get("/api/conversations", response_model=list[ConversationOut])
async def api_list_conversations():
    pool = await get_pool()
    rows = await pool.fetch(
        """SELECT c.id, c.title, c.updated_at,
                  COUNT(m.id) AS message_count,
                  COALESCE(
                      (SELECT content FROM messages
                       WHERE conversation_id = c.id ORDER BY created_at DESC LIMIT 1),
                      ''
                  ) AS last_message
           FROM conversations c
           LEFT JOIN messages m ON m.conversation_id = c.id
           GROUP BY c.id
           ORDER BY c.updated_at DESC"""
    )
    return [
        ConversationOut(
            id=str(r["id"]),
            title=r["title"],
            last_message=r["last_message"][:100],
            message_count=r["message_count"],
            updated_at=r["updated_at"].isoformat(),
        )
        for r in rows
    ]


@app.post("/api/conversations", response_model=ConversationOut)
async def api_create_conversation(body: ConversationCreate | None = None):
    pool = await get_pool()
    title = (body.title if body and body.title else "New Conversation")
    row = await pool.fetchrow(
        "INSERT INTO conversations (title) VALUES ($1) RETURNING id, title, created_at, updated_at",
        title,
    )
    return ConversationOut(
        id=str(row["id"]),
        title=row["title"],
        last_message="",
        message_count=0,
        updated_at=row["updated_at"].isoformat(),
    )


@app.get("/api/conversations/{conv_id}/messages", response_model=list[MessageOut])
async def api_get_messages(conv_id: str):
    pool = await get_pool()
    rows = await pool.fetch(
        """SELECT id, role, content, metadata, created_at
           FROM messages WHERE conversation_id = $1::uuid
           ORDER BY created_at ASC""",
        uuid.UUID(conv_id),
    )
    return [
        MessageOut(
            id=str(r["id"]),
            role=r["role"],
            content=r["content"],
            metadata=json.loads(r["metadata"]) if r["metadata"] else None,
            created_at=r["created_at"].isoformat(),
        )
        for r in rows
    ]


@app.patch("/api/conversations/{conv_id}")
async def api_update_conversation(conv_id: str, body: ConversationUpdate):
    pool = await get_pool()
    result = await pool.execute(
        "UPDATE conversations SET title = $1, updated_at = now() WHERE id = $2::uuid",
        body.title, uuid.UUID(conv_id),
    )
    if result == "UPDATE 0":
        raise HTTPException(404, "Conversation not found")
    return {"ok": True}


@app.delete("/api/conversations/{conv_id}", status_code=204)
async def api_delete_conversation(conv_id: str):
    pool = await get_pool()
    result = await pool.execute(
        "DELETE FROM conversations WHERE id = $1::uuid", uuid.UUID(conv_id),
    )
    if result == "DELETE 0":
        raise HTTPException(404, "Conversation not found")
    return Response(status_code=204)


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
