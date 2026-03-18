"""FastAPI server — HTTP layer for the RAG Chat backend."""

import dataclasses
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
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
    category_id: str | None
    category_name: str | None


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


class CategoryOut(BaseModel):
    id: str
    name: str
    document_count: int


class CategoryCreate(BaseModel):
    name: str


class AgentCreate(BaseModel):
    name: str
    description: str = ""
    system_prompt: str = ""
    category_ids: list[str] = []


class AgentUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    system_prompt: str | None = None
    is_active: bool | None = None
    category_ids: list[str] | None = None


class AgentCategoryOut(BaseModel):
    id: str
    name: str


class AgentOut(BaseModel):
    id: str
    name: str
    description: str
    system_prompt: str
    model: str
    is_active: bool
    categories: list[AgentCategoryOut]
    endpoint: str


class AgentChatRequest(BaseModel):
    query: str
    conversation_id: str | None = None


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
async def api_upload(files: list[UploadFile], category_id: str = Form(...)):
    from app.ingestion.pipeline import ingest_files

    if not files:
        raise HTTPException(400, "No files provided")

    pool = await get_pool()
    cat_row = await pool.fetchrow(
        "SELECT id FROM categories WHERE id = $1::uuid", uuid.UUID(category_id),
    )
    if not cat_row:
        raise HTTPException(400, "Category not found")

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

    # Stamp category on completed documents
    for fr in result.files:
        if fr.status.value == "completed" and fr.document_id:
            await pool.execute(
                "UPDATE documents SET category_id = $1::uuid WHERE id = $2::uuid",
                uuid.UUID(category_id), uuid.UUID(fr.document_id),
            )

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
               d.category_id, cat.name AS category_name,
               COUNT(c.id) AS chunk_count
        FROM documents d
        LEFT JOIN chunks c ON c.document_id = d.id
        LEFT JOIN categories cat ON cat.id = d.category_id
        WHERE d.status = 'active'
        GROUP BY d.id, cat.name
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
            category_id=str(r["category_id"]) if r["category_id"] else None,
            category_name=r["category_name"],
        )
        for r in rows
    ]


# ── Category endpoints ─────────────────────────────────────────────

@app.get("/api/categories", response_model=list[CategoryOut])
async def api_list_categories():
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT cat.id, cat.name,
               COUNT(d.id) FILTER (WHERE d.status = 'active') AS document_count
        FROM categories cat
        LEFT JOIN documents d ON d.category_id = cat.id
        GROUP BY cat.id
        ORDER BY cat.name
        """
    )
    return [
        CategoryOut(
            id=str(r["id"]),
            name=r["name"],
            document_count=r["document_count"],
        )
        for r in rows
    ]


@app.post("/api/categories", response_model=CategoryOut, status_code=201)
async def api_create_category(body: CategoryCreate):
    pool = await get_pool()
    existing = await pool.fetchrow(
        "SELECT id FROM categories WHERE name = $1", body.name,
    )
    if existing:
        raise HTTPException(409, "Category already exists")
    row = await pool.fetchrow(
        "INSERT INTO categories (name) VALUES ($1) RETURNING id, name",
        body.name,
    )
    return CategoryOut(id=str(row["id"]), name=row["name"], document_count=0)


@app.delete("/api/categories/{cat_id}", status_code=204)
async def api_delete_category(cat_id: str):
    pool = await get_pool()
    doc_count = await pool.fetchval(
        "SELECT COUNT(*) FROM documents WHERE category_id = $1::uuid AND status = 'active'",
        uuid.UUID(cat_id),
    )
    if doc_count > 0:
        raise HTTPException(409, f"Category has {doc_count} active document(s) — remove them first")
    result = await pool.execute(
        "DELETE FROM categories WHERE id = $1::uuid", uuid.UUID(cat_id),
    )
    if result == "DELETE 0":
        raise HTTPException(404, "Category not found")
    return Response(status_code=204)


@app.post("/api/transcribe")
async def api_transcribe(audio: UploadFile):
    from openai import AsyncOpenAI
    from app.config import settings

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    audio_bytes = await audio.read()
    result = await client.audio.transcriptions.create(
        model="whisper-1",
        file=("audio.webm", audio_bytes, audio.content_type or "audio/webm"),
    )
    return {"text": result.text}


# ── Agent endpoints ────────────────────────────────────────────────

async def _load_agent(pool, agent_id: str) -> dict:
    """Load agent + categories. Raises HTTPException(404) if not found."""
    row = await pool.fetchrow(
        "SELECT id, name, description, system_prompt, model, is_active FROM agents WHERE id = $1::uuid",
        uuid.UUID(agent_id),
    )
    if not row:
        raise HTTPException(404, "Agent not found")
    cat_rows = await pool.fetch(
        """SELECT c.id, c.name FROM agent_categories ac
           JOIN categories c ON c.id = ac.category_id
           WHERE ac.agent_id = $1::uuid ORDER BY c.name""",
        uuid.UUID(agent_id),
    )
    return {**dict(row), "categories": [dict(r) for r in cat_rows]}


def _agent_out(data: dict) -> AgentOut:
    aid = str(data["id"])
    return AgentOut(
        id=aid,
        name=data["name"],
        description=data["description"],
        system_prompt=data["system_prompt"],
        model=data["model"],
        is_active=data["is_active"],
        categories=[AgentCategoryOut(id=str(c["id"]), name=c["name"]) for c in data["categories"]],
        endpoint=f"/api/agents/{aid}/chat",
    )


@app.get("/api/agents", response_model=list[AgentOut])
async def api_list_agents():
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT id, name, description, system_prompt, model, is_active FROM agents ORDER BY created_at DESC"
    )
    agents = []
    for r in rows:
        cat_rows = await pool.fetch(
            """SELECT c.id, c.name FROM agent_categories ac
               JOIN categories c ON c.id = ac.category_id
               WHERE ac.agent_id = $1 ORDER BY c.name""",
            r["id"],
        )
        agents.append(_agent_out({**dict(r), "categories": [dict(cr) for cr in cat_rows]}))
    return agents


@app.post("/api/agents", response_model=AgentOut, status_code=201)
async def api_create_agent(body: AgentCreate):
    pool = await get_pool()
    row = await pool.fetchrow(
        """INSERT INTO agents (name, description, system_prompt)
           VALUES ($1, $2, $3)
           RETURNING id, name, description, system_prompt, model, is_active""",
        body.name, body.description, body.system_prompt,
    )
    for cid in body.category_ids:
        await pool.execute(
            "INSERT INTO agent_categories (agent_id, category_id) VALUES ($1, $2::uuid)",
            row["id"], uuid.UUID(cid),
        )
    data = await _load_agent(pool, str(row["id"]))
    return _agent_out(data)


@app.get("/api/agents/{agent_id}", response_model=AgentOut)
async def api_get_agent(agent_id: str):
    pool = await get_pool()
    data = await _load_agent(pool, agent_id)
    return _agent_out(data)


@app.patch("/api/agents/{agent_id}", response_model=AgentOut)
async def api_update_agent(agent_id: str, body: AgentUpdate):
    pool = await get_pool()
    aid = uuid.UUID(agent_id)
    existing = await pool.fetchrow("SELECT id FROM agents WHERE id = $1", aid)
    if not existing:
        raise HTTPException(404, "Agent not found")

    updates = []
    params = []
    for field in ("name", "description", "system_prompt", "is_active"):
        val = getattr(body, field)
        if val is not None:
            updates.append(f"{field} = ${len(params) + 2}")
            params.append(val)
    if updates:
        sql = f"UPDATE agents SET {', '.join(updates)} WHERE id = $1"
        await pool.execute(sql, aid, *params)

    if body.category_ids is not None:
        await pool.execute("DELETE FROM agent_categories WHERE agent_id = $1", aid)
        for cid in body.category_ids:
            await pool.execute(
                "INSERT INTO agent_categories (agent_id, category_id) VALUES ($1, $2::uuid)",
                aid, uuid.UUID(cid),
            )

    data = await _load_agent(pool, agent_id)
    return _agent_out(data)


@app.delete("/api/agents/{agent_id}", status_code=204)
async def api_delete_agent(agent_id: str):
    pool = await get_pool()
    result = await pool.execute("DELETE FROM agents WHERE id = $1::uuid", uuid.UUID(agent_id))
    if result == "DELETE 0":
        raise HTTPException(404, "Agent not found")
    return Response(status_code=204)


@app.post("/api/agents/{agent_id}/chat", response_model=QueryResponse)
async def api_agent_chat(agent_id: str, body: AgentChatRequest):
    from app.retrieval import query as retrieval_query

    pool = await get_pool()
    data = await _load_agent(pool, agent_id)
    if not data["is_active"]:
        raise HTTPException(400, "Agent is inactive")

    category_ids = [str(c["id"]) for c in data["categories"]]

    conversation_history: list[dict] | None = None
    if body.conversation_id:
        cid = uuid.UUID(body.conversation_id)
        await pool.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES ($1, 'user', $2)",
            cid, body.query,
        )
        rows = await pool.fetch(
            "SELECT role, content FROM messages WHERE conversation_id = $1 ORDER BY created_at ASC",
            cid,
        )
        if len(rows) > 1:
            conversation_history = [{"role": r["role"], "content": r["content"]} for r in rows[:-1]]

    result = await retrieval_query(
        body.query,
        conversation_history=conversation_history,
        category_ids=category_ids or None,
        system_prompt=data["system_prompt"] or None,
    )

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
        await pool.execute("UPDATE conversations SET updated_at = now() WHERE id = $1", cid)

    chunks = [
        ChunkOut(
            chunk_id=c.chunk_id, content=c.content, score=round(c.score, 4),
            section_path=c.section_path, filename=c.filename, source=c.source,
        )
        for c in result.chunks_used
    ]
    return QueryResponse(
        answer=result.answer, chunks_used=chunks, query_type=result.query_type.value,
        cached=result.cached, step_timings=result.step_timings, error=result.error,
    )


class DocumentUpdate(BaseModel):
    category_id: str | None = None


@app.patch("/api/documents/{doc_id}")
async def api_update_document(doc_id: str, body: DocumentUpdate):
    pool = await get_pool()
    did = uuid.UUID(doc_id)
    if body.category_id is not None:
        cat = await pool.fetchrow(
            "SELECT id FROM categories WHERE id = $1::uuid", uuid.UUID(body.category_id),
        )
        if not cat:
            raise HTTPException(400, "Category not found")
        result = await pool.execute(
            "UPDATE documents SET category_id = $1::uuid WHERE id = $2 AND status = 'active'",
            uuid.UUID(body.category_id), did,
        )
        if result == "UPDATE 0":
            raise HTTPException(404, "Document not found")
    return {"ok": True}


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
