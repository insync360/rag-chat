"""Hybrid vector + BM25 retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

import json
import logging
import math
import uuid as _uuid

from openai import AsyncOpenAI

from app.config import settings
from app.database import get_pool
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


def _is_uuid(value: str) -> bool:
    try:
        _uuid.UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


def _normalize_filters(raw: dict) -> dict:
    """Validate and normalize metadata_filters from the classifier.

    The LLM sometimes puts chapter/section names in 'document_id'.
    If the value isn't a valid UUID, reclassify it as 'section_path'.
    """
    out: dict = {}
    if "document_id" in raw:
        val = raw["document_id"]
        if _is_uuid(str(val)):
            out["document_id"] = str(val)
        else:
            # Not a UUID — treat as section path filter
            out["section_path"] = str(val)
    if "filename" in raw:
        out["filename"] = str(raw["filename"])
    if "section_path" in raw:
        out["section_path"] = str(raw["section_path"])
    return out


# ---------------------------------------------------------------------------
# Query embedding (reuses hybrid_embeddings utilities)
# ---------------------------------------------------------------------------

def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-12:
        return vec
    return [x / norm for x in vec]


def _truncate_and_normalize(vec: list[float], dim: int) -> list[float]:
    return _l2_normalize(vec[:dim])


async def embed_query(
    client: AsyncOpenAI, query: str,
) -> tuple[list[float], list[float]]:
    """Embed query at full 2000-dim. Returns (full_2000, hybrid_768).

    hybrid_768 = 512 MRL text + 256 zero-pad (graph dims), L2-normalized.
    """
    resp = await client.embeddings.create(
        model=settings.CHUNK_EMBEDDING_MODEL,
        input=query,
        dimensions=settings.CHUNK_EMBEDDING_DIMENSIONS,
    )
    full = resp.data[0].embedding

    # Build hybrid query vector: 512 text + 128 zeros (GraphSAGE) + 128 zeros (TransE)
    text_part = _truncate_and_normalize(full, settings.HYBRID_CHUNK_TEXT_DIM)
    graph_pad = [0.0] * (settings.GRAPHSAGE_OUTPUT_DIM + settings.TRANSE_DIM)
    hybrid = _l2_normalize(text_part + graph_pad)

    return full, hybrid


async def embed_query_small(client: AsyncOpenAI, query: str) -> list[float]:
    """Embed query at 256-dim (for entity search + semantic cache)."""
    resp = await client.embeddings.create(
        model=settings.ENTITY_EMBEDDING_MODEL,
        input=query,
        dimensions=settings.CACHE_EMBEDDING_DIM,
    )
    return resp.data[0].embedding


# ---------------------------------------------------------------------------
# Vector search (pgvector HNSW on hybrid_embedding)
# ---------------------------------------------------------------------------

async def _vector_search(
    query_embedding_768: list[float],
    top_k: int,
    metadata_filters: dict | None = None,
    category_ids: list[str] | None = None,
) -> list[RetrievedChunk]:
    pool = await get_pool()
    emb_str = str(query_embedding_768)

    filter_clause = ""
    params: list = [emb_str, top_k]

    if metadata_filters:
        filters = _normalize_filters(metadata_filters)
        if "document_id" in filters:
            filter_clause += f" AND c.document_id = ${len(params) + 1}::uuid"
            params.append(filters["document_id"])
        if "filename" in filters:
            filter_clause += f" AND d.filename ILIKE ${len(params) + 1}"
            params.append(f"%{filters['filename']}%")
        if "section_path" in filters:
            filter_clause += f" AND c.section_path ILIKE ${len(params) + 1}"
            params.append(f"%{filters['section_path']}%")

    if category_ids:
        filter_clause += f" AND d.category_id = ANY(${len(params) + 1}::uuid[])"
        params.append(category_ids)

    # Exclude structural chunk types (HEADING, INDEX)
    exclude_types = settings.RETRIEVAL_EXCLUDE_CHUNK_TYPES
    if exclude_types:
        placeholders = ", ".join(f"${len(params) + 1 + i}" for i in range(len(exclude_types)))
        filter_clause += f" AND COALESCE(c.metadata->>'chunk_type', 'PARAGRAPH') NOT IN ({placeholders})"
        params.extend(exclude_types)

    sql = f"""
        SELECT c.id::text AS chunk_id, c.document_id::text, c.content,
               c.section_path, c.metadata,
               d.filename, d.version, d.ingested_at,
               1 - (ce.hybrid_embedding <=> $1::vector) AS similarity
        FROM chunk_embeddings ce
        JOIN chunks c ON c.id = ce.chunk_id
        JOIN documents d ON d.id = c.document_id
        WHERE d.status = 'active' AND ce.hybrid_embedding IS NOT NULL
        {filter_clause}
        ORDER BY ce.hybrid_embedding <=> $1::vector
        LIMIT $2
    """
    rows = await pool.fetch(sql, *params)

    return [
        RetrievedChunk(
            chunk_id=r["chunk_id"],
            document_id=r["document_id"],
            content=r["content"],
            score=float(r["similarity"]),
            section_path=r["section_path"],
            metadata=json.loads(r["metadata"]) if isinstance(r["metadata"], str) else dict(r["metadata"] or {}),
            source="vector",
            filename=r["filename"],
            version=r["version"],
            ingested_at=str(r["ingested_at"]),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# BM25 search (tsvector GIN index)
# ---------------------------------------------------------------------------

async def _bm25_search(
    query_text: str,
    top_k: int,
    metadata_filters: dict | None = None,
    category_ids: list[str] | None = None,
) -> list[RetrievedChunk]:
    pool = await get_pool()

    filter_clause = ""
    params: list = [query_text, top_k]

    if metadata_filters:
        filters = _normalize_filters(metadata_filters)
        if "document_id" in filters:
            filter_clause += f" AND c.document_id = ${len(params) + 1}::uuid"
            params.append(filters["document_id"])
        if "filename" in filters:
            filter_clause += f" AND d.filename ILIKE ${len(params) + 1}"
            params.append(f"%{filters['filename']}%")
        if "section_path" in filters:
            filter_clause += f" AND c.section_path ILIKE ${len(params) + 1}"
            params.append(f"%{filters['section_path']}%")

    if category_ids:
        filter_clause += f" AND d.category_id = ANY(${len(params) + 1}::uuid[])"
        params.append(category_ids)

    # Exclude structural chunk types (HEADING, INDEX)
    exclude_types = settings.RETRIEVAL_EXCLUDE_CHUNK_TYPES
    if exclude_types:
        placeholders = ", ".join(f"${len(params) + 1 + i}" for i in range(len(exclude_types)))
        filter_clause += f" AND COALESCE(c.metadata->>'chunk_type', 'PARAGRAPH') NOT IN ({placeholders})"
        params.extend(exclude_types)

    sql = f"""
        SELECT c.id::text AS chunk_id, c.document_id::text, c.content,
               c.section_path, c.metadata,
               d.filename, d.version, d.ingested_at,
               ts_rank_cd(c.search_tsvector,
                          plainto_tsquery('english', $1), 32) AS bm25_score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE d.status = 'active'
          AND c.search_tsvector @@ plainto_tsquery('english', $1)
        {filter_clause}
        ORDER BY bm25_score DESC
        LIMIT $2
    """
    rows = await pool.fetch(sql, *params)

    return [
        RetrievedChunk(
            chunk_id=r["chunk_id"],
            document_id=r["document_id"],
            content=r["content"],
            score=float(r["bm25_score"]),
            section_path=r["section_path"],
            metadata=json.loads(r["metadata"]) if isinstance(r["metadata"], str) else dict(r["metadata"] or {}),
            source="bm25",
            filename=r["filename"],
            version=r["version"],
            ingested_at=str(r["ingested_at"]),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists via RRF. Returns (chunk_id, rrf_score) sorted desc."""
    rrf_scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank_pos, (chunk_id, _) in enumerate(ranked_list):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank_pos + 1)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Public hybrid search
# ---------------------------------------------------------------------------

async def hybrid_search(
    query: str,
    query_embedding_768: list[float],
    top_k: int | None = None,
    metadata_filters: dict | None = None,
    category_ids: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Run vector + BM25 search, fuse with RRF, return top_k chunks."""
    top_k = top_k or settings.RETRIEVAL_TOP_K_FINAL
    k_vec = settings.RETRIEVAL_TOP_K_VECTOR
    k_bm25 = settings.RETRIEVAL_TOP_K_BM25

    # Run vector and BM25 in parallel
    import asyncio
    vec_chunks, bm25_chunks = await asyncio.gather(
        _vector_search(query_embedding_768, k_vec, metadata_filters, category_ids),
        _bm25_search(query, k_bm25, metadata_filters, category_ids),
    )

    # Build ranked lists for RRF
    vec_ranked = [(c.chunk_id, c.score) for c in vec_chunks]
    bm25_ranked = [(c.chunk_id, c.score) for c in bm25_chunks]

    fused = reciprocal_rank_fusion(
        [vec_ranked, bm25_ranked], k=settings.RETRIEVAL_RRF_K,
    )

    # Build chunk lookup for metadata
    chunk_map: dict[str, RetrievedChunk] = {}
    for c in vec_chunks:
        chunk_map[c.chunk_id] = c
    for c in bm25_chunks:
        if c.chunk_id not in chunk_map:
            chunk_map[c.chunk_id] = c

    # Return fused results
    results: list[RetrievedChunk] = []
    for chunk_id, rrf_score in fused[:top_k]:
        if chunk_id in chunk_map:
            chunk = chunk_map[chunk_id]
            chunk.score = rrf_score
            chunk.source = "fused"
            results.append(chunk)

    logger.info(
        "Hybrid search: %d vector + %d BM25 → %d fused",
        len(vec_chunks), len(bm25_chunks), len(results),
    )
    return results
