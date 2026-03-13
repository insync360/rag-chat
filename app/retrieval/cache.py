"""Semantic cache — embed query at 256-dim, cosine search for cache hits."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

from app.config import settings
from app.database import get_pool
from app.retrieval.models import (
    ConflictResolution,
    GraphPath,
    QueryResult,
    QueryType,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)


async def check_cache(
    query: str,
    query_embedding_256: list[float],
) -> QueryResult | None:
    """Check semantic cache for a similar query. Returns None on miss."""
    if not settings.CACHE_ENABLED:
        return None

    try:
        pool = await get_pool()
        emb_str = str(query_embedding_256)
        threshold = settings.CACHE_SIMILARITY_THRESHOLD

        row = await pool.fetchrow(
            """
            SELECT query_text, query_type, answer, result_json,
                   1 - (query_embedding <=> $1::vector) AS similarity
            FROM semantic_cache
            WHERE expires_at > now()
              AND 1 - (query_embedding <=> $1::vector) > $2
            ORDER BY query_embedding <=> $1::vector
            LIMIT 1
            """,
            emb_str, threshold,
        )

        if row is None:
            return None

        data = json.loads(row["result_json"]) if isinstance(row["result_json"], str) else dict(row["result_json"])

        result = QueryResult(
            answer=row["answer"],
            chunks_used=[RetrievedChunk(**c) for c in data.get("chunks_used", [])],
            graph_paths=[GraphPath(**g) for g in data.get("graph_paths", [])],
            conflicts=[ConflictResolution(**c) for c in data.get("conflicts", [])],
            query_type=QueryType(data.get("query_type", "SIMPLE")),
            cached=True,
            step_timings=data.get("step_timings", {}),
        )

        logger.info(
            "Cache hit (%.3f similarity): '%s' → '%s'",
            float(row["similarity"]), query[:50], row["query_text"][:50],
        )
        return result

    except Exception as exc:
        logger.warning("Cache check failed: %s", exc)
        return None


_DECLINE_PHRASES = (
    "could not find",
    "no relevant information",
    "unable to find",
    "insufficient context",
    "failed to generate",
    "an error occurred",
)


def is_cacheable(result: QueryResult) -> bool:
    """Check whether a QueryResult is worth caching.

    Returns False for empty results, graceful declines, and errors.
    """
    if not result.chunks_used:
        return False
    if result.error:
        return False
    if result.skipped:
        return False
    answer_lower = result.answer.lower()
    if any(phrase in answer_lower for phrase in _DECLINE_PHRASES):
        return False
    return True


async def store_cache(
    query: str,
    query_embedding_256: list[float],
    result: QueryResult,
) -> None:
    """Store query result in semantic cache. Never raises."""
    if not settings.CACHE_ENABLED:
        return

    try:
        pool = await get_pool()
        emb_str = str(query_embedding_256)

        result_json = json.dumps({
            "chunks_used": [asdict(c) for c in result.chunks_used],
            "graph_paths": [asdict(g) for g in result.graph_paths],
            "conflicts": [asdict(c) for c in result.conflicts],
            "query_type": result.query_type.value,
            "step_timings": result.step_timings,
        })

        await pool.execute(
            """
            INSERT INTO semantic_cache
                (query_text, query_embedding, query_type, answer, result_json, expires_at)
            VALUES ($1, $2::vector, $3, $4, $5::jsonb,
                    now() + make_interval(hours => $6))
            """,
            query, emb_str, result.query_type.value,
            result.answer, result_json, settings.CACHE_TTL_HOURS,
        )
        logger.info("Cached result for query: '%s'", query[:50])

    except Exception as exc:
        logger.warning("Cache store failed: %s", exc)
