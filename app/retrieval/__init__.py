"""Agentic retrieval engine — single entry point for querying.

NOTE: Imports are deferred to avoid circular imports and heavy startup cost.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from app.config import settings
from app.retrieval.models import QueryResult, QueryType

logger = logging.getLogger(__name__)


async def query(user_query: str) -> QueryResult:
    """Main entry point. Checks cache → runs LangGraph → caches result. Never raises."""
    try:
        from openai import AsyncOpenAI

        from app.retrieval.vector_search import embed_query_small

        t0 = time.monotonic()

        # 1. Embed query at 256-dim (for cache + entity search)
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        query_emb_256 = await embed_query_small(client, user_query)

        # 2. Check semantic cache
        from app.retrieval.cache import check_cache, is_cacheable, store_cache

        cached = await check_cache(user_query, query_emb_256)
        if cached is not None:
            asyncio.create_task(_log_query(user_query, cached))
            return cached

        # 3. Build and invoke LangGraph
        from app.retrieval.graph_builder import build_retrieval_graph

        graph = build_retrieval_graph()

        initial_state = {
            "original_query": user_query,
            "plan": None,
            "retrieved_chunks": [],
            "graph_paths": [],
            "conflicts": [],
            "calculation_result": None,
            "final_answer": "",
            "errors": [],
            "step_timings": {},
            "pass_count": 0,
            "query_embedding_256": query_emb_256,
            "query_embedding_768": None,
        }

        final_state = await graph.ainvoke(initial_state)

        # 4. Build QueryResult
        total_elapsed = round(time.monotonic() - t0, 3)
        step_timings = dict(final_state.get("step_timings", {}))
        step_timings["total"] = total_elapsed

        plan = final_state.get("plan")
        result = QueryResult(
            answer=final_state.get("final_answer", ""),
            chunks_used=final_state.get("retrieved_chunks", []),
            graph_paths=final_state.get("graph_paths", []),
            conflicts=final_state.get("conflicts", []),
            query_type=plan.query_type if plan else QueryType.SIMPLE,
            cached=False,
            step_timings=step_timings,
            error="; ".join(final_state.get("errors", [])) or None,
        )

        # 5. Cache result (fire-and-forget) — only if result is valid
        if is_cacheable(result):
            asyncio.create_task(store_cache(user_query, query_emb_256, result))
        else:
            logger.info("Skipping cache: result not cacheable (chunks=%d, error=%s)", len(result.chunks_used), result.error)

        # 6. Log query (fire-and-forget)
        asyncio.create_task(_log_query(user_query, result))

        logger.info(
            "Query completed in %.1fs: type=%s, chunks=%d, paths=%d, conflicts=%d",
            total_elapsed, result.query_type.value,
            len(result.chunks_used), len(result.graph_paths), len(result.conflicts),
        )
        return result

    except Exception as exc:
        logger.error("Query failed: %s", exc)
        return QueryResult(
            answer=f"An error occurred while processing your query: {exc}",
            query_type=QueryType.SIMPLE,
            skipped=True,
            error=str(exc),
        )


async def _log_query(user_query: str, result: QueryResult) -> None:
    """Log query to query_logs table. Never raises."""
    try:
        from app.database import get_pool

        pool = await get_pool()
        await pool.execute(
            """
            INSERT INTO query_logs
                (query_text, query_type, cached, chunks_retrieved,
                 graph_paths, conflicts, step_timings, error)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)
            """,
            user_query,
            result.query_type.value,
            result.cached,
            len(result.chunks_used),
            len(result.graph_paths),
            len(result.conflicts),
            json.dumps(result.step_timings),
            result.error,
        )
    except Exception as exc:
        logger.warning("Query logging failed: %s", exc)
