"""Six LangGraph agent node functions. Each: async (state) -> dict."""

from __future__ import annotations

import asyncio
import logging
import time

from openai import AsyncOpenAI

from app.config import settings
from app.retrieval.models import QueryType, RetrievedChunk

logger = logging.getLogger(__name__)


async def _retry(fn, *args, max_retries: int | None = None, **kwargs):
    """Retry with exponential backoff. Returns fn result or raises last error."""
    max_retries = max_retries if max_retries is not None else settings.AGENT_MAX_RETRIES
    base_ms = settings.AGENT_RETRY_BASE_MS
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                await asyncio.sleep(base_ms * (2 ** attempt) / 1000)
    raise last_exc


# ---------------------------------------------------------------------------
# 1. Planner Agent
# ---------------------------------------------------------------------------

async def planner_agent(state: dict) -> dict:
    """Classify query and create execution plan."""
    t0 = time.monotonic()
    try:
        from app.retrieval.classifier import classify_query
        plan = await _retry(classify_query, state["original_query"])
        elapsed = time.monotonic() - t0
        return {
            "plan": plan,
            "step_timings": {"planner": round(elapsed, 3)},
            "errors": [],
        }
    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.warning("Planner agent failed: %s", exc)
        # Default plan — SIMPLE with vector-only
        from app.retrieval.models import ExecutionPlan, QueryType
        return {
            "plan": ExecutionPlan(query_type=QueryType.SIMPLE),
            "step_timings": {"planner": round(elapsed, 3)},
            "errors": [f"planner: {exc}"],
        }


# ---------------------------------------------------------------------------
# 2. Vector Agent
# ---------------------------------------------------------------------------

async def vector_agent(state: dict) -> dict:
    """Hybrid search + HyDE + rerank + freshness scoring."""
    t0 = time.monotonic()
    try:
        from app.retrieval.hyde import generate_hyde_passage
        from app.retrieval.reranker import rerank
        from app.retrieval.vector_search import embed_query, embed_query_small, hybrid_search

        query = state["original_query"]
        plan = state.get("plan")
        filters = plan.metadata_filters if plan else {}
        query_type = plan.query_type if plan else QueryType.SIMPLE
        pass_count = state.get("pass_count", 0)

        # On 2nd pass: clear filters to broaden search
        if pass_count > 0:
            filters = {}
            logger.info("Vector agent 2nd pass: cleared filters")

        # Use expanded queries if available
        queries = [query]
        if plan and plan.expanded_queries:
            queries = plan.expanded_queries

        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        # HyDE: generate hypothetical passage for vector search
        hyde_passage = await generate_hyde_passage(client, query, query_type)

        # Embed: HyDE passage for vector search, raw query for BM25
        if hyde_passage:
            _, hyde_hybrid_emb = await _retry(embed_query, client, hyde_passage)
            _, raw_hybrid_emb = await _retry(embed_query, client, query)
        else:
            _, raw_hybrid_emb = await _retry(embed_query, client, query)
            hyde_hybrid_emb = raw_hybrid_emb

        small_emb = await _retry(embed_query_small, client, query)

        # Search with all queries, merge
        all_chunks: list[RetrievedChunk] = []
        for q in queries:
            if q == query:
                # Primary query: HyDE embedding for vector, raw text for BM25
                chunks = await _retry(hybrid_search, q, hyde_hybrid_emb, None, filters)
            else:
                # Expanded queries: embed individually (no HyDE)
                _, q_hybrid = await _retry(embed_query, client, q)
                chunks = await _retry(hybrid_search, q, q_hybrid, None, filters)
            all_chunks.extend(chunks)

        # Deduplicate by chunk_id (keep highest score)
        seen: dict[str, RetrievedChunk] = {}
        for c in all_chunks:
            if c.chunk_id not in seen or c.score > seen[c.chunk_id].score:
                seen[c.chunk_id] = c
        deduped = sorted(seen.values(), key=lambda c: c.score, reverse=True)

        # Rerank with raw query (not HyDE) — judge relevance to user intent
        reranked = await _retry(rerank, query, deduped)

        elapsed = time.monotonic() - t0
        return {
            "retrieved_chunks": reranked,
            "query_embedding_256": small_emb,
            "query_embedding_768": raw_hybrid_emb,
            "step_timings": {"vector": round(elapsed, 3)},
            "errors": [],
        }

    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.warning("Vector agent failed: %s", exc)
        return {
            "retrieved_chunks": [],
            "step_timings": {"vector": round(elapsed, 3)},
            "errors": [f"vector: {exc}"],
        }


# ---------------------------------------------------------------------------
# 3. Graph Agent
# ---------------------------------------------------------------------------

async def graph_agent(state: dict) -> dict:
    """Entity match → Neo4j traverse → chunk retrieval."""
    t0 = time.monotonic()
    try:
        from app.retrieval.graph_search import graph_search

        query = state["original_query"]
        emb_256 = state.get("query_embedding_256")

        if emb_256 is None:
            from app.retrieval.vector_search import embed_query_small
            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            emb_256 = await embed_query_small(client, query)

        chunks, paths = await _retry(graph_search, query, emb_256)

        elapsed = time.monotonic() - t0
        return {
            "retrieved_chunks": chunks,
            "graph_paths": paths,
            "step_timings": {"graph": round(elapsed, 3)},
            "errors": [],
        }

    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.warning("Graph agent failed: %s", exc)
        return {
            "retrieved_chunks": [],
            "graph_paths": [],
            "step_timings": {"graph": round(elapsed, 3)},
            "errors": [f"graph: {exc}"],
        }


# ---------------------------------------------------------------------------
# 4. Conflict Agent
# ---------------------------------------------------------------------------

async def conflict_agent(state: dict) -> dict:
    """Detect and resolve contradictions in retrieved chunks."""
    t0 = time.monotonic()
    try:
        from app.retrieval.conflict import detect_and_resolve_conflicts
        chunks = state.get("retrieved_chunks", [])
        conflicts = await _retry(detect_and_resolve_conflicts, chunks)

        elapsed = time.monotonic() - t0
        return {
            "conflicts": conflicts,
            "step_timings": {"conflict": round(elapsed, 3)},
            "errors": [],
        }

    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.warning("Conflict agent failed: %s", exc)
        return {
            "conflicts": [],
            "step_timings": {"conflict": round(elapsed, 3)},
            "errors": [f"conflict: {exc}"],
        }


# ---------------------------------------------------------------------------
# 5. Calculator Agent
# ---------------------------------------------------------------------------

async def calculator_agent(state: dict) -> dict:
    """Safe arithmetic via structured extraction."""
    t0 = time.monotonic()
    try:
        from app.retrieval.calculator import calculate
        query = state["original_query"]
        chunks = state.get("retrieved_chunks", [])
        result = await _retry(calculate, query, chunks)

        elapsed = time.monotonic() - t0
        return {
            "calculation_result": result,
            "step_timings": {"calculator": round(elapsed, 3)},
            "errors": [],
        }

    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.warning("Calculator agent failed: %s", exc)
        return {
            "calculation_result": None,
            "step_timings": {"calculator": round(elapsed, 3)},
            "errors": [f"calculator: {exc}"],
        }


# ---------------------------------------------------------------------------
# 6. Summariser Agent
# ---------------------------------------------------------------------------

async def summariser_agent(state: dict) -> dict:
    """Compress retrieved chunks into final answer."""
    t0 = time.monotonic()
    try:
        from app.retrieval.summariser import summarise_chunks
        query = state["original_query"]
        chunks = state.get("retrieved_chunks", [])
        graph_paths = state.get("graph_paths", [])

        # Deduplicate chunks across passes (operator.add reducer accumulates)
        seen_ids: set[str] = set()
        unique_chunks: list[RetrievedChunk] = []
        for c in chunks:
            if c.chunk_id not in seen_ids:
                seen_ids.add(c.chunk_id)
                unique_chunks.append(c)
        chunks = unique_chunks
        calc_result = state.get("calculation_result")

        answer = await _retry(summarise_chunks, query, chunks, graph_paths, calc_result)

        elapsed = time.monotonic() - t0
        return {
            "final_answer": answer,
            "step_timings": {"summariser": round(elapsed, 3)},
            "errors": [],
            "pass_count": state.get("pass_count", 0) + 1,
        }

    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.warning("Summariser agent failed: %s", exc)
        return {
            "final_answer": f"Failed to generate answer: {exc}",
            "step_timings": {"summariser": round(elapsed, 3)},
            "errors": [f"summariser: {exc}"],
            "pass_count": state.get("pass_count", 0) + 1,
        }
