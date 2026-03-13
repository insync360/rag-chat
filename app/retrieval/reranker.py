"""Cohere Rerank 3 cross-encoder + freshness scoring. Never raises."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.config import settings
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Freshness scoring
# ---------------------------------------------------------------------------

def _freshness_boost(ingested_at: str) -> float:
    """Multiplicative freshness boost. Max +5% for brand-new docs."""
    if not settings.FRESHNESS_BOOST_ENABLED or not ingested_at:
        return 1.0
    try:
        ts = datetime.fromisoformat(ingested_at.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        days_since = max(0.0, (now - ts).total_seconds() / 86400)
        freshness = max(0.0, 1.0 - days_since / settings.FRESHNESS_DECAY_DAYS)
        return 1.0 + settings.FRESHNESS_WEIGHT * freshness
    except (ValueError, TypeError):
        return 1.0


# ---------------------------------------------------------------------------
# Cohere reranking
# ---------------------------------------------------------------------------

async def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_n: int | None = None,
) -> list[RetrievedChunk]:
    """Rerank chunks with Cohere, apply freshness. Falls back to original order."""
    top_n = top_n or settings.RERANK_TOP_N
    if not chunks:
        return []

    reranked = chunks

    if settings.RERANK_ENABLED and settings.COHERE_API_KEY:
        try:
            import cohere

            client = cohere.AsyncClientV2(api_key=settings.COHERE_API_KEY)
            documents = [c.content[:4096] for c in chunks[:20]]

            resp = await client.rerank(
                model=settings.RERANK_MODEL,
                query=query,
                documents=documents,
                top_n=top_n,
            )

            reranked = []
            for r in resp.results:
                chunk = chunks[r.index]
                chunk.score = r.relevance_score
                reranked.append(chunk)

            logger.info("Cohere reranked %d → %d chunks", len(chunks), len(reranked))

        except Exception as exc:
            logger.warning("Cohere rerank failed, using original order: %s", exc)
            reranked = chunks[:top_n]
    else:
        reranked = chunks[:top_n]

    # Apply freshness boost
    for chunk in reranked:
        chunk.score *= _freshness_boost(chunk.ingested_at)

    return reranked
