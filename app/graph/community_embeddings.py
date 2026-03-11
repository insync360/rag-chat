"""Community summary embeddings — embed Leiden community summaries via OpenAI.

Reads community summaries (from pipeline or Neo4j), builds enriched text,
embeds via text-embedding-3-large (512 dims), and stores in Neon pgvector.

Enables high-level thematic retrieval: match queries to community themes
first, then drill into entity/chunk results within that community.
"""

import logging

from openai import AsyncOpenAI

from app.config import settings
from app.database import get_pool
from app.graph.models import CommunitySummaryEmbeddingResult, CommunityInfo
from app.graph.neo4j_client import get_driver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text building
# ---------------------------------------------------------------------------

def _build_embedding_text(comm: CommunityInfo) -> str:
    """Build enriched text for embedding — mirrors embedder.py pattern."""
    if not comm.summary:
        return ""

    parts: list[str] = []

    if comm.entity_names:
        parts.append(f"Entities: {', '.join(comm.entity_names)}")

    if comm.entity_types:
        parts.append(f"Entity types: {', '.join(comm.entity_types)}")

    if comm.relationship_types:
        parts.append(f"Relationships: {', '.join(comm.relationship_types)}")

    parts.append(f"Summary: {comm.summary}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Neo4j read (standalone path — when communities not passed from pipeline)
# ---------------------------------------------------------------------------

async def _read_communities_from_neo4j() -> list[CommunityInfo]:
    """Read Community nodes + grouped Entity data from Neo4j."""
    driver = await get_driver()
    async with driver.session() as session:
        # Get community summaries
        comm_result = await session.run(
            "MATCH (c:Community) "
            "RETURN c.community_id AS community_id, c.summary AS summary, "
            "c.entity_count AS size"
        )
        comm_rows = [dict(r) async for r in comm_result]

        if not comm_rows:
            return []

        # Get entities grouped by community_id
        ent_result = await session.run(
            "MATCH (e:Entity) WHERE e.status = 'active' AND e.community_id IS NOT NULL "
            "RETURN e.community_id AS community_id, "
            "collect(e.name) AS names, collect(DISTINCT e.type) AS types"
        )
        ent_by_comm: dict[int, dict] = {}
        async for r in ent_result:
            ent_by_comm[r["community_id"]] = {
                "names": r["names"], "types": r["types"],
            }

        # Get relationship types per community
        rel_result = await session.run(
            "MATCH (s:Entity)-[r]->(t:Entity) "
            "WHERE s.status = 'active' AND s.community_id IS NOT NULL "
            "RETURN s.community_id AS community_id, "
            "collect(DISTINCT type(r)) AS rel_types"
        )
        rel_by_comm: dict[int, list[str]] = {}
        async for r in rel_result:
            rel_by_comm[r["community_id"]] = r["rel_types"]

    communities: list[CommunityInfo] = []
    for row in comm_rows:
        cid = row["community_id"]
        ent_data = ent_by_comm.get(cid, {"names": [], "types": []})
        communities.append(CommunityInfo(
            community_id=cid,
            entity_names=ent_data["names"],
            entity_types=ent_data["types"],
            relationship_types=sorted(rel_by_comm.get(cid, [])),
            size=row["size"] or 0,
            summary=row["summary"],
        ))

    return communities


# ---------------------------------------------------------------------------
# OpenAI embedding
# ---------------------------------------------------------------------------

async def _embed_summaries(texts: list[str]) -> list[list[float]]:
    """Batched OpenAI embedding call — follows embedder.py pattern."""
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    all_embeddings: list[list[float]] = []
    batch_size = settings.COMMUNITY_SUMMARY_EMBEDDING_BATCH_SIZE

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = await client.embeddings.create(
            model=settings.COMMUNITY_SUMMARY_EMBEDDING_MODEL,
            input=batch,
            dimensions=settings.COMMUNITY_SUMMARY_EMBEDDING_DIMENSIONS,
        )
        all_embeddings.extend([d.embedding for d in resp.data])

    return all_embeddings


# ---------------------------------------------------------------------------
# Neon storage
# ---------------------------------------------------------------------------

async def _store_community_embeddings(
    communities: list[CommunityInfo],
    texts: list[str],
    embeddings: list[list[float]],
) -> None:
    """Batch upsert community summary embeddings to Neon."""
    pool = await get_pool()
    rows = [
        (
            communities[i].community_id,
            texts[i],
            str(embeddings[i]),
            settings.COMMUNITY_SUMMARY_EMBEDDING_MODEL,
        )
        for i in range(len(communities))
    ]
    await pool.executemany(
        "INSERT INTO community_summary_embeddings "
        "(community_id, summary_text, embedding, model) "
        "VALUES ($1, $2, $3::vector, $4) "
        "ON CONFLICT (community_id) DO UPDATE SET "
        "summary_text = EXCLUDED.summary_text, "
        "embedding = EXCLUDED.embedding, "
        "model = EXCLUDED.model, "
        "updated_at = now()",
        rows,
    )


async def _cleanup_stale_communities(active_ids: list[int]) -> None:
    """Delete embeddings for community IDs no longer in the current Leiden run."""
    if not active_ids:
        return
    pool = await get_pool()
    # Use ANY($1::int[]) for parameterized IN clause
    await pool.execute(
        "DELETE FROM community_summary_embeddings "
        "WHERE community_id != ALL($1::int[])",
        active_ids,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def generate_community_summary_embeddings(
    communities: list[CommunityInfo] | None = None,
) -> CommunitySummaryEmbeddingResult:
    """Embed community summaries and store in Neon pgvector. Never raises."""
    if not settings.COMMUNITY_SUMMARY_EMBEDDING_ENABLED:
        return CommunitySummaryEmbeddingResult(
            community_count=0,
            embedding_dim=settings.COMMUNITY_SUMMARY_EMBEDDING_DIMENSIONS,
            skipped=True,
        )

    try:
        # Get communities (from pipeline or Neo4j)
        if communities is None:
            communities = await _read_communities_from_neo4j()

        # Filter to those with summaries
        with_summaries = [c for c in communities if c.summary]

        if not with_summaries:
            return CommunitySummaryEmbeddingResult(
                community_count=0,
                embedding_dim=settings.COMMUNITY_SUMMARY_EMBEDDING_DIMENSIONS,
                skipped=False,
            )

        # Build enriched texts
        texts = [_build_embedding_text(c) for c in with_summaries]

        # Embed
        embeddings = await _embed_summaries(texts)

        # Store
        await _store_community_embeddings(with_summaries, texts, embeddings)

        # Cleanup stale community IDs from prior Leiden runs
        active_ids = [c.community_id for c in communities]
        await _cleanup_stale_communities(active_ids)

        logger.info(
            "Community summary embeddings: %d communities embedded (%d-dim)",
            len(with_summaries),
            settings.COMMUNITY_SUMMARY_EMBEDDING_DIMENSIONS,
        )
        return CommunitySummaryEmbeddingResult(
            community_count=len(with_summaries),
            embedding_dim=settings.COMMUNITY_SUMMARY_EMBEDDING_DIMENSIONS,
            skipped=False,
        )

    except Exception as exc:
        logger.warning("Community summary embeddings failed: %s", exc)
        return CommunitySummaryEmbeddingResult(
            community_count=0,
            embedding_dim=settings.COMMUNITY_SUMMARY_EMBEDDING_DIMENSIONS,
            skipped=True,
            error=str(exc),
        )
