"""Community detection via Leiden algorithm — clusters densely connected entities.

Reads active entities + relationships from Neo4j, runs Leiden (Python-side via
leidenalg + igraph), writes community_id back to Entity nodes, and optionally
generates LLM summaries stored on dedicated Community nodes.

Degrades gracefully if leidenalg/igraph not installed.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone

from openai import AsyncOpenAI

from app.config import settings
from app.graph.models import CommunityDetectionResult, CommunityInfo
from app.graph.neo4j_client import get_driver

logger = logging.getLogger(__name__)

_leiden_available: bool | None = None


def _check_leiden() -> bool:
    """Lazy-check whether leidenalg + igraph are importable."""
    global _leiden_available
    if _leiden_available is not None:
        return _leiden_available
    try:
        import igraph as _ig  # noqa: F401
        import leidenalg as _la  # noqa: F401
        _leiden_available = True
    except ImportError:
        _leiden_available = False
        logger.warning("leidenalg/igraph not installed — community detection disabled")
    return _leiden_available


# ---------------------------------------------------------------------------
# Neo4j reads
# ---------------------------------------------------------------------------

async def _read_graph() -> tuple[list[dict], list[dict]]:
    """Read active entities and their relationships from Neo4j."""
    driver = await get_driver()
    async with driver.session() as session:
        ent_result = await session.run(
            "MATCH (e:Entity) WHERE e.status = 'active' "
            "RETURN id(e) AS neo4j_id, e.name AS name, e.type AS type"
        )
        entities = [dict(r) async for r in ent_result]

        rel_result = await session.run(
            "MATCH (s:Entity)-[r]->(t:Entity) "
            "WHERE s.status = 'active' AND t.status = 'active' "
            "RETURN id(s) AS source_id, id(t) AS target_id, "
            "type(r) AS rel_type, coalesce(r.confidence, 1.0) AS weight"
        )
        relationships = [dict(r) async for r in rel_result]

    return entities, relationships


# ---------------------------------------------------------------------------
# igraph + Leiden (CPU-bound, run in executor)
# ---------------------------------------------------------------------------

def _build_igraph(
    entities: list[dict], relationships: list[dict],
) -> "tuple[igraph.Graph, dict[int, int], list[dict]]":  # noqa: F821
    """Build an undirected igraph.Graph from Neo4j entities/relationships."""
    import igraph

    id_to_idx: dict[int, int] = {}
    for i, ent in enumerate(entities):
        id_to_idx[ent["neo4j_id"]] = i

    g = igraph.Graph(n=len(entities), directed=False)
    g.vs["neo4j_id"] = [e["neo4j_id"] for e in entities]
    g.vs["name"] = [e["name"] for e in entities]
    g.vs["type"] = [e["type"] for e in entities]

    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    for rel in relationships:
        src_idx = id_to_idx.get(rel["source_id"])
        tgt_idx = id_to_idx.get(rel["target_id"])
        if src_idx is None or tgt_idx is None:
            continue
        if src_idx == tgt_idx:
            continue  # filter self-loops
        edges.append((src_idx, tgt_idx))
        weights.append(float(rel["weight"]))

    if edges:
        g.add_edges(edges)
        g.es["weight"] = weights

    return g, id_to_idx, entities


def _run_leiden(g: "igraph.Graph") -> list[int]:  # noqa: F821
    """Run Leiden algorithm, return membership list (community_id per vertex)."""
    import leidenalg

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight" if g.ecount() > 0 else None,
        n_iterations=-1,
        seed=42,
        resolution_parameter=settings.COMMUNITY_RESOLUTION,
    )
    return list(partition.membership)


# ---------------------------------------------------------------------------
# Write results back to Neo4j
# ---------------------------------------------------------------------------

async def _write_community_ids(assignments: list[dict]) -> None:
    """Batch-write community_id to Entity nodes."""
    driver = await get_driver()
    async with driver.session() as session:
        await session.run(
            "UNWIND $assignments AS a "
            "MATCH (e:Entity) WHERE id(e) = a.neo4j_id "
            "SET e.community_id = a.community_id",
            assignments=assignments,
        )


async def _store_community_nodes(communities: list[CommunityInfo]) -> None:
    """MERGE Community summary nodes — one per community."""
    driver = await get_driver()
    now = datetime.now(timezone.utc).isoformat()
    async with driver.session() as session:
        for comm in communities:
            await session.run(
                "MERGE (c:Community {community_id: $cid}) "
                "SET c.summary = $summary, c.entity_count = $size, "
                "c.updated_at = $now",
                cid=comm.community_id,
                summary=comm.summary,
                size=comm.size,
                now=now,
            )


# ---------------------------------------------------------------------------
# Community info building
# ---------------------------------------------------------------------------

def _build_community_infos(
    entities: list[dict], membership: list[int], relationships: list[dict],
) -> list[CommunityInfo]:
    """Build CommunityInfo structs from membership + entity/relationship data."""
    comm_entities: dict[int, list[dict]] = defaultdict(list)
    for ent, cid in zip(entities, membership):
        comm_entities[cid].append(ent)

    # Collect relationship types per community
    entity_to_community: dict[int, int] = {}
    for ent, cid in zip(entities, membership):
        entity_to_community[ent["neo4j_id"]] = cid

    comm_rel_types: dict[int, set[str]] = defaultdict(set)
    for rel in relationships:
        src_cid = entity_to_community.get(rel["source_id"])
        tgt_cid = entity_to_community.get(rel["target_id"])
        if src_cid is not None:
            comm_rel_types[src_cid].add(rel["rel_type"])
        if tgt_cid is not None and tgt_cid != src_cid:
            comm_rel_types[tgt_cid].add(rel["rel_type"])

    communities: list[CommunityInfo] = []
    for cid, ents in sorted(comm_entities.items()):
        communities.append(CommunityInfo(
            community_id=cid,
            entity_names=[e["name"] for e in ents],
            entity_types=list({e["type"] for e in ents}),
            relationship_types=sorted(comm_rel_types.get(cid, set())),
            size=len(ents),
        ))

    return communities


# ---------------------------------------------------------------------------
# LLM summaries
# ---------------------------------------------------------------------------

async def _generate_community_summaries(
    communities: list[CommunityInfo],
) -> list[CommunityInfo]:
    """Generate a 1-2 sentence topic summary per community via GPT-4o-mini."""
    if not settings.COMMUNITY_SUMMARY_ENABLED:
        return communities

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    for comm in communities:
        if comm.size < settings.COMMUNITY_MIN_SIZE:
            continue
        try:
            prompt = (
                f"Entities: {', '.join(comm.entity_names)}\n"
                f"Entity types: {', '.join(comm.entity_types)}\n"
                f"Relationship types: {', '.join(comm.relationship_types)}\n\n"
                "Write a 1-2 sentence summary describing the topic or theme "
                "of this entity cluster."
            )
            resp = await client.chat.completions.create(
                model=settings.COMMUNITY_SUMMARY_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3,
            )
            comm.summary = resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning(
                "Summary generation failed for community %d: %s",
                comm.community_id, exc,
            )

    return communities


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def detect_communities() -> CommunityDetectionResult:
    """Detect communities in the knowledge graph using Leiden algorithm.

    Never raises — returns skipped=True on any failure so callers can proceed.
    """
    if not settings.COMMUNITY_DETECTION_ENABLED:
        return CommunityDetectionResult(
            communities=[], total_entities=0, total_communities=0,
            skipped=True, error=None,
        )

    if not _check_leiden():
        return CommunityDetectionResult(
            communities=[], total_entities=0, total_communities=0,
            skipped=True, error="leidenalg/igraph not installed",
        )

    try:
        entities, relationships = await _read_graph()

        if not entities:
            return CommunityDetectionResult(
                communities=[], total_entities=0, total_communities=0,
                skipped=False, error=None,
            )

        # CPU-bound: run in executor
        loop = asyncio.get_event_loop()
        g, id_to_idx, _ = await loop.run_in_executor(
            None, _build_igraph, entities, relationships,
        )
        membership = await loop.run_in_executor(None, _run_leiden, g)

        # Write community_ids back to Neo4j
        assignments = [
            {"neo4j_id": ent["neo4j_id"], "community_id": cid}
            for ent, cid in zip(entities, membership)
        ]
        await _write_community_ids(assignments)

        # Build community info
        communities = _build_community_infos(entities, membership, relationships)

        # Generate + store summaries
        communities = await _generate_community_summaries(communities)
        await _store_community_nodes(communities)

        logger.info(
            "Community detection complete: %d entities → %d communities",
            len(entities), len(communities),
        )
        return CommunityDetectionResult(
            communities=communities,
            total_entities=len(entities),
            total_communities=len(communities),
            skipped=False,
            error=None,
        )

    except Exception as exc:
        logger.warning("Community detection failed: %s", exc)
        return CommunityDetectionResult(
            communities=[], total_entities=0, total_communities=0,
            skipped=True, error=str(exc),
        )
