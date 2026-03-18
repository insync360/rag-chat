"""Graph-based retrieval — entity matching from query → Neo4j traversal → chunks."""

from __future__ import annotations

import asyncio
import json
import logging

from openai import AsyncOpenAI

from app.config import settings
from app.database import get_pool
from app.graph.neo4j_client import get_driver
from app.retrieval.models import GraphPath, RetrievedChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Find seed entities
# ---------------------------------------------------------------------------

async def _find_seed_entities(
    query_embedding_256: list[float],
    top_k: int = 10,
) -> list[tuple[str, str, float]]:
    """Search entity_embeddings.text_embedding for nearest entities.

    Returns [(entity_name, entity_type, similarity), ...].
    """
    pool = await get_pool()
    emb_str = str(query_embedding_256)

    rows = await pool.fetch(
        """
        SELECT entity_name, entity_type,
               1 - (text_embedding <=> $1::vector) AS similarity
        FROM entity_embeddings
        WHERE text_embedding IS NOT NULL
          AND 1 - (text_embedding <=> $1::vector) > 0.75
        ORDER BY text_embedding <=> $1::vector
        LIMIT $2
        """,
        emb_str, top_k,
    )
    return [(r["entity_name"], r["entity_type"], float(r["similarity"])) for r in rows]


async def _fallback_entity_extraction(
    query: str,
    client: AsyncOpenAI,
) -> list[str]:
    """GPT-5.4 extracts entity names from query, matched against Neo4j."""
    try:
        resp = await client.chat.completions.create(
            model=settings.QUERY_CLASSIFIER_MODEL,
            messages=[
                {"role": "system", "content": (
                    "Extract entity names from the query. Return JSON: "
                    '{"entities": ["name1", "name2"]}. '
                    "Only include proper nouns, technical terms, or specific concepts."
                )},
                {"role": "user", "content": query},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=256,
        )
        data = json.loads(resp.choices[0].message.content)
        return data.get("entities", [])
    except Exception as exc:
        logger.warning("Fallback entity extraction failed: %s", exc)
        return []


async def _fuzzy_match_entities(
    names: list[str],
) -> list[tuple[str, str]]:
    """Match extracted names against Neo4j entities using case-insensitive search."""
    if not names:
        return []

    driver = await get_driver()
    matched: list[tuple[str, str]] = []

    async with driver.session() as session:
        for name in names:
            result = await session.run(
                "MATCH (e:Entity {status: 'active'}) "
                "WHERE toLower(e.name) CONTAINS toLower($name) "
                "RETURN e.name AS name, e.type AS type LIMIT 3",
                name=name,
            )
            async for r in result:
                matched.append((r["name"], r["type"]))

    return matched


# ---------------------------------------------------------------------------
# Step 2 — Neo4j traversal
# ---------------------------------------------------------------------------

async def _traverse_graph(
    entity_names: list[str],
    max_hops: int = 2,
) -> tuple[list[dict], list[GraphPath]]:
    """Traverse Neo4j from seed entities. Returns (related_entities, graph_paths)."""
    if not entity_names:
        return [], []

    driver = await get_driver()
    related: list[dict] = []
    paths: list[GraphPath] = []

    async with driver.session() as session:
        result = await session.run(
            """
            UNWIND $entity_names AS name
            MATCH (e:Entity {status: 'active'}) WHERE toLower(e.name) = toLower(name)
            OPTIONAL MATCH path = (e)-[r*1..2]-(related:Entity {status: 'active'})
            WITH e, related, path,
                 CASE size(relationships(path))
                     WHEN 1 THEN 1.0
                     WHEN 2 THEN 0.5
                     ELSE 0.25
                 END AS hop_weight
            WHERE related IS NOT NULL AND related <> e
            RETURN DISTINCT
                e.name AS seed_name,
                related.name AS related_name, related.type AS related_type,
                related.source_document_ids AS doc_ids,
                related.source_chunk_index AS chunk_index,
                hop_weight,
                [r IN relationships(path) | type(r)] AS rel_types
            """,
            entity_names=entity_names,
        )

        seen = set()
        async for r in result:
            key = (r["seed_name"], r["related_name"])
            if key in seen:
                continue
            seen.add(key)

            related.append({
                "name": r["related_name"],
                "type": r["related_type"],
                "doc_ids": r["doc_ids"] or [],
                "chunk_index": r["chunk_index"],
                "hop_weight": r["hop_weight"],
            })

            paths.append(GraphPath(
                entities=[r["seed_name"], r["related_name"]],
                relationships=r["rel_types"] or [],
                source_chunks=[],
                confidence=r["hop_weight"],
            ))

    # Hard limit on paths to prevent traversal runaway
    max_paths = settings.GRAPH_SEARCH_MAX_PATHS
    if len(paths) > max_paths:
        paths.sort(key=lambda p: p.confidence, reverse=True)
        paths = paths[:max_paths]
        retained = {name for p in paths for name in p.entities}
        related = [r for r in related if r["name"] in retained]

    return related, paths


# ---------------------------------------------------------------------------
# Step 3 — Resolve to chunks
# ---------------------------------------------------------------------------

async def _resolve_to_chunks(
    entity_data: list[dict],
    seed_entity_names: list[str],
    top_k: int,
    category_ids: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Map entities to chunks via (document_id, chunk_index)."""
    pool = await get_pool()

    # Collect (doc_id, chunk_index) pairs from related entities
    pairs: list[tuple[str, int]] = []
    for ent in entity_data:
        for doc_id in ent.get("doc_ids", []):
            if ent.get("chunk_index") is not None:
                pairs.append((doc_id, ent["chunk_index"]))

    if not pairs:
        return []

    # Also get chunks for seed entities from Neo4j
    driver = await get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            UNWIND $names AS name
            MATCH (e:Entity {status: 'active'})
            WHERE toLower(e.name) = toLower(name)
              AND e.source_document_ids IS NOT NULL
              AND e.source_chunk_index IS NOT NULL
            RETURN e.source_document_ids[0] AS doc_id, e.source_chunk_index AS chunk_index
            """,
            names=seed_entity_names,
        )
        async for r in result:
            if r["doc_id"] and r["chunk_index"] is not None:
                pairs.append((r["doc_id"], r["chunk_index"]))

    # Deduplicate
    pairs = list(set(pairs))

    # Fetch chunks from Neon
    if not pairs:
        return []

    doc_ids = [p[0] for p in pairs]
    chunk_indices = [p[1] for p in pairs]

    exclude_types = settings.RETRIEVAL_EXCLUDE_CHUNK_TYPES  # ["HEADING", "INDEX"]

    cat_clause = ""
    params: list = [doc_ids, chunk_indices, top_k, exclude_types]
    if category_ids:
        cat_clause = f"AND d.category_id = ANY(${len(params) + 1}::uuid[])"
        params.append(category_ids)

    rows = await pool.fetch(
        f"""
        SELECT c.id::text AS chunk_id, c.document_id::text, c.content,
               c.section_path, c.metadata, c.chunk_index,
               d.filename, d.version, d.ingested_at
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE d.status = 'active'
          AND (c.document_id::text, c.chunk_index) IN (
              SELECT unnest($1::text[]), unnest($2::int[])
          )
          AND COALESCE(c.metadata->>'chunk_type', 'PARAGRAPH') != ALL($4::text[])
          {cat_clause}
        LIMIT $3
        """,
        *params,
    )

    # Weight by hop distance
    weight_map: dict[tuple[str, int], float] = {}
    for ent in entity_data:
        for doc_id in ent.get("doc_ids", []):
            if ent.get("chunk_index") is not None:
                key = (doc_id, ent["chunk_index"])
                weight_map[key] = max(weight_map.get(key, 0.0), ent["hop_weight"])

    return [
        RetrievedChunk(
            chunk_id=r["chunk_id"],
            document_id=r["document_id"],
            content=r["content"],
            score=weight_map.get((r["document_id"], r["chunk_index"]), 0.5),
            section_path=r["section_path"],
            metadata=json.loads(r["metadata"]) if isinstance(r["metadata"], str) else dict(r["metadata"] or {}),
            source="graph",
            filename=r["filename"],
            version=r["version"],
            ingested_at=str(r["ingested_at"]),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def graph_search(
    query: str,
    query_embedding_256: list[float],
    max_hops: int | None = None,
    top_k: int | None = None,
    category_ids: list[str] | None = None,
) -> tuple[list[RetrievedChunk], list[GraphPath]]:
    """Entity match → Neo4j traverse → chunk retrieval. Never raises."""
    max_hops = max_hops or settings.GRAPH_SEARCH_MAX_HOPS
    top_k = top_k or settings.GRAPH_SEARCH_TOP_K

    try:
        # Step 1 — Find seed entities via embedding similarity
        seeds = await _find_seed_entities(
            query_embedding_256, settings.GRAPH_SEARCH_ENTITY_TOP_K,
        )
        seed_names = [s[0] for s in seeds]

        # Fallback if too few embedding matches
        if len(seed_names) < 2:
            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            extracted = await _fallback_entity_extraction(query, client)
            fuzzy_matched = await _fuzzy_match_entities(extracted)
            for name, _ in fuzzy_matched:
                if name not in seed_names:
                    seed_names.append(name)

        if not seed_names:
            logger.info("Graph search: no seed entities found")
            return [], []

        # Deduplicate seeds case-insensitively (e.g., "promoter" and "The promoter")
        seen_lower: set[str] = set()
        deduped: list[str] = []
        for name in seed_names:
            if name.lower() not in seen_lower:
                seen_lower.add(name.lower())
                deduped.append(name)
        seed_names = deduped

        # Filter hub entities with too many connections (breadth-first noise)
        max_degree = settings.GRAPH_SEARCH_MAX_SEED_DEGREE
        driver = await get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                UNWIND $names AS name
                MATCH (e:Entity {status: 'active'})
                WHERE toLower(e.name) = toLower(name)
                OPTIONAL MATCH (e)-[r]-()
                RETURN e.name AS name, count(r) AS degree
                """,
                names=seed_names,
            )
            degrees: dict[str, int] = {}
            async for r in result:
                degrees[r["name"]] = r["degree"]

        filtered = [n for n in seed_names if degrees.get(n, 0) <= max_degree]
        if filtered:
            seed_names = filtered
        else:
            # Keep at least the lowest-degree seed if all are hubs
            seed_names = sorted(seed_names, key=lambda n: degrees.get(n, 0))[:1]
            logger.info("Graph search: all seeds exceed degree %d, kept lowest: %s", max_degree, seed_names)

        logger.info("Graph search: %d seed entities: %s", len(seed_names), seed_names[:5])

        # Step 2 — Neo4j traversal
        related_entities, graph_paths = await _traverse_graph(seed_names, max_hops)

        # Step 3 — Resolve to chunks
        chunks = await _resolve_to_chunks(related_entities, seed_names, top_k, category_ids)

        # Attach chunk IDs to graph paths
        chunk_ids_set = {c.chunk_id for c in chunks}
        for path in graph_paths:
            path.source_chunks = list(chunk_ids_set)

        logger.info(
            "Graph search: %d related entities → %d chunks, %d paths",
            len(related_entities), len(chunks), len(graph_paths),
        )
        return chunks, graph_paths

    except Exception as exc:
        logger.warning("Graph search failed: %s", exc)
        return [], []
