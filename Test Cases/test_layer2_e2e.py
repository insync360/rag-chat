"""Layer 2 E2E test: Neon chunks → graph extraction → Neo4j verification.

Fetches existing documents/chunks from Neon, runs graph extraction + community
detection, then verifies results in Neo4j with sample queries.

Run: .venv/Scripts/python "Test Cases/test_layer2_e2e.py"
"""

import asyncio
import json
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def run():
    from app.config import settings
    from app.database import get_pool, close_pool
    from app.graph.neo4j_client import get_driver, close_driver
    from app.graph import extract_and_store_graph
    from app.graph.community import detect_communities
    from app.ingestion.chunker import Chunk
    from app.ingestion.version_tracker import VersionTracker

    # ── Step 1: Setup & connection verification ──
    logger.info("=" * 60)
    logger.info("STEP 1: Connection verification")
    logger.info("=" * 60)

    pool = await get_pool()
    val = await pool.fetchval("SELECT 1")
    logger.info("Neon connected (SELECT 1 = %s)", val)

    driver = await get_driver()
    await driver.verify_connectivity()
    logger.info("Neo4j connected (%s)", settings.NEO4J_URI)

    logger.info("Feature flags:")
    logger.info("  GRAPH_EXTRACTION_ENABLED:   %s", settings.GRAPH_EXTRACTION_ENABLED)
    logger.info("  COREF_ENABLED:              %s", settings.COREF_ENABLED)
    logger.info("  INCREMENTAL_GRAPH_ENABLED:  %s", settings.INCREMENTAL_GRAPH_ENABLED)
    logger.info("  COMMUNITY_DETECTION_ENABLED:%s", settings.COMMUNITY_DETECTION_ENABLED)
    logger.info("  COMMUNITY_SUMMARY_ENABLED:  %s", settings.COMMUNITY_SUMMARY_ENABLED)

    # ── Step 2: Fetch documents & chunks from Neon ──
    logger.info("=" * 60)
    logger.info("STEP 2: Fetching documents & chunks from Neon")
    logger.info("=" * 60)

    doc_rows = await pool.fetch(
        "SELECT id, filename, content_hash, version, status, "
        "page_count, ingested_at, metadata "
        "FROM documents WHERE status = 'active' ORDER BY ingested_at"
    )
    if not doc_rows:
        logger.error("No active documents in Neon. Run Layer 1 ingestion first.")
        await close_pool()
        await close_driver()
        sys.exit(1)

    doc_records = [VersionTracker._row_to_record(r) for r in doc_rows]
    logger.info("Found %d active document(s)", len(doc_records))

    docs_with_chunks: list[tuple] = []
    for doc in doc_records:
        chunk_rows = await pool.fetch(
            "SELECT document_id, chunk_index, content, token_count, "
            "section_path, has_table, has_code, overlap_tokens, metadata "
            "FROM chunks WHERE document_id = $1::uuid ORDER BY chunk_index",
            doc.id,
        )
        chunks = []
        for r in chunk_rows:
            meta = r["metadata"]
            if isinstance(meta, str):
                meta = json.loads(meta)
            chunks.append(Chunk(
                document_id=str(r["document_id"]),
                chunk_index=r["chunk_index"],
                content=r["content"],
                token_count=r["token_count"],
                section_path=r["section_path"] or "",
                has_table=r["has_table"],
                has_code=r["has_code"],
                overlap_tokens=r["overlap_tokens"],
                metadata=meta if meta else {},
            ))
        docs_with_chunks.append((doc, chunks))
        logger.info("  %s — v%d, %d pages, %d chunks",
                     doc.filename, doc.version, doc.page_count, len(chunks))

    total_chunks = sum(len(c) for _, c in docs_with_chunks)
    logger.info("Total: %d documents, %d chunks", len(docs_with_chunks), total_chunks)

    # ── Step 3: Clear Neo4j for clean run ──
    logger.info("=" * 60)
    logger.info("STEP 3: Clearing Neo4j (clean test run)")
    logger.info("=" * 60)

    async with driver.session() as session:
        result = await session.run("MATCH (n) RETURN count(n) AS cnt")
        record = await result.single()
        node_count = record["cnt"] if record else 0
        logger.info("Existing nodes in Neo4j: %d", node_count)

        if node_count > 0:
            await session.run("MATCH (n) DETACH DELETE n")
            logger.info("Cleared all nodes")

    # ── Step 4: Graph extraction per document ──
    logger.info("=" * 60)
    logger.info("STEP 4: Graph extraction")
    logger.info("=" * 60)

    total_entities = 0
    total_relationships = 0

    for doc, chunks in docs_with_chunks:
        if not chunks:
            logger.info("  Skipping %s (no chunks)", doc.filename)
            continue

        logger.info("  Extracting: %s (%d chunks)...", doc.filename, len(chunks))
        t0 = time.perf_counter()

        result = await extract_and_store_graph(
            chunks, doc, run_community_detection=False,
        )

        elapsed = time.perf_counter() - t0
        logger.info("    → %d entities, %d relationships (%.1fs) skipped=%s error=%s",
                     result.entity_count, result.relationship_count,
                     elapsed, result.skipped, result.error)

        total_entities += result.entity_count
        total_relationships += result.relationship_count

    logger.info("Extraction totals: %d entities, %d relationships",
                total_entities, total_relationships)

    # ── Step 5: Community detection ──
    logger.info("=" * 60)
    logger.info("STEP 5: Community detection")
    logger.info("=" * 60)

    t0 = time.perf_counter()
    comm_result = await detect_communities()
    elapsed = time.perf_counter() - t0

    logger.info("Communities: %d (from %d entities) in %.1fs | skipped=%s error=%s",
                comm_result.total_communities, comm_result.total_entities,
                elapsed, comm_result.skipped, comm_result.error)

    for c in comm_result.communities:
        logger.info("  Community %d: %d entities — %s",
                     c.community_id, c.size,
                     c.summary[:100] if c.summary else "(no summary)")

    # ── Step 6: Neo4j verification queries ──
    logger.info("=" * 60)
    logger.info("STEP 6: Neo4j verification")
    logger.info("=" * 60)

    async with driver.session() as session:
        # Counts
        r = await session.run("MATCH (e:Entity) WHERE e.status = 'active' RETURN count(e) AS cnt")
        rec = await result.single() if False else await r.single()
        logger.info("Active entities:  %d", rec["cnt"])

        r = await session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
        rec = await r.single()
        logger.info("Relationships:    %d", rec["cnt"])

        r = await session.run("MATCH (c:Community) RETURN count(c) AS cnt")
        rec = await r.single()
        logger.info("Communities:      %d", rec["cnt"])

        # Sample entities
        logger.info("")
        logger.info("Top 10 entities by confidence:")
        r = await session.run(
            "MATCH (e:Entity) WHERE e.status = 'active' "
            "RETURN e.name AS name, e.type AS type, "
            "e.confidence AS confidence, e.community_id AS community_id "
            "ORDER BY e.confidence DESC LIMIT 10"
        )
        records = [dict(rec) async for rec in r]
        for rec in records:
            logger.info("  %-30s  type=%-12s  conf=%.2f  community=%s",
                         rec["name"], rec["type"],
                         rec["confidence"] or 0,
                         rec["community_id"])

        # Sample relationships
        logger.info("")
        logger.info("Top 10 relationships by confidence:")
        r = await session.run(
            "MATCH (s:Entity)-[r]->(t:Entity) "
            "RETURN s.name AS source, type(r) AS rel_type, t.name AS target, "
            "r.confidence AS confidence "
            "ORDER BY r.confidence DESC LIMIT 10"
        )
        records = [dict(rec) async for rec in r]
        for rec in records:
            logger.info("  %-25s -[%s]-> %-25s  conf=%.2f",
                         rec["source"], rec["rel_type"], rec["target"],
                         rec["confidence"] or 0)

        # Community summaries
        logger.info("")
        logger.info("Community summaries:")
        r = await session.run(
            "MATCH (c:Community) "
            "RETURN c.community_id AS community_id, "
            "c.entity_count AS entity_count, c.summary AS summary "
            "ORDER BY c.entity_count DESC"
        )
        records = [dict(rec) async for rec in r]
        for rec in records:
            logger.info("  Community %s: %d entities — %s",
                         rec["community_id"], rec["entity_count"] or 0,
                         (rec["summary"][:100] + "...") if rec["summary"] and len(rec["summary"]) > 100 else rec["summary"])

        # Entities per community
        logger.info("")
        logger.info("Top 5 communities by size:")
        r = await session.run(
            "MATCH (e:Entity) WHERE e.community_id IS NOT NULL AND e.status = 'active' "
            "RETURN e.community_id AS community_id, "
            "collect(e.name) AS members, count(e) AS size "
            "ORDER BY size DESC LIMIT 5"
        )
        records = [dict(rec) async for rec in r]
        for rec in records:
            members = rec["members"]
            preview = ", ".join(members[:5])
            if len(members) > 5:
                preview += f", ... (+{len(members) - 5} more)"
            logger.info("  Community %s (%d members): %s",
                         rec["community_id"], rec["size"], preview)

    # ── Step 7: Cleanup ──
    logger.info("=" * 60)
    logger.info("STEP 7: Cleanup")
    logger.info("=" * 60)

    await close_driver()
    await close_pool()
    logger.info("Connections closed")

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("LAYER 2 E2E SUMMARY")
    logger.info("  Documents:     %d", len(docs_with_chunks))
    logger.info("  Chunks:        %d", total_chunks)
    logger.info("  Entities:      %d", total_entities)
    logger.info("  Relationships: %d", total_relationships)
    logger.info("  Communities:   %d", comm_result.total_communities)
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(run())
