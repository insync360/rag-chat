"""Reset all data (Neon + Neo4j) and ingest the project blueprint PDF end-to-end."""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import get_pool, close_pool
from app.graph.neo4j_client import get_driver, close_driver
from app.ingestion.pipeline import ingest_files, PipelineStatus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reset_and_ingest")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATHS = [
    PROJECT_ROOT / "the_real_estate_(regulation_and_development)_act,_2016.pdf",
]

TRUNCATE_SQL = """
TRUNCATE TABLE semantic_cache, query_logs, community_summary_embeddings,
               relation_embeddings, entity_embeddings, chunk_embeddings,
               ingestion_logs, chunks, documents CASCADE;
"""

CLEAR_NEO4J = "MATCH (n) DETACH DELETE n"


async def clear_neon() -> None:
    """Truncate all 9 Neon tables (FK-safe via CASCADE)."""
    pool = await get_pool()
    await pool.execute(TRUNCATE_SQL)
    logger.info("Neon: all tables truncated")


async def clear_neo4j() -> None:
    """Delete all nodes and relationships from Neo4j."""
    driver = await get_driver()
    async with driver.session() as session:
        result = await session.run(CLEAR_NEO4J)
        summary = await result.consume()
        logger.info(
            "Neo4j: deleted %d nodes, %d relationships",
            summary.counters.nodes_deleted,
            summary.counters.relationships_deleted,
        )


async def verify(pool, driver) -> None:
    """Print post-ingestion counts from Neon and Neo4j."""
    rows = await pool.fetch("""
        SELECT 'documents'       AS tbl, count(*) AS n FROM documents WHERE status = 'active'
        UNION ALL
        SELECT 'chunks',                 count(*)       FROM chunks
        UNION ALL
        SELECT 'chunk_embeddings',       count(*)       FROM chunk_embeddings
        UNION ALL
        SELECT 'hybrid_embeddings',      count(*)       FROM chunk_embeddings WHERE hybrid_embedding IS NOT NULL
        UNION ALL
        SELECT 'entity_embeddings',      count(*)       FROM entity_embeddings
        UNION ALL
        SELECT 'graphsage_embeddings',   count(*)       FROM entity_embeddings WHERE embedding IS NOT NULL
        UNION ALL
        SELECT 'community_summaries',    count(*)       FROM community_summary_embeddings
    """)
    logger.info("── Neon verification ──")
    for r in rows:
        logger.info("  %-25s %d", r["tbl"], r["n"])

    async with driver.session() as session:
        neo4j_checks = [
            ("active entities",  "MATCH (e:Entity {status: 'active'}) RETURN count(e) AS n"),
            ("relationships",    "MATCH ()-[r]-() RETURN count(r) AS n"),
            ("communities",      "MATCH (c:Community) RETURN count(c) AS n"),
        ]
        logger.info("── Neo4j verification ──")
        for label, cypher in neo4j_checks:
            result = await session.run(cypher)
            record = await result.single()
            logger.info("  %-25s %d", label, record["n"])


async def main() -> None:
    for p in PDF_PATHS:
        if not p.exists():
            logger.error("PDF not found: %s", p)
            sys.exit(1)

    t_total = time.perf_counter()

    # Step A+B: Clear databases
    logger.info("═══ Step 1/3: Clearing databases ═══")
    await clear_neon()
    await clear_neo4j()

    # Step C: Ingest
    names = ", ".join(p.name for p in PDF_PATHS)
    logger.info("═══ Step 2/3: Ingesting %d file(s): %s ═══", len(PDF_PATHS), names)
    result = await ingest_files(PDF_PATHS)

    for fr in result.files:
        if fr.status == PipelineStatus.COMPLETED:
            logger.info("[%s] completed — %d chunks, %d entities, %d relationships",
                        fr.filename, fr.chunk_count or 0, fr.entity_count or 0, fr.relationship_count or 0)
            t = fr.timings
            for label, ms in [
                ("Parse (LlamaParse)",     t.parse_ms),
                ("Version track",          t.version_track_ms),
                ("Chunk",                  t.chunk_ms),
                ("Enrich (GPT-4o-mini)",   t.enrich_ms),
                ("Graph extract (GPT-4o)", t.graph_extract_ms),
                ("Save chunks",            t.save_ms),
                ("Embed chunks",           t.embed_ms),
            ]:
                if ms > 0:
                    logger.info("  %-25s %8.1f ms", label, ms)
        else:
            logger.error("[%s] %s at step '%s': %s", fr.filename, fr.status.value, fr.step, fr.error)

    logger.info("Pipeline total: %.1f ms", result.total_duration_ms)

    if result.failed > 0:
        logger.error("%d file(s) failed", result.failed)
        await close_pool()
        await close_driver()
        sys.exit(1)

    # Step D: Verify
    logger.info("═══ Step 3/3: Verification ═══")
    pool = await get_pool()
    driver = await get_driver()
    await verify(pool, driver)

    total_s = time.perf_counter() - t_total
    logger.info("═══ Done in %.1f s ═══", total_s)

    await close_pool()
    await close_driver()


if __name__ == "__main__":
    asyncio.run(main())
