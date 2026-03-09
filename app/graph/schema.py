"""Idempotent Neo4j constraint and index setup."""

import logging

from app.graph.neo4j_client import get_driver

logger = logging.getLogger(__name__)

_SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT entity_unique IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE",
    "CREATE INDEX entity_document_ids IF NOT EXISTS FOR (e:Entity) ON (e.source_document_ids)",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
    "CREATE INDEX entity_status IF NOT EXISTS FOR (e:Entity) ON (e.status)",
    "CREATE INDEX entity_community IF NOT EXISTS FOR (e:Entity) ON (e.community_id)",
    "CREATE CONSTRAINT community_unique IF NOT EXISTS "
    "FOR (c:Community) REQUIRE c.community_id IS UNIQUE",
]

# Migration: convert old source_document_id → source_document_ids array
_MIGRATION_STATEMENTS = [
    """
    MATCH (e:Entity)
    WHERE e.source_document_id IS NOT NULL AND e.source_document_ids IS NULL
    SET e.source_document_ids = [e.source_document_id]
    REMOVE e.source_document_id
    """,
    """
    MATCH (e:Entity)
    WHERE e.status IS NULL
    SET e.status = 'active'
    """,
]


async def ensure_schema() -> None:
    """Create constraints, indexes, and run migrations if needed."""
    driver = await get_driver()
    async with driver.session() as session:
        # Run migrations first
        for stmt in _MIGRATION_STATEMENTS:
            await session.run(stmt)
        # Drop old index if it exists (ignore error if not present)
        try:
            await session.run("DROP INDEX entity_document IF EXISTS")
        except Exception:
            pass
        for stmt in _SCHEMA_STATEMENTS:
            await session.run(stmt)
    logger.info("Neo4j schema ensured (%d statements)", len(_SCHEMA_STATEMENTS))
