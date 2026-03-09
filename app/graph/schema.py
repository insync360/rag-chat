"""Idempotent Neo4j constraint and index setup."""

import logging

from app.graph.neo4j_client import get_driver

logger = logging.getLogger(__name__)

_SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT entity_unique IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE",
    "CREATE INDEX entity_document IF NOT EXISTS FOR (e:Entity) ON (e.source_document_id)",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
]


async def ensure_schema() -> None:
    """Create constraints and indexes if they don't exist."""
    driver = await get_driver()
    async with driver.session() as session:
        for stmt in _SCHEMA_STATEMENTS:
            await session.run(stmt)
    logger.info("Neo4j schema ensured (%d statements)", len(_SCHEMA_STATEMENTS))
