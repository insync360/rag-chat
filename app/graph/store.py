"""Persist entities and relationships to Neo4j."""

import logging
from datetime import datetime, timezone

from app.config import settings
from app.graph.models import Entity, Relationship
from app.graph.neo4j_client import get_driver

logger = logging.getLogger(__name__)


def _batch(items: list, size: int) -> list[list]:
    """Split a list into batches."""
    return [items[i:i + size] for i in range(0, len(items), size)]


async def store_graph(
    entities: list[Entity], relationships: list[Relationship], document_id: str,
) -> None:
    """MERGE entities and relationships into Neo4j."""
    driver = await get_driver()
    now = datetime.now(timezone.utc).isoformat()

    # Upsert entities in batches
    for batch in _batch(entities, settings.GRAPH_BATCH_SIZE):
        params = [
            {
                "name": e.name,
                "type": e.type,
                "doc_id": e.source_document_id,
                "chunk_index": e.source_chunk_index,
                "confidence": e.confidence,
                "properties": e.properties,
            }
            for e in batch
        ]
        async with driver.session() as session:
            await session.run(
                """
                UNWIND $entities AS e
                MERGE (n:Entity {name: e.name, type: e.type})
                ON CREATE SET
                    n.source_document_id = e.doc_id,
                    n.source_chunk_index = e.chunk_index,
                    n.confidence = e.confidence,
                    n.properties = e.properties,
                    n.created_at = $now,
                    n.updated_at = $now
                ON MATCH SET
                    n.confidence = CASE WHEN e.confidence > n.confidence
                        THEN e.confidence ELSE n.confidence END,
                    n.properties = e.properties,
                    n.updated_at = $now
                """,
                entities=params, now=now,
            )

    # Upsert relationships — group by type, one query per type
    by_type: dict[str, list[Relationship]] = {}
    for r in relationships:
        by_type.setdefault(r.type, []).append(r)

    for rel_type, rels in by_type.items():
        for batch in _batch(rels, settings.GRAPH_BATCH_SIZE):
            params = [
                {
                    "source_name": r.source_entity,
                    "target_name": r.target_entity,
                    "doc_id": r.source_document_id,
                    "chunk_index": r.source_chunk_index,
                    "confidence": r.confidence,
                    "properties": r.properties,
                }
                for r in batch
            ]
            # Dynamic relationship type requires string interpolation.
            # rel_type is always derived from LLM output that's been parsed
            # into a Relationship dataclass — not user input.
            cypher = f"""
                UNWIND $rels AS r
                MATCH (s:Entity {{name: r.source_name}})
                MATCH (t:Entity {{name: r.target_name}})
                MERGE (s)-[rel:`{rel_type}`]->(t)
                ON CREATE SET
                    rel.source_document_id = r.doc_id,
                    rel.source_chunk_index = r.chunk_index,
                    rel.confidence = r.confidence,
                    rel.properties = r.properties,
                    rel.created_at = $now
                ON MATCH SET
                    rel.confidence = CASE WHEN r.confidence > rel.confidence
                        THEN r.confidence ELSE rel.confidence END,
                    rel.properties = r.properties,
                    rel.updated_at = $now
            """
            async with driver.session() as session:
                await session.run(cypher, rels=params, now=now)

    logger.info(
        "Stored %d entities, %d relationships for doc %s",
        len(entities), len(relationships), document_id[:12],
    )


async def clear_document_graph(document_id: str) -> None:
    """Remove all entities (and their relationships) for a document."""
    driver = await get_driver()
    async with driver.session() as session:
        await session.run(
            "MATCH (e:Entity {source_document_id: $doc_id}) DETACH DELETE e",
            doc_id=document_id,
        )
    logger.info("Cleared graph for document %s", document_id[:12])
