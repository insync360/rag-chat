"""Unit tests for graph entity/relationship deduplication."""

import pytest

from app.graph.dedup import deduplicate_entities, deduplicate_relationships
from app.graph.models import Entity, Relationship


def _entity(name="Acme Corporation", type="Organization", confidence=0.9, props=None):
    return Entity(
        name=name, type=type, source_chunk_index=0,
        source_document_id="doc-123", confidence=confidence,
        properties=props or {},
    )


def _rel(source="Acme Corporation", target="John Smith", type="EMPLOYS", confidence=0.9):
    return Relationship(
        source_entity=source, target_entity=target, type=type,
        source_chunk_index=0, source_document_id="doc-123",
        confidence=confidence, properties={},
    )


# --- Entity Dedup Tests ---


def test_exact_duplicate_merged():
    entities = [
        _entity("Acme Corporation", confidence=0.8),
        _entity("Acme Corporation", confidence=0.95),
        _entity("acme corporation", confidence=0.7),  # normalized match
    ]
    result = deduplicate_entities(entities)

    assert len(result) == 1
    assert result[0].confidence == 0.95


def test_different_types_kept_separate():
    entities = [
        _entity("Apple", type="Organization"),
        _entity("Apple", type="Product"),
    ]
    result = deduplicate_entities(entities)

    assert len(result) == 2
    types = {e.type for e in result}
    assert types == {"Organization", "Product"}


def test_highest_confidence_wins():
    entities = [
        _entity("John Smith", type="Person", confidence=0.5),
        _entity("john smith", type="Person", confidence=0.99),
        _entity("John Smith.", type="Person", confidence=0.6),  # trailing punct
    ]
    result = deduplicate_entities(entities)

    assert len(result) == 1
    assert result[0].confidence == 0.99


def test_properties_merged():
    entities = [
        _entity("John Smith", type="Person", confidence=0.9, props={"role": "CEO"}),
        _entity("john smith", type="Person", confidence=0.8, props={"department": "Executive"}),
    ]
    result = deduplicate_entities(entities)

    assert len(result) == 1
    assert result[0].properties["role"] == "CEO"
    assert result[0].properties["department"] == "Executive"


def test_empty_input():
    assert deduplicate_entities([]) == []
    assert deduplicate_relationships([], []) == []


# --- Relationship Dedup Tests ---


def test_relationship_dedup():
    entities = [_entity("Acme Corporation"), _entity("John Smith", type="Person")]
    rels = [
        _rel("Acme Corporation", "John Smith", "EMPLOYS", confidence=0.8),
        _rel("acme corporation", "john smith", "EMPLOYS", confidence=0.95),
    ]
    result = deduplicate_relationships(rels, entities)

    assert len(result) == 1
    assert result[0].confidence == 0.95
