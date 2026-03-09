"""Unit tests for enhanced entity deduplication (fuzzy + embedding)."""

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.graph.dedup import (
    _cosine_similarity,
    _merge_entity_group,
    deduplicate_entities_enhanced,
)
from app.graph.models import Entity


def _entity(name="Acme Corporation", type="Organization", confidence=0.9,
            props=None, chunk_index=0, doc_id="doc-123"):
    return Entity(
        name=name, type=type, source_chunk_index=chunk_index,
        source_document_id=doc_id, confidence=confidence,
        properties=props or {},
    )


# --- Fuzzy merge tests ---


@pytest.mark.asyncio
async def test_fuzzy_merges_similar_names():
    """'Smith, John' + 'John Smith' same type → merged (score ~95)."""
    entities = [
        _entity("Smith, John", type="Person", confidence=0.8),
        _entity("John Smith", type="Person", confidence=0.95),
    ]
    with patch("app.graph.dedup._embedding_merge_groups", new_callable=AsyncMock,
               side_effect=lambda g, t: g):
        result = await deduplicate_entities_enhanced(entities)

    assert len(result) == 1
    assert result[0].confidence == 0.95


@pytest.mark.asyncio
async def test_fuzzy_below_threshold_kept_separate():
    """Dissimilar names not merged by fuzzy."""
    entities = [
        _entity("Acme Corporation", confidence=0.9),
        _entity("Global Industries", confidence=0.85),
    ]
    with patch("app.graph.dedup._embedding_merge_groups", new_callable=AsyncMock,
               side_effect=lambda g, t: g):
        result = await deduplicate_entities_enhanced(entities)

    assert len(result) == 2
    names = {e.name for e in result}
    assert "Acme Corporation" in names
    assert "Global Industries" in names


# --- Embedding merge tests ---


@pytest.mark.asyncio
async def test_embedding_merges_semantic_equivalents():
    """'CEO' + 'Chief Executive Officer' merged via embedding similarity (mocked)."""
    entities = [
        _entity("CEO", type="Role", confidence=0.8),
        _entity("Chief Executive Officer", type="Role", confidence=0.95),
    ]

    # Mock embeddings that are very similar (cosine >= 0.92)
    mock_resp = MagicMock()
    mock_resp.data = [
        MagicMock(embedding=[1.0, 0.0, 0.0]),
        MagicMock(embedding=[0.99, 0.1, 0.0]),
    ]
    mock_client = AsyncMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_resp)

    with patch("openai.AsyncOpenAI", return_value=mock_client), \
         patch("app.graph.dedup._fuzz", None):  # disable fuzzy to test embedding only
        result = await deduplicate_entities_enhanced(entities)

    assert len(result) == 1
    assert result[0].confidence == 0.95


# --- Type separation ---


@pytest.mark.asyncio
async def test_different_types_never_merged():
    """'Apple' Org + 'Apple' Product kept separate even with identical names."""
    entities = [
        _entity("Apple", type="Organization", confidence=0.9),
        _entity("Apple", type="Product", confidence=0.85),
    ]
    with patch("app.graph.dedup._embedding_merge_groups", new_callable=AsyncMock,
               side_effect=lambda g, t: g):
        result = await deduplicate_entities_enhanced(entities)

    assert len(result) == 2
    types = {e.type for e in result}
    assert types == {"Organization", "Product"}


# --- Existing entity merge ---


@pytest.mark.asyncio
async def test_existing_entities_merged_with_new():
    """Existing Neo4j entity merges with new extraction via fuzzy."""
    existing = [_entity("Acme Corporation", confidence=0.7, doc_id="old-doc")]
    new = [_entity("Acme Corporations", confidence=0.95, doc_id="new-doc")]

    with patch("app.graph.dedup._embedding_merge_groups", new_callable=AsyncMock,
               side_effect=lambda g, t: g):
        result = await deduplicate_entities_enhanced(new, existing_entities=existing)

    assert len(result) == 1
    assert result[0].confidence == 0.95


# --- Confidence ---


@pytest.mark.asyncio
async def test_highest_confidence_wins():
    """Merge picks highest confidence."""
    entities = [
        _entity("John Smith", type="Person", confidence=0.5),
        _entity("john smith", type="Person", confidence=0.99),
    ]
    with patch("app.graph.dedup._embedding_merge_groups", new_callable=AsyncMock,
               side_effect=lambda g, t: g):
        result = await deduplicate_entities_enhanced(entities)

    assert len(result) == 1
    assert result[0].confidence == 0.99


# --- Properties ---


@pytest.mark.asyncio
async def test_properties_merged_across_fuzzy():
    """All properties present after fuzzy merge."""
    entities = [
        _entity("Acme Corporation", confidence=0.8, props={"sector": "tech"}),
        _entity("Acme Corporations", confidence=0.95, props={"hq": "NYC"}),
    ]
    with patch("app.graph.dedup._embedding_merge_groups", new_callable=AsyncMock,
               side_effect=lambda g, t: g):
        result = await deduplicate_entities_enhanced(entities)

    assert len(result) == 1
    assert result[0].properties["sector"] == "tech"
    assert result[0].properties["hq"] == "NYC"


# --- Fallbacks ---


@pytest.mark.asyncio
async def test_fallback_on_rapidfuzz_unavailable():
    """Degrades to exact-only when rapidfuzz unavailable."""
    entities = [
        _entity("Acme Corp", confidence=0.8),
        _entity("Acme Corporation", confidence=0.95),
    ]
    with patch("app.graph.dedup._fuzz", None), \
         patch("app.graph.dedup._embedding_merge_groups", new_callable=AsyncMock,
               side_effect=lambda g, t: g):
        result = await deduplicate_entities_enhanced(entities)

    # Without fuzzy, "Acme Corp" ≠ "Acme Corporation" (not exact match)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_fallback_on_embedding_error():
    """Degrades to fuzzy+exact when embedding API fails."""
    entities = [
        _entity("CEO", type="Role", confidence=0.8),
        _entity("Chief Executive Officer", type="Role", confidence=0.95),
    ]
    with patch("app.graph.dedup._embedding_merge_groups", new_callable=AsyncMock,
               side_effect=Exception("API timeout")):
        result = await deduplicate_entities_enhanced(entities)

    # CEO vs Chief Executive Officer — fuzzy score too low, kept separate
    assert len(result) == 2


# --- Relationship remap with fuzzy ---


@pytest.mark.asyncio
async def test_relationship_dedup_with_fuzzy_canonical():
    """Relationships remap source/target via fuzzy-merged canonical names."""
    from app.graph.dedup import deduplicate_relationships
    from app.graph.models import Relationship

    entities = [
        _entity("Acme Corp", confidence=0.8),
        _entity("Acme Corporation", confidence=0.95),
    ]
    with patch("app.graph.dedup._embedding_merge_groups", new_callable=AsyncMock,
               side_effect=lambda g, t: g):
        deduped = await deduplicate_entities_enhanced(entities)

    rels = [
        Relationship(
            source_entity="Acme Corp", target_entity="John Smith",
            type="EMPLOYS", source_chunk_index=0,
            source_document_id="doc-123", confidence=0.9,
        ),
    ]
    result = deduplicate_relationships(rels, deduped)

    assert len(result) == 1
    # Source should be remapped to the canonical (highest confidence) name
    assert result[0].source_entity == deduped[0].name


# --- Helper tests ---


def test_cosine_similarity():
    assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
    assert _cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)
    assert _cosine_similarity([1, 0, 0], [-1, 0, 0]) == pytest.approx(-1.0)
    assert _cosine_similarity([0, 0, 0], [1, 0, 0]) == pytest.approx(0.0)


def test_merge_entity_group():
    group = [
        _entity("Acme Corp", confidence=0.8, props={"a": 1}),
        _entity("Acme Corporation", confidence=0.95, props={"b": 2}),
    ]
    merged = _merge_entity_group(group)
    assert merged.name == "Acme Corporation"
    assert merged.confidence == 0.95
    assert merged.properties == {"a": 1, "b": 2}
