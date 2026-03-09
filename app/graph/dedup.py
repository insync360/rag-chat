"""Deterministic entity and relationship deduplication."""

import logging
import math
import re

from app.config import settings
from app.graph.models import Entity, Relationship

logger = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz as _fuzz
except ImportError:
    _fuzz = None
    logger.warning("rapidfuzz not installed — fuzzy entity dedup disabled")

_TRAILING_PUNCT = re.compile(r"[.,;:!?]+$")


def _normalize_name(name: str) -> str:
    """Lowercase, strip whitespace, remove trailing punctuation."""
    return _TRAILING_PUNCT.sub("", name.strip().lower())


def deduplicate_entities(entities: list[Entity]) -> list[Entity]:
    """Deduplicate entities by (normalized_name, type). Keep highest confidence, merge properties."""
    if not entities:
        return []

    groups: dict[tuple[str, str], list[Entity]] = {}
    for e in entities:
        key = (_normalize_name(e.name), e.type)
        groups.setdefault(key, []).append(e)

    result: list[Entity] = []
    for group in groups.values():
        best = max(group, key=lambda e: e.confidence)
        merged_props = {}
        for e in group:
            merged_props.update(e.properties)
        best.properties = merged_props
        result.append(best)

    return result


def deduplicate_relationships(
    relationships: list[Relationship], entities: list[Entity],
) -> list[Relationship]:
    """Deduplicate relationships and remap entity names to canonical forms."""
    if not relationships:
        return []

    # Build name mapping: normalized -> canonical name from deduplicated entities
    canonical: dict[tuple[str, str], str] = {}
    for e in entities:
        canonical[(_normalize_name(e.name), e.type)] = e.name

    # Build entity type lookup: normalized_name -> set of types
    entity_types: dict[str, set[str]] = {}
    for e in entities:
        entity_types.setdefault(_normalize_name(e.name), set()).add(e.type)

    # Remap source/target names to canonical and deduplicate
    seen: dict[tuple[str, str, str], Relationship] = {}
    for r in relationships:
        norm_src = _normalize_name(r.source_entity)
        norm_tgt = _normalize_name(r.target_entity)

        # Resolve canonical names (try all type combinations)
        src_name = r.source_entity
        tgt_name = r.target_entity
        for t in entity_types.get(norm_src, set()):
            if (norm_src, t) in canonical:
                src_name = canonical[(norm_src, t)]
                break
        for t in entity_types.get(norm_tgt, set()):
            if (norm_tgt, t) in canonical:
                tgt_name = canonical[(norm_tgt, t)]
                break

        r.source_entity = src_name
        r.target_entity = tgt_name

        key = (norm_src, norm_tgt, r.type)
        if key not in seen or r.confidence > seen[key].confidence:
            seen[key] = r

    return list(seen.values())


# ---------------------------------------------------------------------------
# Enhanced dedup — fuzzy + embedding tiers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _merge_entity_group(entities: list[Entity]) -> Entity:
    """Merge a group of entities — highest confidence wins name + confidence, all properties merged."""
    best = max(entities, key=lambda e: e.confidence)
    merged_props: dict = {}
    for e in entities:
        merged_props.update(e.properties)
    return Entity(
        name=best.name,
        type=best.type,
        source_chunk_index=best.source_chunk_index,
        source_document_id=best.source_document_id,
        properties=merged_props,
        confidence=best.confidence,
    )


def _fuzzy_merge_groups(
    groups_by_type: dict[str, list[list[Entity]]],
    threshold: float,
) -> dict[str, list[list[Entity]]]:
    """Merge groups within each type using fuzzy string matching."""
    if _fuzz is None:
        return groups_by_type

    result: dict[str, list[list[Entity]]] = {}
    for entity_type, groups in groups_by_type.items():
        merged: list[list[Entity]] = []
        for group in groups:
            representative = group[0].name
            matched = False
            for existing in merged:
                existing_rep = existing[0].name
                score = _fuzz.token_sort_ratio(
                    _normalize_name(representative),
                    _normalize_name(existing_rep),
                )
                if score >= threshold:
                    existing.extend(group)
                    matched = True
                    break
            if not matched:
                merged.append(list(group))
        result[entity_type] = merged
    return result


async def _embedding_merge_groups(
    groups_by_type: dict[str, list[list[Entity]]],
    threshold: float,
) -> dict[str, list[list[Entity]]]:
    """Merge groups within each type using embedding cosine similarity."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    result: dict[str, list[list[Entity]]] = {}

    for entity_type, groups in groups_by_type.items():
        if len(groups) <= 1:
            result[entity_type] = groups
            continue

        # Batch embed all group representative names
        names = [groups[i][0].name for i in range(len(groups))]
        try:
            resp = await client.embeddings.create(
                model=settings.ENTITY_EMBEDDING_MODEL,
                input=names,
                dimensions=settings.ENTITY_EMBEDDING_DIMENSIONS,
            )
            embeddings = [item.embedding for item in resp.data]
        except Exception as exc:
            logger.warning("Embedding API failed for type %s: %s", entity_type, exc)
            result[entity_type] = groups
            continue

        # Greedy merge by cosine similarity
        merged: list[list[Entity]] = []
        merged_embeddings: list[list[float]] = []
        for i, group in enumerate(groups):
            emb = embeddings[i]
            matched = False
            for j, existing_emb in enumerate(merged_embeddings):
                if _cosine_similarity(emb, existing_emb) >= threshold:
                    merged[j].extend(group)
                    matched = True
                    break
            if not matched:
                merged.append(list(group))
                merged_embeddings.append(emb)
        result[entity_type] = merged

    return result


async def deduplicate_entities_enhanced(
    entities: list[Entity],
    existing_entities: list[Entity] | None = None,
) -> list[Entity]:
    """Three-tier dedup: exact → fuzzy → embedding. Scoped by entity type.

    When existing_entities provided, prepend them so dedup naturally merges new with existing.
    """
    if not entities and not existing_entities:
        return []

    all_entities = list(existing_entities or []) + list(entities)
    if not all_entities:
        return []

    # Tier 1: Exact — group by (normalized_name, type)
    exact_groups: dict[tuple[str, str], list[Entity]] = {}
    for e in all_entities:
        key = (_normalize_name(e.name), e.type)
        exact_groups.setdefault(key, []).append(e)

    # Organize by type for tier 2+3
    by_type: dict[str, list[list[Entity]]] = {}
    for (_, etype), group in exact_groups.items():
        by_type.setdefault(etype, []).append(group)

    # Tier 2: Fuzzy
    try:
        by_type = _fuzzy_merge_groups(by_type, settings.ENTITY_FUZZY_THRESHOLD)
    except Exception as exc:
        logger.warning("Fuzzy dedup failed, using exact only: %s", exc)

    # Tier 3: Embedding
    try:
        by_type = await _embedding_merge_groups(by_type, settings.ENTITY_EMBEDDING_THRESHOLD)
    except Exception as exc:
        logger.warning("Embedding dedup failed, using fuzzy+exact: %s", exc)

    # Flatten and merge each group
    result: list[Entity] = []
    for groups in by_type.values():
        for group in groups:
            result.append(_merge_entity_group(group))

    return result
