"""Deterministic entity and relationship deduplication."""

import re

from app.graph.models import Entity, Relationship

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
