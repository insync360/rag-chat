"""Hybrid chunk-entity embeddings — combine truncated text with graph structure.

For each chunk, concatenates:
- Chunk text (512 dims) — MRL truncation of 2000-dim embedding, L2-normalized
- Entity structural mean (128 dims) — mean-pooled GraphSAGE, L2-normalized
- Entity TransE mean (128 dims) — mean-pooled TransE, L2-normalized
- Final 768-dim vector L2-normalized

No additional OpenAI API calls — reuses existing embeddings from both tables.
Chunks without entities get zero-filled entity portions (text-only hybrid).
"""

import json
import logging
import math

from app.config import settings
from app.database import get_pool
from app.graph.models import HybridEmbeddingResult
from app.graph.neo4j_client import get_driver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vector utilities (pure math, no numpy/torch)
# ---------------------------------------------------------------------------

def _parse_pg_vector(text: str) -> list[float]:
    """Parse pgvector text representation '[0.1,0.2,...]' to list."""
    return json.loads(text)


def _l2_normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector. Returns zero vector if near-zero norm."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-12:
        return vec
    return [x / norm for x in vec]


def _truncate_and_normalize(vec: list[float], dim: int) -> list[float]:
    """MRL truncation: take first `dim` elements, then L2-normalize."""
    return _l2_normalize(vec[:dim])


def _mean_pool(vecs: list[list[float]], dim: int) -> list[float]:
    """Mean-pool vectors. Returns zero vector if empty."""
    if not vecs:
        return [0.0] * dim
    if len(vecs) == 1:
        return list(vecs[0])
    n = len(vecs)
    return [sum(v[i] for v in vecs) / n for i in range(dim)]


def _build_hybrid_embedding(
    chunk_emb: list[float],
    structural_embs: list[list[float]],
    transe_embs: list[list[float]],
) -> list[float]:
    """Build 768-dim hybrid embedding from chunk + entity embeddings."""
    text_dim = settings.HYBRID_CHUNK_TEXT_DIM
    struct_dim = settings.GRAPHSAGE_OUTPUT_DIM
    transe_dim = settings.TRANSE_DIM

    text_part = _truncate_and_normalize(chunk_emb, text_dim)
    struct_part = _l2_normalize(_mean_pool(structural_embs, struct_dim))
    transe_part = _l2_normalize(_mean_pool(transe_embs, transe_dim))

    hybrid = text_part + struct_part + transe_part
    return _l2_normalize(hybrid)


# ---------------------------------------------------------------------------
# Neon reads
# ---------------------------------------------------------------------------

async def _read_chunk_data(
    document_ids: list[str] | None,
) -> list[dict]:
    """Read chunk IDs + embeddings from Neon."""
    pool = await get_pool()

    if document_ids:
        rows = await pool.fetch(
            "SELECT c.id AS chunk_id, c.document_id::text, c.chunk_index, "
            "ce.embedding::text AS embedding "
            "FROM chunks c JOIN chunk_embeddings ce ON ce.chunk_id = c.id "
            "WHERE c.document_id = ANY($1::uuid[])",
            document_ids,
        )
    else:
        rows = await pool.fetch(
            "SELECT c.id AS chunk_id, c.document_id::text, c.chunk_index, "
            "ce.embedding::text AS embedding "
            "FROM chunks c JOIN chunk_embeddings ce ON ce.chunk_id = c.id",
        )

    return [dict(r) for r in rows]


async def _read_entity_embeddings() -> dict[tuple[str, str], dict]:
    """Read structural + TransE embeddings from Neon entity_embeddings table.

    Returns {(name, type): {"structural": [...], "transe": [...] | None}}.
    """
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT entity_name, entity_type, "
        "embedding::text AS structural, "
        "transe_embedding::text AS transe "
        "FROM entity_embeddings"
    )

    result: dict[tuple[str, str], dict] = {}
    for r in rows:
        structural = _parse_pg_vector(r["structural"]) if r["structural"] else None
        transe = _parse_pg_vector(r["transe"]) if r["transe"] else None
        result[(r["entity_name"], r["entity_type"])] = {
            "structural": structural,
            "transe": transe,
        }

    return result


# ---------------------------------------------------------------------------
# Neo4j read — entity-chunk mapping
# ---------------------------------------------------------------------------

async def _read_entity_chunk_map() -> dict[tuple[str, int], list[tuple[str, str]]]:
    """Read entity-to-chunk mapping from Neo4j.

    Returns {(document_id, chunk_index): [(entity_name, entity_type), ...]}.

    Uses source_document_ids[0] only — source_chunk_index is set ON CREATE and
    only valid for the first extraction document. V1 limitation for multi-doc
    entities (see plan notes).
    """
    driver = await get_driver()
    async with driver.session() as session:
        result = await session.run(
            "MATCH (e:Entity) WHERE e.status = 'active' "
            "AND e.source_chunk_index IS NOT NULL "
            "AND e.source_document_ids IS NOT NULL "
            "RETURN e.name AS name, e.type AS type, "
            "e.source_chunk_index AS chunk_index, "
            "e.source_document_ids[0] AS document_id"
        )

        mapping: dict[tuple[str, int], list[tuple[str, str]]] = {}
        async for r in result:
            key = (r["document_id"], r["chunk_index"])
            mapping.setdefault(key, []).append((r["name"], r["type"]))

    return mapping


# ---------------------------------------------------------------------------
# Neon storage
# ---------------------------------------------------------------------------

async def _store_hybrid_embeddings(
    updates: list[tuple[str, str]],
) -> None:
    """Batch UPDATE hybrid_embedding on chunk_embeddings rows.

    updates: list of (chunk_id, embedding_str) tuples.
    """
    pool = await get_pool()
    await pool.executemany(
        "UPDATE chunk_embeddings SET hybrid_embedding = $2::vector "
        "WHERE chunk_id = $1::uuid",
        updates,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def generate_hybrid_embeddings(
    document_ids: list[str] | None = None,
) -> HybridEmbeddingResult:
    """Generate hybrid chunk-entity embeddings. Never raises."""
    hybrid_dim = settings.HYBRID_CHUNK_TEXT_DIM + settings.GRAPHSAGE_OUTPUT_DIM + settings.TRANSE_DIM

    if not settings.HYBRID_CHUNK_ENTITY_ENABLED:
        return HybridEmbeddingResult(
            chunk_count=0, embedding_dim=hybrid_dim, skipped=True,
        )

    try:
        # 1. Read chunk data from Neon
        chunks = await _read_chunk_data(document_ids)

        if not chunks:
            return HybridEmbeddingResult(
                chunk_count=0, embedding_dim=hybrid_dim, skipped=False,
            )

        # 2. Read entity-chunk map from Neo4j
        entity_chunk_map = await _read_entity_chunk_map()

        # 3. Read entity embeddings from Neon
        entity_embs = await _read_entity_embeddings()

        # 4. Build hybrid embedding per chunk
        updates: list[tuple[str, str]] = []
        with_entities = 0

        for chunk in chunks:
            chunk_emb = _parse_pg_vector(chunk["embedding"])
            doc_id = chunk["document_id"]
            chunk_idx = chunk["chunk_index"]

            # Look up entities for this chunk
            entity_keys = entity_chunk_map.get((doc_id, chunk_idx), [])

            structural_embs: list[list[float]] = []
            transe_embs: list[list[float]] = []

            for name, etype in entity_keys:
                emb_data = entity_embs.get((name, etype))
                if emb_data is None:
                    continue  # Entity not in entity_embeddings (GraphSAGE may have failed)
                if emb_data["structural"] is not None:
                    structural_embs.append(emb_data["structural"])
                if emb_data["transe"] is not None:
                    transe_embs.append(emb_data["transe"])

            if structural_embs or transe_embs:
                with_entities += 1

            hybrid = _build_hybrid_embedding(chunk_emb, structural_embs, transe_embs)
            updates.append((str(chunk["chunk_id"]), str(hybrid)))

        # 5. Store
        await _store_hybrid_embeddings(updates)

        logger.info(
            "Hybrid embeddings: %d chunks (%d with entities, %d text-only), %d-dim",
            len(updates), with_entities, len(updates) - with_entities, hybrid_dim,
        )
        return HybridEmbeddingResult(
            chunk_count=len(updates), embedding_dim=hybrid_dim, skipped=False,
        )

    except Exception as exc:
        logger.warning("Hybrid embeddings failed: %s", exc)
        return HybridEmbeddingResult(
            chunk_count=0, embedding_dim=hybrid_dim, skipped=True, error=str(exc),
        )
