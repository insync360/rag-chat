"""TransE relation embeddings — translational structure where h + r ≈ t.

Reads (head, relation, tail) triples from Neo4j, trains TransE with
margin-based ranking loss, and stores:
  - Per-relation embeddings in `relation_embeddings` table
  - Per-entity TransE embeddings in `entity_embeddings.transe_embedding`

Runs post-batch after GraphSAGE — same non-blocking pattern.
"""

import asyncio
import logging
from pathlib import Path

from app.config import settings
from app.database import get_pool
from app.graph.embeddings import _check_torch
from app.graph.models import TransEResult
from app.graph.neo4j_client import get_driver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TransE model (pure PyTorch)
# ---------------------------------------------------------------------------

def _build_transe_class():
    """Import torch and define TransE model. Returns (torch, TransE)."""
    import torch
    import torch.nn as nn

    class TransE(nn.Module):
        def __init__(self, num_entities: int, num_relations: int, dim: int):
            super().__init__()
            self.ent_emb = nn.Embedding(num_entities, dim)
            self.rel_emb = nn.Embedding(num_relations, dim)
            nn.init.xavier_uniform_(self.ent_emb.weight)
            nn.init.xavier_uniform_(self.rel_emb.weight)
            self.dim = dim

        def entity_embeddings(self) -> "torch.Tensor":
            return torch.nn.functional.normalize(self.ent_emb.weight.detach(), p=2, dim=1)

        def relation_embeddings(self) -> "torch.Tensor":
            return torch.nn.functional.normalize(self.rel_emb.weight.detach(), p=2, dim=1)

    return torch, TransE


# ---------------------------------------------------------------------------
# Neo4j reads
# ---------------------------------------------------------------------------

async def _read_triples_for_transe() -> list[dict]:
    """Read (head_name, head_type, rel_type, tail_name, tail_type) triples."""
    driver = await get_driver()
    async with driver.session() as session:
        result = await session.run(
            "MATCH (h:Entity)-[r]->(t:Entity) "
            "WHERE h.status = 'active' AND t.status = 'active' "
            "RETURN h.name AS head_name, h.type AS head_type, "
            "type(r) AS rel_type, "
            "t.name AS tail_name, t.type AS tail_type"
        )
        return [dict(r) async for r in result]


# ---------------------------------------------------------------------------
# Triple data preparation
# ---------------------------------------------------------------------------

def _build_triple_data(
    raw_triples: list[dict],
) -> tuple[dict[str, int], dict[str, int], list[tuple[int, int, int]]]:
    """Map entities/relations to integer indices, build (h, r, t) tuples.

    Returns (ent_to_idx, rel_to_idx, triples).
    Filters self-loops and deduplicates.
    """
    entities: set[str] = set()
    relations: set[str] = set()

    for tri in raw_triples:
        h_key = f"{tri['head_name']}::{tri['head_type']}"
        t_key = f"{tri['tail_name']}::{tri['tail_type']}"
        entities.add(h_key)
        entities.add(t_key)
        relations.add(tri["rel_type"])

    ent_to_idx = {e: i for i, e in enumerate(sorted(entities))}
    rel_to_idx = {r: i for i, r in enumerate(sorted(relations))}

    seen: set[tuple[int, int, int]] = set()
    triples: list[tuple[int, int, int]] = []

    for tri in raw_triples:
        h_key = f"{tri['head_name']}::{tri['head_type']}"
        t_key = f"{tri['tail_name']}::{tri['tail_type']}"
        h = ent_to_idx[h_key]
        r = rel_to_idx[tri["rel_type"]]
        t = ent_to_idx[t_key]
        if h == t:
            continue  # self-loop
        triple = (h, r, t)
        if triple not in seen:
            seen.add(triple)
            triples.append(triple)

    return ent_to_idx, rel_to_idx, triples


# ---------------------------------------------------------------------------
# Training (CPU-bound, runs in executor)
# ---------------------------------------------------------------------------

def _train_transe(
    num_entities: int,
    num_relations: int,
    triples: list[tuple[int, int, int]],
) -> "TransE":
    """Train TransE with margin-based ranking loss. Returns trained model."""
    torch, TransE = _build_transe_class()

    torch.manual_seed(settings.TRANSE_SEED)

    model = TransE(num_entities, num_relations, settings.TRANSE_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.TRANSE_LR)

    triple_tensor = torch.tensor(triples, dtype=torch.long)
    num_triples = len(triples)

    for epoch in range(settings.TRANSE_EPOCHS):
        model.train()

        # Sample batch
        if num_triples > settings.TRANSE_BATCH_SIZE:
            perm = torch.randperm(num_triples)[:settings.TRANSE_BATCH_SIZE]
            batch = triple_tensor[perm]
        else:
            batch = triple_tensor

        h_idx = batch[:, 0]
        r_idx = batch[:, 1]
        t_idx = batch[:, 2]

        h_emb = model.ent_emb(h_idx)
        r_emb = model.rel_emb(r_idx)
        t_emb = model.ent_emb(t_idx)

        # Positive distance: ||h + r - t||
        pos_dist = torch.norm(h_emb + r_emb - t_emb, p=2, dim=1)

        # Negative sampling: 50/50 corrupt head or tail
        batch_size = batch.size(0)
        corrupt_head = torch.rand(batch_size) < 0.5
        neg_entities = torch.randint(0, num_entities, (batch_size,))

        neg_h_idx = torch.where(corrupt_head, neg_entities, h_idx)
        neg_t_idx = torch.where(~corrupt_head, neg_entities, t_idx)

        neg_h_emb = model.ent_emb(neg_h_idx)
        neg_t_emb = model.ent_emb(neg_t_idx)

        # Negative distance: ||h' + r - t'||
        neg_dist = torch.norm(neg_h_emb + r_emb - neg_t_emb, p=2, dim=1)

        # Margin-based ranking loss: max(0, pos_dist - neg_dist + margin)
        loss = torch.relu(pos_dist - neg_dist + settings.TRANSE_MARGIN).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # L2 normalize entity embeddings after each step (standard TransE constraint)
        with torch.no_grad():
            model.ent_emb.weight.data = torch.nn.functional.normalize(
                model.ent_emb.weight.data, p=2, dim=1,
            )

        if (epoch + 1) % 50 == 0:
            logger.info(
                "TransE epoch %d/%d, loss=%.4f",
                epoch + 1, settings.TRANSE_EPOCHS, loss.item(),
            )

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def _transe_model_path() -> Path:
    return Path(settings.TRANSE_MODEL_DIR) / "transe_weights.pt"


def _save_transe(model, ent_to_idx: dict[str, int], rel_to_idx: dict[str, int]) -> None:
    torch = __import__("torch")
    path = _transe_model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "dim": model.dim,
        "ent_to_idx": ent_to_idx,
        "rel_to_idx": rel_to_idx,
    }, str(path))
    logger.info("TransE model saved to %s", path)


def _load_transe():
    """Load TransE model from disk. Returns (model, ent_to_idx, rel_to_idx) or None."""
    torch, TransE = _build_transe_class()
    path = _transe_model_path()
    if not path.exists():
        return None
    checkpoint = torch.load(str(path), weights_only=False)
    ent_to_idx = checkpoint["ent_to_idx"]
    rel_to_idx = checkpoint["rel_to_idx"]
    model = TransE(len(ent_to_idx), len(rel_to_idx), checkpoint["dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    logger.info("TransE model loaded from %s", path)
    return model, ent_to_idx, rel_to_idx


# ---------------------------------------------------------------------------
# Neon storage
# ---------------------------------------------------------------------------

async def _store_relation_embeddings(
    rel_to_idx: dict[str, int], model,
) -> None:
    """Upsert relation embeddings to relation_embeddings table."""
    pool = await get_pool()
    rel_embs = model.relation_embeddings()

    args = []
    for rel_type, idx in rel_to_idx.items():
        emb = rel_embs[idx].tolist()
        args.append((rel_type, str(emb)))

    await pool.executemany(
        "INSERT INTO relation_embeddings (relation_type, embedding, updated_at) "
        "VALUES ($1, $2::vector, now()) "
        "ON CONFLICT (relation_type) DO UPDATE SET "
        "embedding = EXCLUDED.embedding, "
        "updated_at = now()",
        args,
    )
    logger.info("Stored %d relation embeddings in Neon", len(args))


async def _store_transe_entity_embeddings(
    ent_to_idx: dict[str, int], model,
) -> None:
    """UPDATE existing entity_embeddings rows with transe_embedding."""
    pool = await get_pool()
    ent_embs = model.entity_embeddings()
    batch_size = 100

    # ent_to_idx keys are "name::type"
    args = []
    for ent_key, idx in ent_to_idx.items():
        name, ent_type = ent_key.rsplit("::", 1)
        emb = ent_embs[idx].tolist()
        args.append((str(emb), name, ent_type))

    for start in range(0, len(args), batch_size):
        batch = args[start:start + batch_size]
        await pool.executemany(
            "UPDATE entity_embeddings SET transe_embedding = $1::vector, "
            "updated_at = now() "
            "WHERE entity_name = $2 AND entity_type = $3",
            batch,
        )

    logger.info("Stored %d TransE entity embeddings in Neon", len(args))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def generate_transe_embeddings(
    *, force_retrain: bool = False,
) -> TransEResult:
    """Generate TransE embeddings for all active entities and relations.

    Never raises — returns TransEResult with error details on failure.
    """
    if not settings.TRANSE_ENABLED:
        return TransEResult(
            entity_count=0, relation_count=0, embedding_dim=0, skipped=True,
        )

    if not _check_torch():
        return TransEResult(
            entity_count=0, relation_count=0, embedding_dim=0, skipped=True,
            error="torch not installed",
        )

    try:
        # 1. Read triples from Neo4j
        raw_triples = await _read_triples_for_transe()

        if not raw_triples:
            return TransEResult(
                entity_count=0, relation_count=0,
                embedding_dim=settings.TRANSE_DIM,
                skipped=False, error="no valid triples",
            )

        # 2. Build triple data
        loop = asyncio.get_event_loop()
        ent_to_idx, rel_to_idx, triples = await loop.run_in_executor(
            None, _build_triple_data, raw_triples,
        )

        if not triples:
            return TransEResult(
                entity_count=len(ent_to_idx), relation_count=len(rel_to_idx),
                embedding_dim=settings.TRANSE_DIM,
                skipped=False, error="no valid triples",
            )

        # 3. Train or load model
        retrained = False
        if force_retrain or _load_transe() is None:
            model = await loop.run_in_executor(
                None, _train_transe, len(ent_to_idx), len(rel_to_idx), triples,
            )
            await loop.run_in_executor(
                None, _save_transe, model, ent_to_idx, rel_to_idx,
            )
            retrained = True
        else:
            model, ent_to_idx, rel_to_idx = _load_transe()

        # 4. Store in Neon
        await _store_relation_embeddings(rel_to_idx, model)
        await _store_transe_entity_embeddings(ent_to_idx, model)

        logger.info(
            "TransE embeddings complete: %d entities, %d relations, %d-dim",
            len(ent_to_idx), len(rel_to_idx), settings.TRANSE_DIM,
        )
        return TransEResult(
            entity_count=len(ent_to_idx),
            relation_count=len(rel_to_idx),
            embedding_dim=settings.TRANSE_DIM,
            retrained=retrained,
        )

    except Exception as exc:
        logger.warning("TransE embeddings failed: %s", exc)
        return TransEResult(
            entity_count=0, relation_count=0, embedding_dim=0,
            skipped=True, error=str(exc),
        )
