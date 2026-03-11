"""GraphSAGE structural entity embeddings — pure PyTorch, no torch-geometric.

Reads entities + relationships from Neo4j, trains a 2-layer GraphSAGE model
via unsupervised link prediction, and stores 128-dim structural embeddings
(+ 256-dim OpenAI text embeddings) in Neon pgvector.

Runs post-batch after community detection — same non-blocking pattern.
"""

import asyncio
import logging
import os
from collections import defaultdict
from pathlib import Path

from openai import AsyncOpenAI

from app.config import settings
from app.database import get_pool
from app.graph.models import GraphEmbeddingResult
from app.graph.neo4j_client import get_driver

logger = logging.getLogger(__name__)

_torch_available: bool | None = None


def _check_torch() -> bool:
    """Lazy-check whether torch is importable."""
    global _torch_available
    if _torch_available is not None:
        return _torch_available
    try:
        import torch as _t  # noqa: F401
        _torch_available = True
    except ImportError:
        _torch_available = False
        logger.warning("torch not installed — graph embeddings disabled")
    return _torch_available


# ---------------------------------------------------------------------------
# GraphSAGE model (pure PyTorch)
# ---------------------------------------------------------------------------

def _build_model_classes():
    """Import torch and define model classes. Returns (torch, GraphSAGELayer, GraphSAGE)."""
    import torch
    import torch.nn as nn

    class GraphSAGELayer(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.W_self = nn.Linear(in_dim, out_dim, bias=False)
            self.W_neigh = nn.Linear(in_dim, out_dim, bias=False)

        def forward(self, x: torch.Tensor, adj: list[list[int]], k: int, seed: int) -> torch.Tensor:
            """Mean aggregation with neighbor sampling."""
            n = x.size(0)
            gen = torch.Generator().manual_seed(seed)
            neigh_agg = torch.zeros(n, x.size(1), device=x.device)

            for i in range(n):
                neighbors = adj[i]
                if not neighbors:
                    # Isolated node: self-loop
                    neigh_agg[i] = x[i]
                else:
                    if len(neighbors) > k:
                        indices = torch.randint(len(neighbors), (k,), generator=gen)
                        sampled = [neighbors[idx] for idx in indices.tolist()]
                    else:
                        sampled = neighbors
                    neigh_agg[i] = x[sampled].mean(dim=0)

            out = self.W_self(x) + self.W_neigh(neigh_agg)
            return torch.relu(out)

    class GraphSAGE(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
            super().__init__()
            self.layer1 = GraphSAGELayer(in_dim, hidden_dim)
            self.layer2 = GraphSAGELayer(hidden_dim, out_dim)
            self.in_dim = in_dim
            self.hidden_dim = hidden_dim
            self.out_dim = out_dim

        def forward(self, x: torch.Tensor, adj: list[list[int]], k: int, seed: int) -> torch.Tensor:
            h = self.layer1(x, adj, k, seed)
            h = self.layer2(h, adj, k, seed + 1)
            # L2 normalize for cosine similarity
            return torch.nn.functional.normalize(h, p=2, dim=1)

    return torch, GraphSAGELayer, GraphSAGE


# ---------------------------------------------------------------------------
# Neo4j reads
# ---------------------------------------------------------------------------

async def _read_graph_for_embeddings() -> tuple[list[dict], list[dict]]:
    """Read active entities and relationships from Neo4j."""
    driver = await get_driver()
    async with driver.session() as session:
        ent_result = await session.run(
            "MATCH (e:Entity) WHERE e.status = 'active' "
            "RETURN elementId(e) AS neo4j_id, e.name AS name, e.type AS type"
        )
        entities = [dict(r) async for r in ent_result]

        rel_result = await session.run(
            "MATCH (s:Entity)-[r]->(t:Entity) "
            "WHERE s.status = 'active' AND t.status = 'active' "
            "RETURN elementId(s) AS source_id, elementId(t) AS target_id"
        )
        relationships = [dict(r) async for r in rel_result]

    return entities, relationships


# ---------------------------------------------------------------------------
# Adjacency + features
# ---------------------------------------------------------------------------

def _build_adjacency(
    entities: list[dict], relationships: list[dict],
) -> tuple[dict[str, int], list[list[int]], list[tuple[int, int]]]:
    """Build undirected adjacency lists. Returns (id_to_idx, adj, edges)."""
    id_to_idx: dict[str, int] = {}
    for i, ent in enumerate(entities):
        id_to_idx[ent["neo4j_id"]] = i

    adj: list[list[int]] = [[] for _ in range(len(entities))]
    edges: list[tuple[int, int]] = []

    for rel in relationships:
        src = id_to_idx.get(rel["source_id"])
        tgt = id_to_idx.get(rel["target_id"])
        if src is None or tgt is None:
            continue
        if src == tgt:
            continue  # filter self-loops
        # Undirected
        adj[src].append(tgt)
        adj[tgt].append(src)
        edges.append((src, tgt))

    return id_to_idx, adj, edges


async def _get_node_features(entities: list[dict]) -> list[list[float]]:
    """Get OpenAI embeddings for entity names as initial node features."""
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    texts = [f"{e['name']} ({e['type']})" for e in entities]

    all_embeddings: list[list[float]] = [[] for _ in range(len(texts))]
    batch_size = 2048

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        resp = await client.embeddings.create(
            model=settings.ENTITY_EMBEDDING_MODEL,
            input=batch,
            dimensions=settings.GRAPHSAGE_INPUT_DIM,
        )
        for i, item in enumerate(resp.data):
            all_embeddings[start + i] = item.embedding

    return all_embeddings


# ---------------------------------------------------------------------------
# Training (CPU-bound, runs in executor)
# ---------------------------------------------------------------------------

def _train_graphsage(
    features_list: list[list[float]],
    adj: list[list[int]],
    edges: list[tuple[int, int]],
) -> "GraphSAGE":
    """Train GraphSAGE via unsupervised link prediction. Returns trained model."""
    torch, _, GraphSAGE = _build_model_classes()

    torch.manual_seed(settings.GRAPHSAGE_SEED)

    n = len(features_list)
    x = torch.tensor(features_list, dtype=torch.float32)

    model = GraphSAGE(
        settings.GRAPHSAGE_INPUT_DIM,
        settings.GRAPHSAGE_HIDDEN_DIM,
        settings.GRAPHSAGE_OUTPUT_DIM,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.GRAPHSAGE_LR)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    if not edges:
        # No edges — just run forward pass, no training needed
        logger.info("No edges in graph, skipping GraphSAGE training")
        model.eval()
        return model

    edge_tensor = torch.tensor(edges, dtype=torch.long)
    num_edges = len(edges)

    for epoch in range(settings.GRAPHSAGE_EPOCHS):
        model.train()

        # Sample edge batch if needed
        if num_edges > settings.GRAPHSAGE_BATCH_SIZE:
            perm = torch.randperm(num_edges)[:settings.GRAPHSAGE_BATCH_SIZE]
            batch_edges = edge_tensor[perm]
        else:
            batch_edges = edge_tensor

        batch_size = batch_edges.size(0)

        # Forward pass
        embs = model(x, adj, settings.GRAPHSAGE_NEIGHBOR_SAMPLES, settings.GRAPHSAGE_SEED)

        # Positive scores
        src_embs = embs[batch_edges[:, 0]]
        tgt_embs = embs[batch_edges[:, 1]]
        pos_scores = (src_embs * tgt_embs).sum(dim=1)

        # Negative sampling: vectorized
        neg_count = batch_size * settings.GRAPHSAGE_NEG_RATIO
        neg_src = batch_edges[:, 0].repeat(settings.GRAPHSAGE_NEG_RATIO)
        neg_tgt = torch.randint(0, n, (neg_count,))
        neg_src_embs = embs[neg_src]
        neg_tgt_embs = embs[neg_tgt]
        neg_scores = (neg_src_embs * neg_tgt_embs).sum(dim=1)

        # Loss
        pos_labels = torch.ones(batch_size)
        neg_labels = torch.zeros(neg_count)
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([pos_labels, neg_labels])
        loss = loss_fn(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            logger.info("GraphSAGE epoch %d/%d, loss=%.4f", epoch + 1, settings.GRAPHSAGE_EPOCHS, loss.item())

    model.eval()
    return model


def _infer_embeddings(
    model, features_list: list[list[float]], adj: list[list[int]],
) -> list[list[float]]:
    """Forward pass → list of embedding vectors."""
    torch = __import__("torch")
    x = torch.tensor(features_list, dtype=torch.float32)
    with torch.no_grad():
        embs = model(x, adj, settings.GRAPHSAGE_NEIGHBOR_SAMPLES, settings.GRAPHSAGE_SEED)
    return embs.tolist()


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def _model_path() -> Path:
    return Path(settings.GRAPHSAGE_MODEL_DIR) / "graphsage_weights.pt"


def _save_model(model) -> None:
    """Save model weights + dim config to disk."""
    torch = __import__("torch")
    path = _model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": model.in_dim,
        "hidden_dim": model.hidden_dim,
        "out_dim": model.out_dim,
    }, str(path))
    logger.info("GraphSAGE model saved to %s", path)


def _load_model():
    """Load model from disk. Returns model or None if not found."""
    torch, _, GraphSAGE = _build_model_classes()
    path = _model_path()
    if not path.exists():
        return None
    checkpoint = torch.load(str(path), weights_only=True)
    model = GraphSAGE(
        checkpoint["in_dim"],
        checkpoint["hidden_dim"],
        checkpoint["out_dim"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    logger.info("GraphSAGE model loaded from %s", path)
    return model


# ---------------------------------------------------------------------------
# Neon storage
# ---------------------------------------------------------------------------

async def _store_embeddings(
    entities: list[dict],
    structural_embs: list[list[float]],
    text_embs: list[list[float]],
) -> None:
    """Batch upsert structural + text embeddings to Neon entity_embeddings table."""
    pool = await get_pool()
    batch_size = 100

    for start in range(0, len(entities), batch_size):
        end = min(start + batch_size, len(entities))
        batch_entities = entities[start:end]
        batch_structural = structural_embs[start:end]
        batch_text = text_embs[start:end]

        # Build args for executemany
        args = []
        for ent, s_emb, t_emb in zip(batch_entities, batch_structural, batch_text):
            args.append((
                ent["name"],
                ent["type"],
                ent["neo4j_id"],
                str(s_emb),
                str(t_emb),
            ))

        await pool.executemany(
            "INSERT INTO entity_embeddings (entity_name, entity_type, neo4j_id, "
            "embedding, text_embedding, updated_at) "
            "VALUES ($1, $2, $3, $4::vector, $5::vector, now()) "
            "ON CONFLICT (entity_name, entity_type) DO UPDATE SET "
            "neo4j_id = EXCLUDED.neo4j_id, "
            "embedding = EXCLUDED.embedding, "
            "text_embedding = EXCLUDED.text_embedding, "
            "model_version = EXCLUDED.model_version, "
            "updated_at = now()",
            args,
        )

    logger.info("Stored %d entity embeddings in Neon", len(entities))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def generate_graph_embeddings(
    *, force_retrain: bool = False,
) -> GraphEmbeddingResult:
    """Generate GraphSAGE structural embeddings for all active entities.

    Never raises — returns GraphEmbeddingResult(skipped=True) on any failure.
    """
    if not settings.GRAPH_EMBEDDINGS_ENABLED:
        return GraphEmbeddingResult(
            entity_count=0, embedding_dim=0, skipped=True, error=None,
        )

    if not _check_torch():
        return GraphEmbeddingResult(
            entity_count=0, embedding_dim=0, skipped=True,
            error="torch not installed",
        )

    try:
        # 1. Read graph from Neo4j
        entities, relationships = await _read_graph_for_embeddings()

        if not entities:
            return GraphEmbeddingResult(
                entity_count=0, embedding_dim=settings.GRAPHSAGE_OUTPUT_DIM,
                skipped=False,
            )

        # 2. Build adjacency
        loop = asyncio.get_event_loop()
        id_to_idx, adj, edges = await loop.run_in_executor(
            None, _build_adjacency, entities, relationships,
        )

        # 3. Get OpenAI text embeddings as initial features
        text_embs = await _get_node_features(entities)

        # 4. Train or load model
        retrained = False
        if force_retrain or _load_model() is None:
            model = await loop.run_in_executor(
                None, _train_graphsage, text_embs, adj, edges,
            )
            await loop.run_in_executor(None, _save_model, model)
            retrained = True
        else:
            model = _load_model()

        # 5. Inference
        structural_embs = await loop.run_in_executor(
            None, _infer_embeddings, model, text_embs, adj,
        )

        # 6. Store in Neon
        await _store_embeddings(entities, structural_embs, text_embs)

        logger.info(
            "Graph embeddings complete: %d entities, %d-dim structural embeddings",
            len(entities), settings.GRAPHSAGE_OUTPUT_DIM,
        )
        return GraphEmbeddingResult(
            entity_count=len(entities),
            embedding_dim=settings.GRAPHSAGE_OUTPUT_DIM,
            skipped=False,
            retrained=retrained,
        )

    except Exception as exc:
        logger.warning("Graph embeddings failed: %s", exc)
        return GraphEmbeddingResult(
            entity_count=0, embedding_dim=0, skipped=True, error=str(exc),
        )
