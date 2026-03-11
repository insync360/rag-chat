"""Tests for GraphSAGE structural entity embeddings."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _build_adjacency
# ---------------------------------------------------------------------------

class TestBuildAdjacency:
    def test_undirected_adjacency(self):
        from app.graph.embeddings import _build_adjacency

        entities = [
            {"neo4j_id": "a", "name": "A", "type": "T"},
            {"neo4j_id": "b", "name": "B", "type": "T"},
            {"neo4j_id": "c", "name": "C", "type": "T"},
        ]
        rels = [
            {"source_id": "a", "target_id": "b"},
            {"source_id": "b", "target_id": "c"},
        ]

        id_to_idx, adj, edges = _build_adjacency(entities, rels)

        assert id_to_idx == {"a": 0, "b": 1, "c": 2}
        # Undirected: a-b means adj[0] has 1 and adj[1] has 0
        assert 1 in adj[0]
        assert 0 in adj[1]
        assert 2 in adj[1]
        assert 1 in adj[2]
        assert len(edges) == 2

    def test_self_loops_filtered(self):
        from app.graph.embeddings import _build_adjacency

        entities = [{"neo4j_id": "a", "name": "A", "type": "T"}]
        rels = [{"source_id": "a", "target_id": "a"}]

        _, adj, edges = _build_adjacency(entities, rels)

        assert adj[0] == []
        assert edges == []

    def test_unknown_ids_skipped(self):
        from app.graph.embeddings import _build_adjacency

        entities = [{"neo4j_id": "a", "name": "A", "type": "T"}]
        rels = [{"source_id": "a", "target_id": "unknown"}]

        _, adj, edges = _build_adjacency(entities, rels)

        assert adj[0] == []
        assert edges == []


# ---------------------------------------------------------------------------
# GraphSAGE model
# ---------------------------------------------------------------------------

class TestGraphSAGEModel:
    def test_output_shape(self):
        _, _, GraphSAGE = __import__(
            "app.graph.embeddings", fromlist=["_build_model_classes"]
        )._build_model_classes()
        import torch

        model = GraphSAGE(in_dim=16, hidden_dim=8, out_dim=8)
        x = torch.randn(5, 16)
        adj = [[1, 2], [0], [0], [4], [3]]

        out = model(x, adj, k=10, seed=42)

        assert out.shape == (5, 8)

    def test_l2_normalized(self):
        _, _, GraphSAGE = __import__(
            "app.graph.embeddings", fromlist=["_build_model_classes"]
        )._build_model_classes()
        import torch

        model = GraphSAGE(in_dim=16, hidden_dim=8, out_dim=8)
        x = torch.randn(3, 16)
        adj = [[1], [0, 2], [1]]

        out = model(x, adj, k=10, seed=42)
        norms = torch.norm(out, p=2, dim=1)

        assert torch.allclose(norms, torch.ones(3), atol=1e-5)

    def test_isolated_nodes(self):
        """Isolated nodes should get valid embeddings via self-loop."""
        _, _, GraphSAGE = __import__(
            "app.graph.embeddings", fromlist=["_build_model_classes"]
        )._build_model_classes()
        import torch

        model = GraphSAGE(in_dim=16, hidden_dim=8, out_dim=8)
        x = torch.randn(3, 16)
        adj = [[], [], []]  # all isolated

        out = model(x, adj, k=10, seed=42)

        assert out.shape == (3, 8)
        norms = torch.norm(out, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5)


# ---------------------------------------------------------------------------
# generate_graph_embeddings — disabled / missing torch
# ---------------------------------------------------------------------------

class TestGenerateGraphEmbeddings:
    @pytest.mark.asyncio
    async def test_disabled_returns_skipped(self):
        with patch("app.graph.embeddings.settings") as mock_settings:
            mock_settings.GRAPH_EMBEDDINGS_ENABLED = False
            from app.graph.embeddings import generate_graph_embeddings

            result = await generate_graph_embeddings()

            assert result.skipped is True

    @pytest.mark.asyncio
    async def test_torch_not_installed_returns_skipped(self):
        import app.graph.embeddings as mod

        # Reset cached check
        original = mod._torch_available
        mod._torch_available = None

        with patch("app.graph.embeddings.settings") as mock_settings:
            mock_settings.GRAPH_EMBEDDINGS_ENABLED = True
            with patch.dict("sys.modules", {"torch": None}):
                # Force re-check
                mod._torch_available = False
                result = await mod.generate_graph_embeddings()

        mod._torch_available = original
        assert result.skipped is True
        assert result.error == "torch not installed"

    @pytest.mark.asyncio
    async def test_empty_graph_returns_zero(self):
        with patch("app.graph.embeddings._check_torch", return_value=True), \
             patch("app.graph.embeddings._read_graph_for_embeddings", new_callable=AsyncMock) as mock_read:
            mock_read.return_value = ([], [])
            from app.graph.embeddings import generate_graph_embeddings

            result = await generate_graph_embeddings()

            assert result.skipped is False
            assert result.entity_count == 0


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    @pytest.mark.asyncio
    async def test_pipeline_calls_embeddings_after_community(self):
        """Verify ingest_files calls generate_graph_embeddings after community detection."""
        with patch("app.ingestion.pipeline.get_pool", new_callable=AsyncMock), \
             patch("app.ingestion.pipeline.LlamaParser"), \
             patch("app.ingestion.pipeline.VersionTracker"), \
             patch("app.ingestion.pipeline._ingest_one", new_callable=AsyncMock) as mock_ingest, \
             patch("app.graph.community.detect_communities", new_callable=AsyncMock) as mock_comm, \
             patch("app.graph.embeddings.generate_graph_embeddings", new_callable=AsyncMock) as mock_emb:
            from app.ingestion.pipeline import PipelineStatus, ingest_files

            mock_ingest.return_value = MagicMock(status=PipelineStatus.COMPLETED)
            mock_comm.return_value = MagicMock()
            mock_emb.return_value = MagicMock()

            await ingest_files(["dummy.pdf"])

            mock_emb.assert_called_once_with(force_retrain=True)

    @pytest.mark.asyncio
    async def test_pipeline_continues_if_embeddings_fail(self):
        """Pipeline should not fail if graph embeddings raise an exception."""
        with patch("app.ingestion.pipeline.get_pool", new_callable=AsyncMock), \
             patch("app.ingestion.pipeline.LlamaParser"), \
             patch("app.ingestion.pipeline.VersionTracker"), \
             patch("app.ingestion.pipeline._ingest_one", new_callable=AsyncMock) as mock_ingest, \
             patch("app.graph.community.detect_communities", new_callable=AsyncMock) as mock_comm, \
             patch("app.graph.embeddings.generate_graph_embeddings", new_callable=AsyncMock) as mock_emb:
            from app.ingestion.pipeline import PipelineStatus, ingest_files

            mock_ingest.return_value = MagicMock(status=PipelineStatus.COMPLETED)
            mock_comm.return_value = MagicMock()
            mock_emb.side_effect = RuntimeError("torch exploded")

            result = await ingest_files(["dummy.pdf"])

            # Pipeline completed despite embeddings failure
            assert result.total == 1
