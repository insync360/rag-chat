"""Tests for TransE relation embeddings."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _build_triple_data
# ---------------------------------------------------------------------------

class TestBuildTripleData:
    def test_basic_triples(self):
        from app.graph.transe import _build_triple_data

        raw = [
            {"head_name": "A", "head_type": "T", "rel_type": "R1",
             "tail_name": "B", "tail_type": "T"},
            {"head_name": "B", "head_type": "T", "rel_type": "R2",
             "tail_name": "C", "tail_type": "T"},
        ]

        ent_to_idx, rel_to_idx, triples = _build_triple_data(raw)

        assert len(ent_to_idx) == 3
        assert len(rel_to_idx) == 2
        assert len(triples) == 2

    def test_self_loops_filtered(self):
        from app.graph.transe import _build_triple_data

        raw = [
            {"head_name": "A", "head_type": "T", "rel_type": "R1",
             "tail_name": "A", "tail_type": "T"},
        ]

        _, _, triples = _build_triple_data(raw)

        assert triples == []

    def test_unknown_ids_skipped(self):
        """All entities come from raw triples, so no unknown IDs possible.
        Instead, test that entity keys include type disambiguation."""
        from app.graph.transe import _build_triple_data

        raw = [
            {"head_name": "A", "head_type": "Person", "rel_type": "R1",
             "tail_name": "A", "tail_type": "Org"},
        ]

        ent_to_idx, _, triples = _build_triple_data(raw)

        # Same name, different type → 2 entities
        assert len(ent_to_idx) == 2
        assert "A::Person" in ent_to_idx
        assert "A::Org" in ent_to_idx
        assert len(triples) == 1

    def test_relation_types_sorted(self):
        from app.graph.transe import _build_triple_data

        raw = [
            {"head_name": "A", "head_type": "T", "rel_type": "ZEBRA",
             "tail_name": "B", "tail_type": "T"},
            {"head_name": "C", "head_type": "T", "rel_type": "APPLE",
             "tail_name": "D", "tail_type": "T"},
        ]

        _, rel_to_idx, _ = _build_triple_data(raw)

        assert rel_to_idx["APPLE"] < rel_to_idx["ZEBRA"]

    def test_empty_triples(self):
        from app.graph.transe import _build_triple_data

        ent_to_idx, rel_to_idx, triples = _build_triple_data([])

        assert ent_to_idx == {}
        assert rel_to_idx == {}
        assert triples == []

    def test_duplicate_triples_deduplicated(self):
        from app.graph.transe import _build_triple_data

        raw = [
            {"head_name": "A", "head_type": "T", "rel_type": "R1",
             "tail_name": "B", "tail_type": "T"},
            {"head_name": "A", "head_type": "T", "rel_type": "R1",
             "tail_name": "B", "tail_type": "T"},
        ]

        _, _, triples = _build_triple_data(raw)

        assert len(triples) == 1


# ---------------------------------------------------------------------------
# TransE model
# ---------------------------------------------------------------------------

class TestTransEModel:
    def test_output_shape(self):
        _, TransE = __import__(
            "app.graph.transe", fromlist=["_build_transe_class"]
        )._build_transe_class()

        model = TransE(num_entities=10, num_relations=3, dim=16)

        assert model.entity_embeddings().shape == (10, 16)
        assert model.relation_embeddings().shape == (3, 16)

    def test_entity_embeddings_l2_normalized(self):
        import torch
        _, TransE = __import__(
            "app.graph.transe", fromlist=["_build_transe_class"]
        )._build_transe_class()

        model = TransE(num_entities=5, num_relations=2, dim=8)
        embs = model.entity_embeddings()
        norms = torch.norm(embs, p=2, dim=1)

        assert torch.allclose(norms, torch.ones(5), atol=1e-5)

    def test_relation_embeddings_l2_normalized(self):
        import torch
        _, TransE = __import__(
            "app.graph.transe", fromlist=["_build_transe_class"]
        )._build_transe_class()

        model = TransE(num_entities=5, num_relations=3, dim=8)
        embs = model.relation_embeddings()
        norms = torch.norm(embs, p=2, dim=1)

        assert torch.allclose(norms, torch.ones(3), atol=1e-5)

    def test_translation_property(self):
        """After training, h + r should be approximately t for known triples."""
        import torch
        from app.graph.transe import _train_transe

        # Simple graph: 0 --R0--> 1
        triples = [(0, 0, 1)]

        with patch("app.graph.transe.settings") as s:
            s.TRANSE_DIM = 16
            s.TRANSE_EPOCHS = 300
            s.TRANSE_LR = 0.01
            s.TRANSE_MARGIN = 1.0
            s.TRANSE_BATCH_SIZE = 512
            s.TRANSE_SEED = 42

            model = _train_transe(
                num_entities=2, num_relations=1, triples=triples,
            )

        ent_embs = model.entity_embeddings()
        rel_embs = model.relation_embeddings()

        h = ent_embs[0]
        r = rel_embs[0]
        t = ent_embs[1]

        dist = torch.norm(h + r - t, p=2).item()
        # After training, distance should be small (< margin)
        assert dist < 1.5, f"Translation distance {dist} too large"


# ---------------------------------------------------------------------------
# generate_transe_embeddings — disabled / missing torch / empty
# ---------------------------------------------------------------------------

class TestGenerateTranseEmbeddings:
    @pytest.mark.asyncio
    async def test_disabled_returns_skipped(self):
        with patch("app.graph.transe.settings") as mock_settings:
            mock_settings.TRANSE_ENABLED = False
            from app.graph.transe import generate_transe_embeddings

            result = await generate_transe_embeddings()

            assert result.skipped is True

    @pytest.mark.asyncio
    async def test_torch_not_installed_returns_skipped(self):
        with patch("app.graph.transe.settings") as mock_settings, \
             patch("app.graph.transe._check_torch", return_value=False):
            mock_settings.TRANSE_ENABLED = True
            from app.graph.transe import generate_transe_embeddings

            result = await generate_transe_embeddings()

            assert result.skipped is True
            assert result.error == "torch not installed"

    @pytest.mark.asyncio
    async def test_empty_graph_returns_no_triples(self):
        with patch("app.graph.transe._check_torch", return_value=True), \
             patch("app.graph.transe._read_triples_for_transe", new_callable=AsyncMock) as mock_read:
            mock_read.return_value = []
            from app.graph.transe import generate_transe_embeddings

            result = await generate_transe_embeddings()

            assert result.skipped is False
            assert result.error == "no valid triples"
            assert result.entity_count == 0

    @pytest.mark.asyncio
    async def test_no_triples_after_filtering(self):
        """All triples are self-loops → no valid triples after filtering."""
        with patch("app.graph.transe._check_torch", return_value=True), \
             patch("app.graph.transe._read_triples_for_transe", new_callable=AsyncMock) as mock_read:
            mock_read.return_value = [
                {"head_name": "A", "head_type": "T", "rel_type": "R1",
                 "tail_name": "A", "tail_type": "T"},
            ]
            from app.graph.transe import generate_transe_embeddings

            result = await generate_transe_embeddings()

            assert result.error == "no valid triples"


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    @pytest.mark.asyncio
    async def test_pipeline_calls_transe_after_graphsage(self):
        """Verify ingest_files calls generate_transe_embeddings after GraphSAGE."""
        with patch("app.ingestion.pipeline.get_pool", new_callable=AsyncMock), \
             patch("app.ingestion.pipeline.LlamaParser"), \
             patch("app.ingestion.pipeline.VersionTracker"), \
             patch("app.ingestion.pipeline._ingest_one", new_callable=AsyncMock) as mock_ingest, \
             patch("app.graph.community.detect_communities", new_callable=AsyncMock) as mock_comm, \
             patch("app.graph.embeddings.generate_graph_embeddings", new_callable=AsyncMock) as mock_emb, \
             patch("app.graph.transe.generate_transe_embeddings", new_callable=AsyncMock) as mock_transe:
            from app.ingestion.pipeline import PipelineStatus, ingest_files

            mock_ingest.return_value = MagicMock(status=PipelineStatus.COMPLETED)
            mock_comm.return_value = MagicMock()
            mock_emb.return_value = MagicMock()
            mock_transe.return_value = MagicMock()

            await ingest_files(["dummy.pdf"])

            mock_transe.assert_called_once_with(force_retrain=True)

    @pytest.mark.asyncio
    async def test_pipeline_continues_if_transe_fails(self):
        """Pipeline should not fail if TransE raises an exception."""
        with patch("app.ingestion.pipeline.get_pool", new_callable=AsyncMock), \
             patch("app.ingestion.pipeline.LlamaParser"), \
             patch("app.ingestion.pipeline.VersionTracker"), \
             patch("app.ingestion.pipeline._ingest_one", new_callable=AsyncMock) as mock_ingest, \
             patch("app.graph.community.detect_communities", new_callable=AsyncMock) as mock_comm, \
             patch("app.graph.embeddings.generate_graph_embeddings", new_callable=AsyncMock) as mock_emb, \
             patch("app.graph.transe.generate_transe_embeddings", new_callable=AsyncMock) as mock_transe:
            from app.ingestion.pipeline import PipelineStatus, ingest_files

            mock_ingest.return_value = MagicMock(status=PipelineStatus.COMPLETED)
            mock_comm.return_value = MagicMock()
            mock_emb.return_value = MagicMock()
            mock_transe.side_effect = RuntimeError("TransE exploded")

            result = await ingest_files(["dummy.pdf"])

            # Pipeline completed despite TransE failure
            assert result.total == 1
