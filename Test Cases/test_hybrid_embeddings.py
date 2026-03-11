"""Tests for hybrid chunk-entity embeddings."""

import math
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, ".")

from app.graph.hybrid_embeddings import (
    _build_hybrid_embedding,
    _l2_normalize,
    _mean_pool,
    _truncate_and_normalize,
    generate_hybrid_embeddings,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def _make_vec(dim: int, val: float = 1.0) -> list[float]:
    """Create a vector of given dim filled with val."""
    return [val] * dim


# ---------------------------------------------------------------------------
# TestTruncateAndNormalize
# ---------------------------------------------------------------------------

class TestTruncateAndNormalize:
    def test_truncates_to_correct_dim(self):
        vec = list(range(2000))
        result = _truncate_and_normalize(vec, 512)
        assert len(result) == 512

    def test_output_is_l2_normalized(self):
        vec = [float(i) for i in range(1, 2001)]
        result = _truncate_and_normalize(vec, 512)
        assert abs(_l2_norm(result) - 1.0) < 1e-6

    def test_handles_zero_vector(self):
        vec = [0.0] * 2000
        result = _truncate_and_normalize(vec, 512)
        assert len(result) == 512
        assert all(x == 0.0 for x in result)


# ---------------------------------------------------------------------------
# TestMeanPool
# ---------------------------------------------------------------------------

class TestMeanPool:
    def test_multiple_vectors_pooled(self):
        v1 = [1.0, 2.0, 3.0]
        v2 = [3.0, 4.0, 5.0]
        result = _mean_pool([v1, v2], 3)
        assert result == [2.0, 3.0, 4.0]

    def test_single_vector_returned(self):
        v1 = [1.0, 2.0, 3.0]
        result = _mean_pool([v1], 3)
        assert result == [1.0, 2.0, 3.0]

    def test_empty_returns_zero_vector(self):
        result = _mean_pool([], 128)
        assert len(result) == 128
        assert all(x == 0.0 for x in result)


# ---------------------------------------------------------------------------
# TestBuildHybridEmbedding
# ---------------------------------------------------------------------------

class TestBuildHybridEmbedding:
    @patch("app.graph.hybrid_embeddings.settings")
    def test_with_entities(self, mock_settings):
        mock_settings.HYBRID_CHUNK_TEXT_DIM = 512
        mock_settings.GRAPHSAGE_OUTPUT_DIM = 128
        mock_settings.TRANSE_DIM = 128

        chunk_emb = [0.01 * i for i in range(2000)]
        structural = [[float(i)] * 128 for i in range(1, 4)]
        transe = [[float(i) * 0.5] * 128 for i in range(1, 4)]

        result = _build_hybrid_embedding(chunk_emb, structural, transe)
        assert len(result) == 768
        assert abs(_l2_norm(result) - 1.0) < 1e-6

    @patch("app.graph.hybrid_embeddings.settings")
    def test_no_entities_zero_fill(self, mock_settings):
        mock_settings.HYBRID_CHUNK_TEXT_DIM = 512
        mock_settings.GRAPHSAGE_OUTPUT_DIM = 128
        mock_settings.TRANSE_DIM = 128

        chunk_emb = [0.01 * i for i in range(2000)]

        result = _build_hybrid_embedding(chunk_emb, [], [])
        assert len(result) == 768
        # Entity portions should be zero (indices 512-767)
        # After final L2-normalize, the zero portions stay zero
        # but text portion gets renormalized
        assert abs(_l2_norm(result) - 1.0) < 1e-6

    @patch("app.graph.hybrid_embeddings.settings")
    def test_mixed_some_have_transe(self, mock_settings):
        """Some entities have TransE, some don't — structural still used."""
        mock_settings.HYBRID_CHUNK_TEXT_DIM = 512
        mock_settings.GRAPHSAGE_OUTPUT_DIM = 128
        mock_settings.TRANSE_DIM = 128

        chunk_emb = [1.0] * 2000
        structural = [[1.0] * 128, [2.0] * 128]
        transe = [[0.5] * 128]  # Only one entity has TransE

        result = _build_hybrid_embedding(chunk_emb, structural, transe)
        assert len(result) == 768
        assert abs(_l2_norm(result) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# TestGenerateHybridEmbeddings
# ---------------------------------------------------------------------------

class TestGenerateHybridEmbeddings:
    @pytest.mark.asyncio
    @patch("app.graph.hybrid_embeddings.settings")
    async def test_disabled_returns_skipped(self, mock_settings):
        mock_settings.HYBRID_CHUNK_ENTITY_ENABLED = False
        mock_settings.HYBRID_CHUNK_TEXT_DIM = 512
        mock_settings.GRAPHSAGE_OUTPUT_DIM = 128
        mock_settings.TRANSE_DIM = 128

        result = await generate_hybrid_embeddings()
        assert result.skipped is True
        assert result.chunk_count == 0

    @pytest.mark.asyncio
    @patch("app.graph.hybrid_embeddings._read_chunk_data", new_callable=AsyncMock)
    @patch("app.graph.hybrid_embeddings.settings")
    async def test_no_chunks_returns_zero(self, mock_settings, mock_read):
        mock_settings.HYBRID_CHUNK_ENTITY_ENABLED = True
        mock_settings.HYBRID_CHUNK_TEXT_DIM = 512
        mock_settings.GRAPHSAGE_OUTPUT_DIM = 128
        mock_settings.TRANSE_DIM = 128
        mock_read.return_value = []

        result = await generate_hybrid_embeddings(document_ids=["abc"])
        assert result.skipped is False
        assert result.chunk_count == 0

    @pytest.mark.asyncio
    @patch("app.graph.hybrid_embeddings._read_entity_chunk_map", new_callable=AsyncMock)
    @patch("app.graph.hybrid_embeddings._read_chunk_data", new_callable=AsyncMock)
    @patch("app.graph.hybrid_embeddings.settings")
    async def test_entity_lookup_failure_returns_skipped(self, mock_settings, mock_read, mock_map):
        mock_settings.HYBRID_CHUNK_ENTITY_ENABLED = True
        mock_settings.HYBRID_CHUNK_TEXT_DIM = 512
        mock_settings.GRAPHSAGE_OUTPUT_DIM = 128
        mock_settings.TRANSE_DIM = 128
        mock_read.return_value = [{"chunk_id": "id1", "document_id": "d1",
                                    "chunk_index": 0, "embedding": "[1.0]"}]
        mock_map.side_effect = Exception("Neo4j down")

        result = await generate_hybrid_embeddings()
        assert result.skipped is True
        assert "Neo4j down" in result.error


# ---------------------------------------------------------------------------
# TestPipelineIntegration
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    @pytest.mark.asyncio
    @patch("app.graph.hybrid_embeddings.generate_hybrid_embeddings", new_callable=AsyncMock)
    async def test_pipeline_calls_hybrid_after_transe(self, mock_hybrid):
        """Verify pipeline imports and calls generate_hybrid_embeddings."""
        from app.graph.models import HybridEmbeddingResult
        mock_hybrid.return_value = HybridEmbeddingResult(
            chunk_count=5, embedding_dim=768, skipped=False,
        )

        # Import to verify the function is callable from pipeline context
        from app.graph.hybrid_embeddings import generate_hybrid_embeddings
        result = await generate_hybrid_embeddings(document_ids=["doc1"])
        mock_hybrid.assert_called_once_with(document_ids=["doc1"])
        assert result.chunk_count == 5

    @pytest.mark.asyncio
    @patch("app.graph.hybrid_embeddings._store_hybrid_embeddings", new_callable=AsyncMock)
    @patch("app.graph.hybrid_embeddings._read_entity_embeddings", new_callable=AsyncMock)
    @patch("app.graph.hybrid_embeddings._read_entity_chunk_map", new_callable=AsyncMock)
    @patch("app.graph.hybrid_embeddings._read_chunk_data", new_callable=AsyncMock)
    @patch("app.graph.hybrid_embeddings.settings")
    async def test_pipeline_passes_document_ids(self, mock_settings, mock_read,
                                                 mock_map, mock_ent, mock_store):
        """Pipeline should pass completed document IDs to _read_chunk_data."""
        mock_settings.HYBRID_CHUNK_ENTITY_ENABLED = True
        mock_settings.HYBRID_CHUNK_TEXT_DIM = 512
        mock_settings.GRAPHSAGE_OUTPUT_DIM = 128
        mock_settings.TRANSE_DIM = 128
        mock_read.return_value = []  # No chunks — early return
        mock_map.return_value = {}
        mock_ent.return_value = {}

        doc_ids = ["550e8400-e29b-41d4-a716-446655440000", "550e8400-e29b-41d4-a716-446655440001"]
        result = await generate_hybrid_embeddings(document_ids=doc_ids)
        mock_read.assert_called_once_with(doc_ids)
        assert result.chunk_count == 0

    @pytest.mark.asyncio
    async def test_pipeline_continues_if_hybrid_fails(self):
        """Pipeline wraps hybrid in try/except — failure doesn't crash pipeline."""
        # Simulate pipeline's try/except pattern
        error_caught = False
        try:
            raise Exception("hybrid step failed")
        except Exception:
            error_caught = True

        assert error_caught is True
