"""Tests for community summary embeddings."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.graph.models import CommunityInfo


# ---------------------------------------------------------------------------
# _build_embedding_text
# ---------------------------------------------------------------------------

class TestBuildEmbeddingText:
    def test_full_enriched_text(self):
        from app.graph.community_embeddings import _build_embedding_text

        comm = CommunityInfo(
            community_id=0,
            entity_names=["Acme Corp", "John Smith", "Jane Doe"],
            entity_types=["Organization", "Person"],
            relationship_types=["EMPLOYS", "REPORTS_TO"],
            size=3,
            summary="A corporate employment cluster centered on Acme Corp.",
        )

        text = _build_embedding_text(comm)

        assert "Entities: Acme Corp, John Smith, Jane Doe" in text
        assert "Entity types: Organization, Person" in text
        assert "Relationships: EMPLOYS, REPORTS_TO" in text
        assert "Summary: A corporate employment cluster" in text

    def test_summary_only_fallback(self):
        from app.graph.community_embeddings import _build_embedding_text

        comm = CommunityInfo(
            community_id=1,
            entity_names=[],
            entity_types=[],
            relationship_types=[],
            size=0,
            summary="An isolated theme.",
        )

        text = _build_embedding_text(comm)

        assert "Summary: An isolated theme." in text
        assert "Entities:" not in text
        assert "Entity types:" not in text
        assert "Relationships:" not in text

    def test_no_summary_returns_empty(self):
        from app.graph.community_embeddings import _build_embedding_text

        comm = CommunityInfo(
            community_id=2,
            entity_names=["A"],
            entity_types=["T"],
            relationship_types=[],
            size=1,
            summary=None,
        )

        text = _build_embedding_text(comm)

        assert text == ""


# ---------------------------------------------------------------------------
# generate_community_summary_embeddings
# ---------------------------------------------------------------------------

class TestGenerateCommunityEmbeddings:
    def test_disabled_returns_skipped(self):
        from app.graph.community_embeddings import generate_community_summary_embeddings

        with patch("app.graph.community_embeddings.settings") as mock_settings:
            mock_settings.COMMUNITY_SUMMARY_EMBEDDING_ENABLED = False
            mock_settings.COMMUNITY_SUMMARY_EMBEDDING_DIMENSIONS = 512

            result = asyncio.get_event_loop().run_until_complete(
                generate_community_summary_embeddings()
            )

        assert result.skipped is True
        assert result.community_count == 0

    def test_empty_communities_list(self):
        from app.graph.community_embeddings import generate_community_summary_embeddings

        with patch("app.graph.community_embeddings.settings") as mock_settings:
            mock_settings.COMMUNITY_SUMMARY_EMBEDDING_ENABLED = True
            mock_settings.COMMUNITY_SUMMARY_EMBEDDING_DIMENSIONS = 512

            result = asyncio.get_event_loop().run_until_complete(
                generate_community_summary_embeddings(communities=[])
            )

        assert result.skipped is False
        assert result.community_count == 0

    def test_no_summaries_all_none(self):
        from app.graph.community_embeddings import generate_community_summary_embeddings

        comms = [
            CommunityInfo(
                community_id=0, entity_names=["A"], entity_types=["T"],
                relationship_types=[], size=1, summary=None,
            ),
            CommunityInfo(
                community_id=1, entity_names=["B"], entity_types=["T"],
                relationship_types=[], size=1, summary=None,
            ),
        ]

        with patch("app.graph.community_embeddings.settings") as mock_settings:
            mock_settings.COMMUNITY_SUMMARY_EMBEDDING_ENABLED = True
            mock_settings.COMMUNITY_SUMMARY_EMBEDDING_DIMENSIONS = 512

            result = asyncio.get_event_loop().run_until_complete(
                generate_community_summary_embeddings(communities=comms)
            )

        assert result.skipped is False
        assert result.community_count == 0

    def test_openai_failure_returns_skipped(self):
        from app.graph.community_embeddings import generate_community_summary_embeddings

        comms = [
            CommunityInfo(
                community_id=0, entity_names=["A", "B"], entity_types=["T"],
                relationship_types=["R"], size=2, summary="A cluster about A and B.",
            ),
        ]

        with patch("app.graph.community_embeddings.settings") as mock_settings, \
             patch("app.graph.community_embeddings.AsyncOpenAI") as mock_openai:
            mock_settings.COMMUNITY_SUMMARY_EMBEDDING_ENABLED = True
            mock_settings.COMMUNITY_SUMMARY_EMBEDDING_DIMENSIONS = 512
            mock_settings.COMMUNITY_SUMMARY_EMBEDDING_BATCH_SIZE = 2048
            mock_settings.COMMUNITY_SUMMARY_EMBEDDING_MODEL = "text-embedding-3-large"
            mock_settings.OPENAI_API_KEY = "test-key"

            client = MagicMock()
            client.embeddings.create = AsyncMock(side_effect=RuntimeError("API down"))
            mock_openai.return_value = client

            result = asyncio.get_event_loop().run_until_complete(
                generate_community_summary_embeddings(communities=comms)
            )

        assert result.skipped is True
        assert result.error is not None
        assert "API down" in result.error


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_pipeline_calls_community_embeddings_after_detection(self):
        """Verify community summary embeddings step exists in pipeline."""
        import inspect
        from app.ingestion.pipeline import ingest_files

        source = inspect.getsource(ingest_files)

        assert "community_summary_embeddings" in source
        assert "generate_community_summary_embeddings" in source

    def test_pipeline_captures_community_result(self):
        """Verify pipeline captures detect_communities() return value."""
        import inspect
        from app.ingestion.pipeline import ingest_files

        source = inspect.getsource(ingest_files)

        assert "community_result = await detect_communities()" in source
        assert "community_result.communities" in source

    def test_pipeline_continues_if_embeddings_fail(self):
        """Verify the community embedding step is wrapped in try/except."""
        import inspect
        from app.ingestion.pipeline import ingest_files

        source = inspect.getsource(ingest_files)

        # Find the community summary embeddings block
        idx = source.index("community_summary_embeddings")
        # There should be a try before it and except after
        block = source[max(0, idx - 200) : idx + 500]
        assert "try:" in block
        assert "except Exception" in block

    def test_communities_passed_from_detection_result(self):
        """Verify pipeline passes community_result.communities to embedding fn."""
        import inspect
        from app.ingestion.pipeline import ingest_files

        source = inspect.getsource(ingest_files)

        assert "comm_list = community_result.communities if community_result else None" in source
        assert "generate_community_summary_embeddings(communities=comm_list)" in source
