"""Tests for community detection via Leiden algorithm."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.graph.models import CommunityDetectionResult


# ---------------------------------------------------------------------------
# 1. Disabled returns skipped
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_disabled_returns_skipped():
    with patch("app.graph.community.settings") as mock_settings:
        mock_settings.COMMUNITY_DETECTION_ENABLED = False
        from app.graph.community import detect_communities
        result = await detect_communities()

    assert result.skipped is True
    assert result.total_entities == 0
    assert result.error is None


# ---------------------------------------------------------------------------
# 2. Leiden unavailable returns skipped
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_leiden_unavailable_returns_skipped():
    with patch("app.graph.community.settings") as mock_settings, \
         patch("app.graph.community._check_leiden", return_value=False):
        mock_settings.COMMUNITY_DETECTION_ENABLED = True
        from app.graph.community import detect_communities
        result = await detect_communities()

    assert result.skipped is True
    assert result.error == "leidenalg/igraph not installed"


# ---------------------------------------------------------------------------
# 3. Empty graph
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_graph():
    with patch("app.graph.community.settings") as mock_settings, \
         patch("app.graph.community._check_leiden", return_value=True), \
         patch("app.graph.community._read_graph", new_callable=AsyncMock, return_value=([], [])):
        mock_settings.COMMUNITY_DETECTION_ENABLED = True
        from app.graph.community import detect_communities
        result = await detect_communities()

    assert result.skipped is False
    assert result.total_entities == 0
    assert result.total_communities == 0
    assert result.error is None


# ---------------------------------------------------------------------------
# 4. Build igraph structure
# ---------------------------------------------------------------------------

def test_build_igraph_structure():
    from app.graph.community import _build_igraph

    entities = [
        {"neo4j_id": 1, "name": "A", "type": "Person"},
        {"neo4j_id": 2, "name": "B", "type": "Person"},
        {"neo4j_id": 3, "name": "C", "type": "Org"},
        {"neo4j_id": 4, "name": "D", "type": "Org"},
    ]
    relationships = [
        {"source_id": 1, "target_id": 2, "rel_type": "KNOWS", "weight": 0.9},
        {"source_id": 3, "target_id": 4, "rel_type": "PART_OF", "weight": 0.8},
        {"source_id": 1, "target_id": 3, "rel_type": "WORKS_AT", "weight": 0.7},
    ]

    g, id_to_idx, _ = _build_igraph(entities, relationships)

    assert g.vcount() == 4
    assert g.ecount() == 3
    assert g.vs["name"] == ["A", "B", "C", "D"]
    assert g.es["weight"] == [0.9, 0.8, 0.7]


# ---------------------------------------------------------------------------
# 5. Build igraph filters self-loops
# ---------------------------------------------------------------------------

def test_build_igraph_filters_self_loops():
    from app.graph.community import _build_igraph

    entities = [
        {"neo4j_id": 1, "name": "A", "type": "Person"},
        {"neo4j_id": 2, "name": "B", "type": "Person"},
    ]
    relationships = [
        {"source_id": 1, "target_id": 1, "rel_type": "SELF_REF", "weight": 1.0},
        {"source_id": 1, "target_id": 2, "rel_type": "KNOWS", "weight": 0.9},
    ]

    g, _, _ = _build_igraph(entities, relationships)

    assert g.vcount() == 2
    assert g.ecount() == 1  # self-loop filtered out


# ---------------------------------------------------------------------------
# 6. Leiden produces two clusters
# ---------------------------------------------------------------------------

def test_run_leiden_two_clusters():
    import igraph
    from app.graph.community import _run_leiden

    # Two clear pairs: (0-1) and (2-3), no cross-edges
    g = igraph.Graph(n=4, edges=[(0, 1), (2, 3)], directed=False)
    g.es["weight"] = [1.0, 1.0]

    membership = _run_leiden(g)

    assert len(membership) == 4
    # Nodes 0 and 1 should be in the same community
    assert membership[0] == membership[1]
    # Nodes 2 and 3 should be in the same community
    assert membership[2] == membership[3]
    # The two pairs should be in different communities
    assert membership[0] != membership[2]


# ---------------------------------------------------------------------------
# 7. Write communities Cypher
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_communities_cypher():
    mock_session = AsyncMock()

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_driver = AsyncMock()
    mock_driver.session = MagicMock(return_value=mock_ctx)

    with patch("app.graph.community.get_driver", new_callable=AsyncMock, return_value=mock_driver):
        from app.graph.community import _write_community_ids
        await _write_community_ids([
            {"neo4j_id": 1, "community_id": 0},
            {"neo4j_id": 2, "community_id": 1},
        ])

    call_args = mock_session.run.call_args
    cypher = call_args[0][0]
    assert "SET e.community_id" in cypher
    assert "UNWIND" in cypher


# ---------------------------------------------------------------------------
# 8. Summary generation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summary_generation():
    from app.graph.community import _generate_community_summaries
    from app.graph.models import CommunityInfo

    communities = [
        CommunityInfo(
            community_id=0,
            entity_names=["Acme Corp", "John"],
            entity_types=["Org", "Person"],
            relationship_types=["EMPLOYS"],
            size=2,
        ),
    ]

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "A corporate employment cluster."

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    with patch("app.graph.community.settings") as mock_settings:
        mock_settings.COMMUNITY_SUMMARY_ENABLED = True
        mock_settings.COMMUNITY_MIN_SIZE = 2
        mock_settings.COMMUNITY_SUMMARY_MODEL = "gpt-4o-mini"
        mock_settings.OPENAI_API_KEY = "test-key"

        with patch("app.graph.community.AsyncOpenAI", return_value=mock_client):
            result = await _generate_community_summaries(communities)

    assert result[0].summary == "A corporate employment cluster."
    mock_client.chat.completions.create.assert_called_once()


# ---------------------------------------------------------------------------
# 9. Summary skips small communities
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summary_skips_small_communities():
    from app.graph.community import _generate_community_summaries
    from app.graph.models import CommunityInfo

    communities = [
        CommunityInfo(
            community_id=0,
            entity_names=["Lonely"],
            entity_types=["Person"],
            relationship_types=[],
            size=1,
        ),
    ]

    mock_client = AsyncMock()

    with patch("app.graph.community.settings") as mock_settings:
        mock_settings.COMMUNITY_SUMMARY_ENABLED = True
        mock_settings.COMMUNITY_MIN_SIZE = 3  # size=1 < 3 → skip
        mock_settings.OPENAI_API_KEY = "test-key"

        with patch("app.graph.community.AsyncOpenAI", return_value=mock_client):
            result = await _generate_community_summaries(communities)

    assert result[0].summary is None
    mock_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# 10. Neo4j failure returns skipped
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_neo4j_failure_returns_skipped():
    with patch("app.graph.community.settings") as mock_settings, \
         patch("app.graph.community._check_leiden", return_value=True), \
         patch("app.graph.community._read_graph", new_callable=AsyncMock,
               side_effect=Exception("Neo4j connection refused")):
        mock_settings.COMMUNITY_DETECTION_ENABLED = True
        from app.graph.community import detect_communities
        result = await detect_communities()

    assert result.skipped is True
    assert "Neo4j connection refused" in result.error
    assert isinstance(result, CommunityDetectionResult)
