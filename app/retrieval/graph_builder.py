"""LangGraph StateGraph wiring — conditional routing and parallel fan-out."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from app.config import settings
from app.retrieval.agents import (
    calculator_agent,
    conflict_agent,
    graph_agent,
    planner_agent,
    summariser_agent,
    vector_agent,
)
from app.retrieval.models import (
    ConflictResolution,
    ExecutionPlan,
    GraphPath,
    QueryType,
    RetrievedChunk,
)


# ---------------------------------------------------------------------------
# Graph state (TypedDict with Annotated reducers for parallel fan-out)
# ---------------------------------------------------------------------------

def _merge_dicts(left: dict, right: dict) -> dict:
    """Merge two dicts, right overwrites left."""
    merged = dict(left)
    merged.update(right)
    return merged


class GraphState(TypedDict):
    original_query: str
    plan: ExecutionPlan | None
    retrieved_chunks: Annotated[list[RetrievedChunk], operator.add]
    graph_paths: Annotated[list[GraphPath], operator.add]
    conflicts: list[ConflictResolution]
    calculation_result: str | None
    final_answer: str
    errors: Annotated[list[str], operator.add]
    step_timings: Annotated[dict, _merge_dicts]
    pass_count: int
    # Embeddings carried through state for downstream agents
    query_embedding_256: list[float] | None
    query_embedding_768: list[float] | None


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _route_after_planner(state: GraphState) -> list[str] | str:
    """Route from planner based on activate_graph flag (set by classifier + conflict override)."""
    plan = state.get("plan")
    if plan is None:
        return "vector"

    if plan.activate_graph:
        return ["vector", "graph"]  # parallel fan-out
    return "vector"


def _route_after_conflict(state: GraphState) -> str:
    """Route from conflict: calculator for ANALYTICAL, else summariser."""
    plan = state.get("plan")
    if plan and plan.activate_calculator:
        return "calculator"
    return "summariser"


_COVERAGE_FAILURE_PHRASES = (
    "insufficient",
    "could not find",
    "no relevant information",
    "unable to locate",
    "not enough context",
    "failed to generate",
)


def _route_after_summariser(state: GraphState) -> str:
    """Optional iterative refinement — re-run vector if coverage insufficient."""
    if not settings.COVERAGE_CHECK_ENABLED:
        return END

    pass_count = state.get("pass_count", 0)
    if pass_count >= settings.MAX_RETRIEVAL_PASSES:
        return END

    plan = state.get("plan")
    answer = state.get("final_answer", "")
    chunks = state.get("retrieved_chunks", [])
    answer_lower = answer.lower()

    # Check 1: explicit failure phrases
    phrase_match = any(p in answer_lower for p in _COVERAGE_FAILURE_PHRASES)

    # Check 2: chunk count below query-type-aware minimum
    min_chunks = settings.COVERAGE_MIN_CHUNKS_ANALYTICAL if (
        plan and plan.query_type == QueryType.ANALYTICAL
    ) else settings.COVERAGE_MIN_CHUNKS_SIMPLE
    too_few_chunks = len(chunks) < min_chunks

    # Check 3: very short answer for complex query
    short_answer = (
        plan
        and plan.query_type in (QueryType.GRAPH, QueryType.ANALYTICAL)
        and len(answer) < 200
    )

    if phrase_match or too_few_chunks or short_answer:
        # On retry: clear metadata_filters to broaden search
        if plan and plan.metadata_filters:
            plan.metadata_filters = {}
        return "vector"

    return END


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_retrieval_graph() -> StateGraph:
    """Build and compile the LangGraph retrieval graph."""
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("planner", planner_agent)
    graph.add_node("vector", vector_agent)
    graph.add_node("graph", graph_agent)
    graph.add_node("conflict", conflict_agent)
    graph.add_node("calculator", calculator_agent)
    graph.add_node("summariser", summariser_agent)

    # Entry point
    graph.set_entry_point("planner")

    # Planner → conditional fan-out
    graph.add_conditional_edges("planner", _route_after_planner, ["vector", "graph"])

    # Vector/Graph → conflict (join point)
    graph.add_edge("vector", "conflict")
    graph.add_edge("graph", "conflict")

    # Conflict → conditional (calculator or summariser)
    graph.add_conditional_edges("conflict", _route_after_conflict, ["calculator", "summariser"])

    # Calculator → summariser
    graph.add_edge("calculator", "summariser")

    # Summariser → conditional (END or re-run vector)
    graph.add_conditional_edges("summariser", _route_after_summariser, ["vector", END])

    return graph.compile()
