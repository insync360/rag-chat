"""Dataclasses for the agentic retrieval engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class QueryType(str, Enum):
    SIMPLE = "SIMPLE"          # single fact → vector only
    FILTERED = "FILTERED"      # metadata-filtered → vector with filters
    GRAPH = "GRAPH"            # relationship-based → vector + graph parallel
    ANALYTICAL = "ANALYTICAL"  # multi-hop → all agents


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    content: str
    score: float
    section_path: str
    metadata: dict = field(default_factory=dict)
    source: str = "vector"  # "vector" | "bm25" | "graph" | "fused"
    filename: str = ""
    version: int = 1
    ingested_at: str = ""


@dataclass
class GraphPath:
    entities: list[str] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)
    source_chunks: list[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class ConflictResolution:
    claim_a: str
    claim_b: str
    resolution: str
    winner_chunk_id: str
    reason: str


@dataclass
class ExecutionPlan:
    query_type: QueryType
    activate_vector: bool = True
    activate_graph: bool = False
    activate_calculator: bool = False
    activate_conflict: bool = True
    metadata_filters: dict = field(default_factory=dict)
    expanded_queries: list[str] = field(default_factory=list)


@dataclass
class QueryResult:
    answer: str
    chunks_used: list[RetrievedChunk] = field(default_factory=list)
    graph_paths: list[GraphPath] = field(default_factory=list)
    conflicts: list[ConflictResolution] = field(default_factory=list)
    query_type: QueryType = QueryType.SIMPLE
    cached: bool = False
    step_timings: dict = field(default_factory=dict)
    skipped: bool = False
    error: str | None = None
