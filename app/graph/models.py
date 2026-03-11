"""Dataclasses for graph entity/relationship extraction."""

from dataclasses import dataclass, field


@dataclass
class Entity:
    name: str
    type: str
    source_chunk_index: int
    source_document_id: str
    properties: dict = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class Relationship:
    source_entity: str
    target_entity: str
    type: str
    source_chunk_index: int
    source_document_id: str
    confidence: float = 1.0
    properties: dict = field(default_factory=dict)


@dataclass
class GraphExtractionResult:
    entities: list[Entity]
    relationships: list[Relationship]
    entity_count: int
    relationship_count: int
    skipped: bool = False
    error: str | None = None


@dataclass
class CommunityInfo:
    community_id: int
    entity_names: list[str]
    entity_types: list[str]
    relationship_types: list[str]
    size: int
    summary: str | None = None


@dataclass
class CommunityDetectionResult:
    communities: list[CommunityInfo]
    total_entities: int
    total_communities: int
    skipped: bool = False
    error: str | None = None


@dataclass
class GraphEmbeddingResult:
    entity_count: int
    embedding_dim: int
    skipped: bool = False
    error: str | None = None
    retrained: bool = False


@dataclass
class TransEResult:
    entity_count: int
    relation_count: int
    embedding_dim: int
    skipped: bool = False
    error: str | None = None
    retrained: bool = False


@dataclass
class CommunitySummaryEmbeddingResult:
    community_count: int
    embedding_dim: int
    skipped: bool = False
    error: str | None = None
