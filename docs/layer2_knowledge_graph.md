# Layer 2: Knowledge Graph & Graph RAG — Technical Documentation

> **Status**: Production-ready (Layer 2 complete)
> **Last updated**: 2026-03-11

---

## 1. Overview

Layer 2 adds a knowledge graph pipeline that extracts entities and relationships from document chunks and persists them to Neo4j. This enables multi-hop reasoning, relationship discovery, and causal chain analysis that vector-only retrieval cannot do.

The graph extraction step is **non-blocking** — if Neo4j is down or extraction fails, the ingestion pipeline continues normally.

**Extended pipeline:**

```
File → [1. Parse] → [2. Version Track] → [3. Chunk] → [4. Enrich] → [5. Graph Extract] → [6. Save]
```

Step 5 was added by Layer 2. Steps 1–4 and 6 are from Layer 1.

**Graph extraction sub-pipeline:**

```
Chunks → [Coref Resolution] → [GPT-4o Extraction] → [3-Tier Dedup] → [Neo4j Store] → [Community Detection] → [Community Summary Embeddings] → [Graph Embeddings] → [TransE Embeddings] → [Hybrid Chunk-Entity Embeddings]
```

| Sub-step | Module | Purpose |
|----------|--------|---------|
| Coref Resolution | `coref.py` | Resolve pronouns/aliases before extraction |
| GPT-4o Extraction | `extractor.py` | 1 LLM call per chunk → entities + relationships |
| 3-Tier Dedup | `dedup.py` | Exact → fuzzy → embedding entity deduplication |
| Neo4j Store | `store.py` | MERGE upsert entities + relationships |
| Community Detection | `community.py` | Leiden clustering + LLM summaries |
| Community Summary Embeddings | `community_embeddings.py` | Embed community summaries → Neon pgvector |
| Graph Embeddings | `embeddings.py` | GraphSAGE structural embeddings → Neon pgvector |
| TransE Embeddings | `transe.py` | TransE relation embeddings → Neon pgvector |
| Hybrid Chunk-Entity Embeddings | `hybrid_embeddings.py` | Combined text + graph structure embeddings → Neon pgvector |

---

## 2. File Map

### Module: `app/graph/`

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 150 | `extract_and_store_graph()` — single integration point called by pipeline |
| `models.py` | 88 | Dataclasses: Entity, Relationship, GraphExtractionResult, CommunityInfo, CommunityDetectionResult, GraphEmbeddingResult, TransEResult, CommunitySummaryEmbeddingResult, HybridEmbeddingResult |
| `extractor.py` | 134 | GPT-4o entity + relationship extraction (1 call per chunk) |
| `coref.py` | 132 | Coreference resolution via fastcoref (FCoref model) |
| `dedup.py` | 246 | 3-tier entity dedup (exact → fuzzy → embedding) + relationship dedup |
| `schema.py` | 51 | Idempotent Neo4j constraints, indexes, and auto-migrations |
| `store.py` | 166 | MERGE entities/relationships into Neo4j, deprecation functions |
| `community.py` | 299 | Leiden community detection + GPT-4o-mini summaries |
| `embeddings.py` | 424 | GraphSAGE structural embeddings (pure PyTorch) + Neon pgvector storage |
| `community_embeddings.py` | 233 | Community summary embeddings (OpenAI) + Neon pgvector storage |
| `transe.py` | 361 | TransE relation embeddings (pure PyTorch) + Neon pgvector storage |
| `hybrid_embeddings.py` | 254 | Hybrid chunk-entity embeddings (text + GraphSAGE + TransE) → Neon pgvector |
| `neo4j_client.py` | 26 | Async Neo4j driver singleton |

### Modified Files

| File | Lines | Change |
|------|-------|--------|
| `app/config.py` | 92 | Neo4j + graph extraction + coref + incremental + community + community summary embeddings + GraphSAGE + TransE + hybrid settings |
| `app/ingestion/pipeline.py` | 340 | Added `graph_extract` step, post-batch community detection + community summary embeddings + graph embeddings + TransE + hybrid |
| `requirements.txt` | — | Added `neo4j`, `fastcoref`, `transformers`, `rapidfuzz`, `leidenalg`, `igraph`, `torch` |

### Test Files

| File | Lines | Tests | Coverage |
|------|-------|-------|----------|
| `Test Cases/test_graph_extractor.py` | 145 | 6 | LLM extraction success, tiny chunk skip, failure fallback, invalid JSON retry, concurrency, freeform types |
| `Test Cases/test_graph_dedup.py` | 93 | 6 | Exact duplicate merge, different types kept, highest confidence wins, property merge, empty input, relationship dedup |
| `Test Cases/test_graph_dedup_enhanced.py` | 244 | 7 | 3-tier dedup (exact → fuzzy → embedding), incremental merge with existing entities |
| `Test Cases/test_graph_store.py` | 100 | 5 | Entity MERGE, relationship MERGE, clear document graph, batching, Neo4j down |
| `Test Cases/test_graph_coref.py` | 335 | 5 | Sliding window, short mention replacement, graceful degradation |
| `Test Cases/test_graph_integration.py` | 126 | 4 | Full flow success, Neo4j failure returns skipped, disabled flag skips, pipeline continues on failure |
| `Test Cases/test_graph_embeddings.py` | 201 | 11 | Adjacency building, GraphSAGE model shape/normalization, isolated nodes, disabled/missing-torch skip, pipeline integration |
| `Test Cases/test_community_detection.py` | 249 | 6 | Leiden clustering, LLM summaries, community assignment |
| `Test Cases/test_community_embeddings.py` | 207 | 11 | Enriched text building, disabled/empty/no-summaries/OpenAI-failure skip, pipeline integration |
| `Test Cases/test_transe.py` | 271 | 13 | Triple data building, TransE model shape/normalization/translation, disabled/missing-torch/empty skip, pipeline integration |
| `Test Cases/test_hybrid_embeddings.py` | 228 | 15 | Truncate/normalize, mean pooling, hybrid building, disabled/empty/failure skip, pipeline integration |
| `Test Cases/test_incremental_graph.py` | 322 | 7 | Incremental extraction, entity merging, chunk deprecation |
| `Test Cases/test_layer2_e2e.py` | 266 | 1 | Full E2E: Neon chunks → graph extraction → community detection → Neo4j verification |

---

## 3. Module Deep Dives

### 3.1 Integration Point (`app/graph/__init__.py`)

**Function**: `extract_and_store_graph(chunks, doc_record, *, old_document_id, changed_indices, run_community_detection) -> GraphExtractionResult`

Orchestrates the full graph extraction sub-pipeline. Two modes:

**Full extraction** (default):
1. Run coreference resolution on all chunks (if `COREF_ENABLED`)
2. Extract entities + relationships via GPT-4o (1 call per chunk)
3. Deduplicate via 3-tier pipeline (exact → fuzzy → embedding)
4. Ensure Neo4j schema (constraints + indexes + migrations)
5. Deprecate old entities for this document (`clear_document_graph`)
6. Store new entities + relationships
7. Optionally run community detection

**Incremental extraction** (when `INCREMENTAL_GRAPH_ENABLED` + `changed_indices` + `old_document_id` provided):
1. Filter chunks to only changed indices (saves GPT-4o cost)
2. Coref + extract on changed chunks only
3. Fetch existing entities from Neo4j for the old document
4. Run enhanced dedup merging new + existing entities
5. Deprecate old entities from changed chunks only (`deprecate_chunk_entities`)
6. Store merged results

**Non-blocking**: Catches all exceptions, returns `GraphExtractionResult(skipped=True)` on failure. The pipeline wraps this in an additional try/except for defense-in-depth.

**Deferred imports**: `coref` and `community` modules are imported inside the function body to avoid circular imports (`app.graph → app.ingestion → app.graph`).

---

### 3.2 Data Models (`app/graph/models.py`)

9 dataclasses:

```python
@dataclass
class Entity:
    name: str                    # Canonical name (e.g. "Acme Corporation")
    type: str                    # Extensible string (e.g. "Organization")
    source_chunk_index: int      # 0-indexed chunk position
    source_document_id: str      # Document UUID
    properties: dict             # Additional attributes from LLM
    confidence: float = 1.0      # 0.0-1.0

@dataclass
class Relationship:
    source_entity: str           # Must match an Entity.name
    target_entity: str           # Must match an Entity.name
    type: str                    # UPPER_SNAKE_CASE (e.g. "REPORTS_TO")
    source_chunk_index: int
    source_document_id: str
    confidence: float = 1.0
    properties: dict

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
```

### Entity Types (extensible, not enum-restricted)

Person, Organization, Product, Policy, Date, Location, Metric, Event, Technology, Role, Document, Regulation

### Relationship Types (extensible, not enum-restricted)

REPORTS_TO, SUPERSEDES, REFERENCES, CAUSED_BY, APPLIES_TO, DEFINED_IN, MEMBER_OF, LOCATED_IN, PRODUCES, EMPLOYS

---

### 3.3 Entity/Relationship Extraction (`app/graph/extractor.py`)

**Function**: `extract_from_chunks(chunks, document_id, resolved_texts) -> (list[Entity], list[Relationship])`

- **Model**: GPT-4o (`GRAPH_EXTRACTION_MODEL`)
- **1 LLM call per chunk** — extracts both entities and relationships in a single call
- **Concurrency**: `asyncio.Semaphore(GRAPH_EXTRACTION_CONCURRENCY)` (3 concurrent)
- **All chunks processed via `asyncio.gather`**

**API call config:**
- `response_format`: `{"type": "json_object"}`
- `temperature`: 0
- `max_tokens`: 2048

**System prompt** instructs GPT-4o to return JSON with `entities` and `relationships` arrays. Suggests common entity/relationship types but accepts any string.

**Chunk skip threshold**: Chunks with < 10 tokens are skipped (no LLM call).

**Coreference integration**: If `resolved_texts` is provided, uses the resolved text instead of the raw chunk content for the LLM call.

**Retry logic**: 3 attempts with exponential backoff (`2^attempt` seconds: 1s, 2s). Catches `json.JSONDecodeError`, `KeyError`, and generic exceptions.

**Never raises** — extraction failures produce empty results for that chunk, pipeline continues.

---

### 3.4 Coreference Resolution (`app/graph/coref.py`)

**Function**: `resolve_coreferences(chunks) -> list[str]`

Resolves pronouns and short aliases to canonical entity names before GPT-4o extraction. Improves entity extraction consistency across chunks.

**Model**: fastcoref `FCoref` (migrated from spaCy + coreferee in an earlier iteration).

**Sliding window**: For each chunk, runs the model on `[prev_chunk + current_chunk]` to capture cross-chunk references. Only replaces mentions within the current chunk's character span.

**Replacement rules**:
- Only replaces short mentions (≤ 3 words) with the longest mention from the coreference cluster
- Replacements applied right-to-left to preserve character offsets
- Skips chunks < 10 tokens

**Device**: CPU (`device="cpu"`)

**Singleton**: Model is lazy-loaded on first call via `_get_model()`. Once loaded, reused across all subsequent calls.

**Async**: Model inference runs in `loop.run_in_executor()` (thread pool) to avoid blocking the event loop.

**Graceful degradation**:
- If fastcoref is not installed → returns original chunk content
- If model fails on a specific chunk → returns original content for that chunk, continues with others
- Never raises

---

### 3.5 Deduplication (`app/graph/dedup.py`)

**Function**: `deduplicate_entities_enhanced(entities, existing_entities) -> list[Entity]`

3-tier deduplication pipeline, scoped by entity type:

| Tier | Method | Threshold | Tool |
|------|--------|-----------|------|
| 1 — Exact | Normalized name match | Exact | `_normalize_name()` (lowercase, strip, remove trailing punctuation) |
| 2 — Fuzzy | Token sort ratio | 85.0 / 100 | `rapidfuzz.fuzz.token_sort_ratio` |
| 3 — Embedding | Cosine similarity | 0.92 / 1.0 | `text-embedding-3-large` (256 dimensions) |

**Flow**:
1. Group entities by `(normalized_name, type)` → exact matches
2. Within each type, merge groups where fuzzy score ≥ 85.0
3. Within each type, merge remaining groups where embedding similarity ≥ 0.92
4. For each final group, merge: highest confidence entity wins name + confidence, all properties merged

**Incremental mode**: When `existing_entities` is provided (re-ingestion), prepends them to the entity list before dedup. The 3-tier pipeline naturally merges new entities with existing ones.

**Graceful degradation**:
- If `rapidfuzz` not installed → fuzzy tier skipped, falls back to exact + embedding
- If embedding API fails → embedding tier skipped, falls back to exact + fuzzy

**Relationship dedup** (`deduplicate_relationships`):
- Remaps `source_entity` / `target_entity` names to canonical forms from deduplicated entities
- Deduplicates by `(normalized_source, normalized_target, type)`, keeping highest confidence

---

### 3.6 Neo4j Schema (`app/graph/schema.py`)

**Function**: `ensure_schema() -> None`

Idempotent — safe to call on every ingestion run.

**Constraints:**

| Constraint | Label | Properties |
|------------|-------|------------|
| `entity_unique` | `Entity` | `(name, type)` UNIQUE |
| `community_unique` | `Community` | `community_id` UNIQUE |

**Indexes:**

| Index | Label | Property |
|-------|-------|----------|
| `entity_document_ids` | `Entity` | `source_document_ids` |
| `entity_type` | `Entity` | `type` |
| `entity_name` | `Entity` | `name` |
| `entity_status` | `Entity` | `status` |
| `entity_community` | `Entity` | `community_id` |

**Auto-migrations** (run before schema statements):
1. Convert old `source_document_id` (string) → `source_document_ids` (array) on Entity nodes
2. Set `status = 'active'` on Entity nodes where `status` is null
3. Drop old `entity_document` index if it exists

---

### 3.7 Graph Storage (`app/graph/store.py`)

**Entity MERGE** (`store_graph`):
- Uses `MERGE ON (name, type)` — same entity from multiple documents shares one node
- `ON CREATE`: initializes `source_document_ids` as `[doc_id]`, sets `status = 'active'`
- `ON MATCH`: appends `doc_id` to `source_document_ids` array (if not already present), keeps the higher confidence value
- Properties serialized as JSON strings via `json.dumps()` (Neo4j only supports primitive types on properties)
- Batch size: `GRAPH_BATCH_SIZE` (100) via `UNWIND`

**Relationship MERGE**:
- Grouped by type, one Cypher query per relationship type
- Dynamic type labels via f-string (e.g. `[:EMPLOYS]`, `[:REPORTS_TO]`)
- `MATCH` source + target Entity nodes first — if either doesn't exist, relationship silently skipped
- Properties serialized as JSON strings
- Batch size: 100 via `UNWIND`

**Deprecation functions** (deprecate, never delete):
- `clear_document_graph(document_id)` — sets `status = 'deprecated'` + `deprecated_at` on all entities for a document
- `deprecate_chunk_entities(document_id, chunk_indices)` — deprecates only entities from specific chunks (used in incremental mode)
- `get_document_entities(document_id)` — fetches active entities for a document (used to seed incremental dedup)

---

### 3.8 Community Detection (`app/graph/community.py`)

**Function**: `detect_communities() -> CommunityDetectionResult`

Clusters densely connected entities using the Leiden algorithm, then generates LLM summaries per community.

**Algorithm**: Leiden via `leidenalg` + `igraph`
- Reads active entities + relationships from Neo4j using `elementId()` (not deprecated `id()`)
- Builds an undirected igraph.Graph with relationship confidence as edge weights
- Filters self-loops
- Runs in `loop.run_in_executor()` (CPU-bound)
- Resolution parameter: `COMMUNITY_RESOLUTION` (1.0 default, higher = more communities)
- Deterministic: `seed=42`

**Post-processing**:
1. Writes `community_id` back to Entity nodes in Neo4j
2. Builds `CommunityInfo` structs with entity names/types, relationship types, size
3. Generates LLM summaries (if `COMMUNITY_SUMMARY_ENABLED`)
4. Stores `Community` summary nodes in Neo4j

**LLM summaries**:
- Model: GPT-4o-mini (`COMMUNITY_SUMMARY_MODEL`)
- `max_tokens`: 150
- `temperature`: 0.3
- Skips singletons: communities with size < `COMMUNITY_MIN_SIZE` (2) get no summary
- Prompt includes: entity names, entity types, relationship types

**Graceful degradation**:
- If `leidenalg`/`igraph` not installed → returns `skipped=True`
- If `COMMUNITY_DETECTION_ENABLED=False` → returns `skipped=True`
- Never raises — catches all exceptions

---

### 3.9 Graph Embeddings (`app/graph/embeddings.py`)

**Function**: `generate_graph_embeddings(*, force_retrain) -> GraphEmbeddingResult`

Generates 128-dim GraphSAGE structural embeddings for all active entities. Encodes each entity's position in the knowledge graph (neighbors, relationship types) as a vector for entity-level semantic search.

**Pure PyTorch** — no torch-geometric dependency. GraphSAGE with mean aggregation is ~50 lines for the model class.

**Pipeline:**
1. Read active entities + relationships from Neo4j
2. Get OpenAI embeddings (256-dim) as initial node features via `text-embedding-3-large`
3. Build undirected adjacency lists
4. Train 2-layer GraphSAGE via unsupervised link prediction (or load saved model)
5. Inference → 128-dim L2-normalized structural embeddings
6. Batch upsert both structural (128-dim) + text (256-dim) embeddings to Neon `entity_embeddings` table

**GraphSAGE model:**
- 2-layer mean aggregation: 256 → 128 → 128
- Neighbor sampling: K=25 per node per layer (deterministic `seed=42`)
- Isolated nodes: self-loop (use own features as neighbor aggregate)
- L2-normalized output for cosine similarity

**Training:**
- Unsupervised link prediction with BCE loss
- Positive pairs: actual edges; negative sampling: 5:1 ratio (vectorized `torch.randint`)
- Optimizer: Adam, lr=0.01, 200 epochs
- Edge batching: if edges > 512, sample a batch per epoch
- CPU-bound: runs in `loop.run_in_executor()`

**Model persistence:**
- Saves to `models/graphsage_weights.pt` (state_dict + dim config)
- Pipeline calls with `force_retrain=True` (graph has changed after ingestion)
- Standalone calls can use `force_retrain=False` for inference only

**Storage:**
- Neon table: `entity_embeddings` with HNSW indexes on both embedding columns
- `embedding` (128-dim): GraphSAGE structural — find structurally similar entities
- `text_embedding` (256-dim): OpenAI features — text-based entity search
- Upsert via `ON CONFLICT (entity_name, entity_type) DO UPDATE`

**Graceful degradation:**
- If `torch` not installed → returns `skipped=True`
- If `GRAPH_EMBEDDINGS_ENABLED=False` → returns `skipped=True`
- Never raises — catches all exceptions

---

### 3.10 TransE Relation Embeddings (`app/graph/transe.py`)

**Function**: `generate_transe_embeddings(*, force_retrain) -> TransEResult`

Generates 128-dim TransE translational embeddings where `h + r ≈ t` for each (head, relation, tail) triple. Captures typed relationship semantics — complementary to GraphSAGE's neighborhood structure.

**Why both GraphSAGE and TransE:** GraphSAGE encodes neighborhood aggregation (who is near whom). TransE encodes typed translational structure (how entities relate). Together they give the retrieval layer two similarity metrics.

**Pure PyTorch** — same dependency as GraphSAGE.

**Pipeline:**
1. Read (head, relation, tail) triples from Neo4j (includes `type(r)` unlike GraphSAGE)
2. Map entities/relations to integer indices, filter self-loops, deduplicate
3. Train TransE with margin-based ranking loss (or load saved model)
4. Store per-relation embeddings in `relation_embeddings` table
5. UPDATE existing `entity_embeddings` rows with `transe_embedding` column

**TransE model:**
- `nn.Embedding` for entities and relations, Xavier initialized
- L2-normalized output accessors for cosine similarity storage
- Entity key format: `"name::type"` for type disambiguation

**Training:**
- Margin-based ranking loss: `max(0, ||h+r-t|| - ||h'+r-t'|| + margin)`
- Negative sampling: 50/50 corrupt head or tail with random entity
- Entity embeddings L2-normalized after each gradient step (standard TransE constraint)
- Relation embeddings unconstrained during training (translation vectors)
- Optimizer: Adam, lr=0.01, 200 epochs, margin=1.0
- Batch size: 512 triples per epoch (sampled if more)
- CPU-bound: runs in `loop.run_in_executor()`

**Model persistence:**
- Saves to `models/transe_weights.pt` (state_dict + `ent_to_idx` + `rel_to_idx`)
- Pipeline calls with `force_retrain=True` (graph has changed after ingestion)

**Storage:**
- `relation_embeddings` table: one 128-dim embedding per relation type (HNSW indexed)
- `entity_embeddings.transe_embedding`: 128-dim per entity (HNSW indexed)
- Enables vector arithmetic retrieval: find entities related via similar relation types

**Ordering:** Runs AFTER GraphSAGE because `_store_transe_entity_embeddings` does UPDATE on existing `entity_embeddings` rows that GraphSAGE creates via INSERT.

**Graceful degradation:**
- If `torch` not installed → returns `skipped=True`
- If `TRANSE_ENABLED=False` → returns `skipped=True`
- Never raises — catches all exceptions

---

### 3.11 Community Summary Embeddings (`app/graph/community_embeddings.py`)

**Function**: `generate_community_summary_embeddings(communities=None) -> CommunitySummaryEmbeddingResult`

Embeds community summaries via OpenAI `text-embedding-3-large` (512 dims) and stores in Neon pgvector. Enables high-level thematic retrieval: match a user query to community themes first, then drill into entity/chunk results within that community.

**Why 512 dims**: Entity names use 256 dims (1-3 words). Community summaries are 1-2 full sentences with richer semantics — 512 captures more nuance while staying well below the 2000-dim HNSW limit.

**Pipeline:**
1. Get communities (from pipeline parameter or Neo4j standalone path)
2. Filter to communities with summaries (`summary is not None`)
3. Build enriched text per community (entities, types, relationships, summary)
4. Batch embed via OpenAI `text-embedding-3-large` (512 dims)
5. Upsert to `community_summary_embeddings` table
6. Cleanup stale community IDs from prior Leiden runs

**Enriched text format** (mirrors `embedder.py` pattern):
```
Entities: Acme Corp, John Smith, Jane Doe
Entity types: Organization, Person
Relationships: EMPLOYS, REPORTS_TO
Summary: A corporate employment cluster centered on Acme Corp...
```

**Standalone path**: When `communities=None`, reads `Community` nodes + `Entity` nodes grouped by `community_id` + relationship types per community from Neo4j. Used for ad-hoc embedding without running the full pipeline.

**Stale cleanup**: Leiden reassigns community IDs on each run. `_cleanup_stale_communities()` deletes rows from `community_summary_embeddings` where `community_id` is not in the current active set.

**Storage:**
- Neon table: `community_summary_embeddings` with HNSW index on 512-dim embedding
- `community_id INTEGER` unique index for upsert
- `summary_text TEXT` stores the enriched text that was embedded (for display/debugging)

**Ordering:** Runs AFTER community detection (needs summaries) and BEFORE GraphSAGE (independent, but logically groups community work together).

**Graceful degradation:**
- If `COMMUNITY_SUMMARY_EMBEDDING_ENABLED=False` → returns `skipped=True`
- If OpenAI API fails → returns `skipped=True` with error
- Never raises — catches all exceptions

---

### 3.12 Hybrid Chunk-Entity Embeddings (`app/graph/hybrid_embeddings.py`)

**Function**: `generate_hybrid_embeddings(document_ids=None) -> HybridEmbeddingResult`

Creates a 768-dim hybrid embedding per chunk that combines truncated text content with the structural graph context of entities extracted from that chunk. Gives the retrieval layer a single vector capturing both "what the chunk says" and "how its entities connect in the knowledge graph."

**No additional API calls** — reuses existing embeddings from `chunk_embeddings` (2000-dim text) and `entity_embeddings` (128-dim GraphSAGE + 128-dim TransE).

**Composition (768 dims):**
1. **Chunk text (512 dims)** — MRL truncation of 2000-dim `text-embedding-3-large` embedding, L2-normalized
2. **Entity structural mean (128 dims)** — mean-pooled GraphSAGE embeddings of entities in the chunk, L2-normalized
3. **Entity TransE mean (128 dims)** — mean-pooled TransE embeddings of entities in the chunk, L2-normalized
4. **Final L2-normalization** of the full 768-dim concatenated vector

**Why NOT entity text embeddings (256-dim)**: Those embed `"name (type)"` — redundant since entity names appear in the chunk text. GraphSAGE and TransE add genuinely new structural/relational signals.

**Pipeline:**
1. Read chunk IDs + embeddings from Neon (scoped by `document_ids` or all)
2. Read entity-chunk mapping from Neo4j (`source_document_ids[0]` + `source_chunk_index`)
3. Read entity structural + TransE embeddings from Neon
4. For each chunk: look up entities, gather their embeddings, build hybrid
5. Batch UPDATE `hybrid_embedding` column on `chunk_embeddings`

**Entity-chunk mapping**: Uses `source_document_ids[0]` only — `source_chunk_index` is set ON CREATE and only valid for the first extraction document. V1 limitation for multi-document entities.

**No torch/numpy dependency** — pure `math` module for vector operations (truncate, mean, normalize).

**Ordering:** Runs AFTER TransE — needs both chunk embeddings and entity embeddings (GraphSAGE structural + TransE relational) to exist.

**Graceful degradation:**
- Chunks with no entities → entity portions zero-filled (text-only hybrid)
- Entity has structural but no TransE → TransE portion zero-filled for that entity
- Entity in chunk map but not in `entity_embeddings` → skipped
- If `HYBRID_CHUNK_ENTITY_ENABLED=False` → returns `skipped=True`
- Never raises — catches all exceptions

---

## 4. Configuration

All settings from `app/config.py`:

### Neo4j Connection

| Setting | Default | Source |
|---------|---------|--------|
| `NEO4J_URI` | `bolt://localhost:7687` | `.env` |
| `NEO4J_USER` | `neo4j` | `.env` |
| `NEO4J_PASSWORD` | `""` | `.env` |

### Graph Extraction

| Setting | Default | Purpose |
|---------|---------|---------|
| `GRAPH_EXTRACTION_MODEL` | `gpt-4o` | LLM for entity/relationship extraction |
| `GRAPH_EXTRACTION_CONCURRENCY` | `3` | Max parallel LLM calls |
| `GRAPH_EXTRACTION_ENABLED` | `True` | Set `False` to skip graph step entirely |
| `GRAPH_BATCH_SIZE` | `100` | Neo4j `UNWIND` batch size |

### Coreference Resolution

| Setting | Default | Purpose |
|---------|---------|---------|
| `COREF_ENABLED` | `True` | Enable/disable coreference resolution |

### Incremental Graph Updates

| Setting | Default | Purpose |
|---------|---------|---------|
| `INCREMENTAL_GRAPH_ENABLED` | `True` | Enable incremental mode on re-ingestion |
| `ENTITY_FUZZY_THRESHOLD` | `85.0` | rapidfuzz `token_sort_ratio` threshold (0–100) |
| `ENTITY_EMBEDDING_THRESHOLD` | `0.92` | Cosine similarity threshold (0–1) |
| `ENTITY_EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model for tier-3 dedup |
| `ENTITY_EMBEDDING_DIMENSIONS` | `256` | Reduced dims (entity names are short) |

### Community Detection

| Setting | Default | Purpose |
|---------|---------|---------|
| `COMMUNITY_DETECTION_ENABLED` | `True` | Enable/disable community detection |
| `COMMUNITY_SUMMARY_ENABLED` | `True` | Enable LLM summaries per community |
| `COMMUNITY_SUMMARY_MODEL` | `gpt-4o-mini` | LLM for community summaries |
| `COMMUNITY_MIN_SIZE` | `2` | Skip singletons for summaries |
| `COMMUNITY_RESOLUTION` | `1.0` | Leiden resolution parameter (higher = more communities) |

### Community Summary Embeddings

| Setting | Default | Purpose |
|---------|---------|---------|
| `COMMUNITY_SUMMARY_EMBEDDING_ENABLED` | `True` | Enable/disable community summary embedding |
| `COMMUNITY_SUMMARY_EMBEDDING_MODEL` | `text-embedding-3-large` | OpenAI embedding model |
| `COMMUNITY_SUMMARY_EMBEDDING_DIMENSIONS` | `512` | Embedding dimensions (richer than 256 entity dims) |
| `COMMUNITY_SUMMARY_EMBEDDING_BATCH_SIZE` | `2048` | Max texts per OpenAI API call |

### Graph Embeddings (GraphSAGE)

| Setting | Default | Purpose |
|---------|---------|---------|
| `GRAPH_EMBEDDINGS_ENABLED` | `True` | Enable/disable GraphSAGE embeddings |
| `GRAPHSAGE_INPUT_DIM` | `256` | OpenAI initial features dimension |
| `GRAPHSAGE_HIDDEN_DIM` | `128` | Hidden layer dimension |
| `GRAPHSAGE_OUTPUT_DIM` | `128` | Structural embedding dimension |
| `GRAPHSAGE_NEIGHBOR_SAMPLES` | `25` | K neighbors per node per layer |
| `GRAPHSAGE_EPOCHS` | `200` | Training epochs |
| `GRAPHSAGE_LR` | `0.01` | Adam learning rate |
| `GRAPHSAGE_NEG_RATIO` | `5` | Negative samples per positive edge |
| `GRAPHSAGE_BATCH_SIZE` | `512` | Edge batch size for training |
| `GRAPHSAGE_SEED` | `42` | Deterministic seed |
| `GRAPHSAGE_MODEL_DIR` | `models` | Directory for saved model weights |

### TransE Relation Embeddings

| Setting | Default | Purpose |
|---------|---------|---------|
| `TRANSE_ENABLED` | `True` | Enable/disable TransE embeddings |
| `TRANSE_DIM` | `128` | Embedding dimension |
| `TRANSE_EPOCHS` | `200` | Training epochs |
| `TRANSE_LR` | `0.01` | Adam learning rate |
| `TRANSE_MARGIN` | `1.0` | Margin for ranking loss |
| `TRANSE_BATCH_SIZE` | `512` | Triple batch size for training |
| `TRANSE_SEED` | `42` | Deterministic seed |
| `TRANSE_MODEL_DIR` | `models` | Directory for saved model weights |

### Hybrid Chunk-Entity Embeddings

| Setting | Default | Purpose |
|---------|---------|---------|
| `HYBRID_CHUNK_ENTITY_ENABLED` | `True` | Enable/disable hybrid chunk-entity embeddings |
| `HYBRID_CHUNK_TEXT_DIM` | `512` | MRL truncation dimension for chunk text embedding |

Total hybrid dim = `HYBRID_CHUNK_TEXT_DIM + GRAPHSAGE_OUTPUT_DIM + TRANSE_DIM` = 512 + 128 + 128 = 768. Derived from existing settings to avoid config drift.

---

## 5. Key Design Decisions

### Non-blocking graph step
`extract_and_store_graph()` catches all exceptions and returns `GraphExtractionResult(skipped=True)`. The pipeline wraps it in an additional try/except for defense-in-depth. Ingestion never fails due to graph issues.

### Single LLM call per chunk
One GPT-4o call extracts both entities and relationships (not two separate calls). Halves API cost and latency compared to separate extraction.

### Extensible types (not enums)
Entity types and relationship types are plain strings. The LLM prompt suggests common types but accepts any. No code changes needed to support new types.

### 3-tier dedup (exact → fuzzy → embedding)
Progressively more expensive matching tiers ensure high recall without excessive API cost. Each tier is scoped by entity type. Tiers degrade gracefully if dependencies are missing.

### Deprecate, never delete (incremental updates)
Old entities get `status='deprecated'` and a `deprecated_at` timestamp — never `DETACH DELETE`. Historical data remains queryable for audit. On re-ingestion, only changed chunks are re-extracted (incremental mode), saving GPT-4o cost.

### MERGE-based upserts
Entities use `MERGE` on `(name, type)` — if an entity already exists from another document, its confidence is updated to the higher value and the document ID is appended to `source_document_ids`. Prevents duplicate nodes across documents.

### Properties as JSON strings
Neo4j only supports primitive types for node/relationship properties. Entity and relationship `properties` dicts are serialized via `json.dumps()` before storage.

### Coreference before extraction
Running coreference resolution before GPT-4o extraction improves entity consistency by replacing pronouns and short aliases ("he", "the company") with canonical names. Sliding window `[prev_chunk + current_chunk]` captures cross-chunk references.

---

## 6. Error Handling

| Scenario | Behavior |
|----------|----------|
| Neo4j unreachable | `extract_and_store_graph` catches, returns `skipped=True`, pipeline continues |
| GPT-4o rate limit / failure | 3 retries with exponential backoff per chunk, then empty result for that chunk |
| Invalid JSON from GPT-4o | Retry up to 3 times, then skip chunk |
| `GRAPH_EXTRACTION_ENABLED=False` | Step skipped immediately, no LLM or Neo4j calls |
| Re-ingestion (full mode) | `clear_document_graph` deprecates old entities first |
| Re-ingestion (incremental mode) | `deprecate_chunk_entities` deprecates only changed chunk entities |
| Relationship target not in entity list | `MATCH` finds no node, relationship silently skipped |
| Chunk < 10 tokens | Skipped (no LLM call, no coref) |
| fastcoref not installed | Coref skipped, returns original chunk text |
| Coref fails on a chunk | Returns original text for that chunk, continues others |
| `rapidfuzz` not installed | Fuzzy tier skipped, exact + embedding only |
| Embedding API fails | Embedding tier skipped, exact + fuzzy only |
| `leidenalg`/`igraph` not installed | Community detection returns `skipped=True` |
| Community summary fails for one community | Logs warning, that community gets no summary, others proceed |
| Incremental entity fetch fails | Logs warning, falls back to dedup without existing entities |
| `torch` not installed | Graph embeddings and TransE return `skipped=True` |
| `GRAPH_EMBEDDINGS_ENABLED=False` | Graph embeddings skipped immediately |
| `TRANSE_ENABLED=False` | TransE embeddings skipped immediately |
| No triples in Neo4j | TransE returns `error="no valid triples"`, `skipped=False` |
| All triples are self-loops | TransE returns `error="no valid triples"` after filtering |
| Single relation type | TransE works — 1 relation embedding, entities still get TransE embeddings |
| TransE training fails | Returns `skipped=True`, pipeline continues |
| OpenAI embedding API fails during feature generation | Graph embeddings returns `skipped=True`, pipeline continues |
| GraphSAGE training fails | Graph embeddings returns `skipped=True`, pipeline continues |
| `COMMUNITY_SUMMARY_EMBEDDING_ENABLED=False` | Community summary embeddings skipped immediately |
| No communities with summaries | Community summary embeddings returns `community_count=0`, `skipped=False` |
| OpenAI API fails during community embedding | Returns `skipped=True` with error, pipeline continues |
| Stale community IDs from prior Leiden run | `_cleanup_stale_communities()` deletes rows not in current ID set |
| `HYBRID_CHUNK_ENTITY_ENABLED=False` | Hybrid embeddings skipped immediately |
| Chunk has no entities | Entity portions zero-filled → text-only hybrid (512 dims active, 256 dims zero) |
| Entity has structural but no TransE | TransE portion zero-filled for that entity, structural still used |
| Entity in chunk map but not in `entity_embeddings` | Entity skipped, other entities for chunk still used |
| No `entity_embeddings` rows at all | All chunks get text-only hybrids |
| Neo4j/Neon failure during hybrid | Returns `skipped=True` with error, pipeline continues |
| Multi-document entity | Only enriches first extraction chunk (V1 limitation) |

---

## 7. Dependencies

### Python Libraries

| Package | Purpose |
|---------|---------|
| `neo4j` | Async Neo4j driver |
| `fastcoref` | Coreference resolution (FCoref model) |
| `transformers` | Required by fastcoref |
| `rapidfuzz` | Fuzzy string matching for tier-2 entity dedup |
| `leidenalg` | Leiden community detection algorithm |
| `igraph` | Graph data structure for Leiden |
| `openai` | GPT-4o extraction + GPT-4o-mini community summaries |
| `torch` | GraphSAGE structural embeddings (CPU-only) |

### External Services

| Service | Purpose | Config |
|---------|---------|--------|
| Neo4j | Knowledge graph storage | `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` |
| OpenAI API | GPT-4o entity extraction, GPT-4o-mini community summaries, text-embedding-3-large dedup | `OPENAI_API_KEY` |

---

## 8. E2E Test

**File**: `Test Cases/test_layer2_e2e.py`

**Run**: `.venv/Scripts/python "Test Cases/test_layer2_e2e.py"`

**Prerequisites**: Active documents in Neon (run Layer 1 ingestion first), Neo4j running.

**7-step flow**:
1. **Connection verification** — Neon `SELECT 1`, Neo4j `verify_connectivity()`, print feature flags
2. **Fetch documents & chunks** from Neon (`documents` + `chunks` tables)
3. **Clear Neo4j** for clean test run (`MATCH (n) DETACH DELETE n`)
4. **Graph extraction** per document — `extract_and_store_graph()` with timing
5. **Community detection** — `detect_communities()` with timing
6. **Neo4j verification** — count entities/relationships/communities, print top 10 by confidence, community summaries, top 5 communities by size
7. **Cleanup** — close Neo4j + Neon connections, print summary
