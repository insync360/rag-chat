# Layer 5: Agentic Retrieval Engine — Technical Documentation

> **Status**: Production-ready (Phase 3 complete)
> **Last updated**: 2026-03-13

---

## 1. Overview

Layer 5 is a LangGraph-orchestrated multi-agent retrieval engine that transforms a user query into a grounded, cited answer. Six agents collaborate via a conditional state graph: a planner classifies the query, vector and graph agents retrieve evidence in parallel, a conflict agent resolves contradictions, an optional calculator handles arithmetic, and a summariser compresses everything into the final answer. An iterative refinement loop re-runs retrieval if coverage is insufficient (max 2 passes).

A semantic cache (256-dim cosine similarity, 0.95 threshold, 24h TTL) short-circuits the graph for repeated queries. Every query is logged to `query_logs` for observability.

**Pipeline:**

```
Query → [Embed 256-dim] → [Cache Check] → [LangGraph] → [Cache Store] → [Log] → QueryResult
```

**LangGraph agent flow:**

```
[Planner] → conditional fan-out → [Vector] ──┐
                                  [Graph]  ───┤ (parallel for GRAPH/ANALYTICAL)
                                              ▼
                                        [Conflict] → conditional → [Calculator] → [Summariser] → conditional → END
                                                                                                            └→ [Vector] (re-run if insufficient)
```

| Agent | Module | Purpose |
|-------|--------|---------|
| Planner | `classifier.py` | Classify query type, expand queries, extract metadata filters |
| Vector | `vector_search.py` + `reranker.py` | Hybrid vector+BM25 search, RRF fusion, Cohere rerank, freshness boost |
| Graph | `graph_search.py` | Entity embedding match, Neo4j traversal (2 hops), chunk resolution |
| Conflict | `conflict.py` | Contradiction detection via GPT-5.4, credibility-based resolution |
| Calculator | `calculator.py` | Structured number extraction + safe arithmetic (no eval/exec) |
| Summariser | `summariser.py` | Compress chunks into cited answer via GPT-5.4 |

---

## 2. Architecture

### Data Flow

```
                     ┌───────────────────────────────────────────────────────────────┐
                     │                    ENTRY POINT (__init__.py)                  │
                     │  1. embed_query_small(256-dim)                                │
                     │  2. check_cache() → hit? return cached result                 │
                     │  3. build_retrieval_graph() → ainvoke()                       │
                     │  4. build QueryResult                                         │
                     │  5. store_cache() (fire-and-forget)                           │
                     │  6. _log_query() (fire-and-forget)                            │
                     └────────────────────────┬──────────────────────────────────────┘
                                              │
                     ┌────────────────────────▼──────────────────────────────────────┐
                     │               LANGGRAPH STATE GRAPH (graph_builder.py)        │
                     │                                                               │
                     │  ┌──────────┐                                                 │
                     │  │ PLANNER  │ classify_query() → ExecutionPlan                 │
                     │  └────┬─────┘                                                 │
                     │       │ _route_after_planner                                  │
                     │       ├──────────────────┐ (GRAPH/ANALYTICAL → parallel)      │
                     │       ▼                  ▼                                    │
                     │  ┌──────────┐    ┌──────────┐                                │
                     │  │  VECTOR  │    │  GRAPH   │                                │
                     │  │ hybrid   │    │ entity→  │                                │
                     │  │ search + │    │ traverse→│                                │
                     │  │ rerank   │    │ chunks   │                                │
                     │  └────┬─────┘    └────┬─────┘                                │
                     │       └───────┬───────┘ (join: chunks merged via operator.add)│
                     │               ▼                                               │
                     │        ┌──────────┐                                           │
                     │        │ CONFLICT │ detect_and_resolve_conflicts()             │
                     │        └────┬─────┘                                           │
                     │             │ _route_after_conflict                            │
                     │             ├──────────────┐ (ANALYTICAL → calculator)         │
                     │             │              ▼                                   │
                     │             │       ┌────────────┐                             │
                     │             │       │ CALCULATOR  │                            │
                     │             │       └──────┬─────┘                             │
                     │             ▼              ▼                                   │
                     │        ┌──────────────┐                                       │
                     │        │  SUMMARISER   │ summarise_chunks()                    │
                     │        └──────┬───────┘                                       │
                     │               │ _route_after_summariser                       │
                     │               ├──→ END (sufficient)                            │
                     │               └──→ VECTOR (re-run if insufficient, ≤2 passes) │
                     └───────────────────────────────────────────────────────────────┘
```

### Routing Logic

| Query Type | Vector | Graph | Calculator | Conflict | Summariser |
|-----------|--------|-------|------------|----------|------------|
| SIMPLE | Yes | No | No | Yes | Yes |
| FILTERED | Yes (with filters) | No | No | Yes | Yes |
| GRAPH | Yes | Yes (parallel) | No | Yes | Yes |
| ANALYTICAL | Yes | Yes (parallel) | Yes | Yes | Yes |

---

## 3. File Map

| File | Lines | Purpose |
|------|-------|---------|
| `app/retrieval/__init__.py` | 125 | `query()` entry point — cache check, LangGraph invoke, cache store, query logging |
| `app/retrieval/models.py` | 68 | Dataclasses: QueryType, RetrievedChunk, GraphPath, ConflictResolution, ExecutionPlan, QueryResult |
| `app/retrieval/classifier.py` | 107 | Query classification — heuristic regex + GPT-5.4 LLM fallback |
| `app/retrieval/cache.py` | 108 | Semantic cache — check + store via 256-dim cosine similarity |
| `app/retrieval/vector_search.py` | 242 | Query embedding, HNSW vector search, BM25 tsvector search, RRF fusion |
| `app/retrieval/graph_search.py` | 309 | Entity embedding match, Neo4j 2-hop traversal, chunk resolution |
| `app/retrieval/reranker.py` | 80 | Cohere Rerank v3.5 cross-encoder + freshness boost |
| `app/retrieval/summariser.py` | 95 | Compress chunks into cited answer via GPT-5.4 |
| `app/retrieval/calculator.py` | 94 | Structured number extraction + safe arithmetic |
| `app/retrieval/conflict.py` | 115 | Contradiction detection + credibility-based resolution |
| `app/retrieval/agents.py` | 255 | Six agent node functions (async state -> dict) with retry wrapper |
| `app/retrieval/graph_builder.py` | 139 | LangGraph StateGraph wiring — conditional routing + parallel fan-out |
| `migrations/011_semantic_cache.sql` | 17 | `semantic_cache` table + HNSW + expiry index |
| `migrations/012_query_logs.sql` | 18 | `query_logs` table + indexes |

---

## 4. Query Types

### SIMPLE — single fact lookup

**Example**: "What is the parental leave policy?"

**Agents activated**: Planner -> Vector -> Conflict -> Summariser

**Behavior**: Vector-only retrieval (hybrid HNSW + BM25). No graph traversal, no calculator.

### FILTERED — metadata-scoped lookup

**Example**: "What does the HR handbook say about overtime?"

**Agents activated**: Planner -> Vector (with `metadata_filters`) -> Conflict -> Summariser

**Behavior**: Same as SIMPLE but vector and BM25 queries include `WHERE d.filename ILIKE $3` or `WHERE c.document_id = $3::uuid` filters extracted by the classifier.

### GRAPH — relationship-based query

**Example**: "How does Acme Corp relate to the FMLA regulation?"

**Agents activated**: Planner -> Vector + Graph (parallel) -> Conflict -> Summariser

**Behavior**: Vector and graph agents run in parallel via LangGraph fan-out. Graph agent finds seed entities via embedding similarity, traverses Neo4j up to 2 hops, resolves to chunks. Results from both agents are merged via `operator.add` reducer on `retrieved_chunks`.

**Heuristic shortcut**: Regex pattern `\b(relationship|connected|between|relate[sd]?|link|how does .+ (?:affect|impact|influence))\b` bypasses LLM classification.

### ANALYTICAL — multi-hop, aggregation, arithmetic

**Example**: "Calculate the total budget difference between 2024 and 2025 across all departments"

**Agents activated**: Planner -> Vector + Graph (parallel) -> Conflict -> Calculator -> Summariser

**Behavior**: Full pipeline. Calculator agent extracts numbers and operation from chunks via structured LLM extraction, computes result with safe Python math, passes to summariser.

**Heuristic shortcut**: Regex pattern `\b(calculate|sum|average|total|mean|percentage|ratio|difference)\b.*\d` bypasses LLM classification.

---

## 5. Hybrid Search

**Module**: `app/retrieval/vector_search.py`

Two parallel search paths fused via Reciprocal Rank Fusion (RRF):

### Vector Search (HNSW)

- **Index**: `idx_chunk_emb_hybrid` — HNSW on `chunk_embeddings.hybrid_embedding` (768-dim, `vector_cosine_ops`, m=16, ef_construction=64)
- **Query embedding**: 2000-dim from `text-embedding-3-large`, truncated to 512-dim via MRL, zero-padded to 768-dim (512 text + 128 zero GraphSAGE + 128 zero TransE), L2-normalized
- **Similarity**: `1 - (hybrid_embedding <=> query::vector)` (cosine distance)
- **Filters**: Active documents only (`d.status = 'active'`), optional `document_id` or `filename ILIKE` filter
- **Top-K**: `RETRIEVAL_TOP_K_VECTOR` (40)

### BM25 Search (tsvector)

- **Index**: `idx_chunks_content_tsvector` — GIN on `to_tsvector('english', content)`
- **Ranking**: `ts_rank_cd` with normalization flag 32 (divides by document length)
- **Query**: `plainto_tsquery('english', $1)` — automatic stemming and stop-word removal
- **Filters**: Same active document + metadata filters as vector search
- **Top-K**: `RETRIEVAL_TOP_K_BM25` (40)

### RRF Fusion

```
RRF_score(chunk) = SUM over lists: 1 / (k + rank_position + 1)
```

- **k**: `RETRIEVAL_RRF_K` (60)
- Chunks appearing in both lists get boosted scores
- Final output: top `RETRIEVAL_TOP_K_FINAL` (10) chunks by RRF score

### Expanded Query Search

When the planner produces `expanded_queries` (1-3 reformulations), the vector agent searches with each query independently, deduplicates by `chunk_id` (keeping highest score), then reranks the merged set.

---

## 6. Graph Search

**Module**: `app/retrieval/graph_search.py`

Three-step pipeline: entity match -> Neo4j traversal -> chunk resolution.

### Step 1 — Find Seed Entities

- Search `entity_embeddings.text_embedding` (256-dim HNSW) for nearest entities to query embedding
- Similarity threshold: > 0.75
- Top-K: `GRAPH_SEARCH_ENTITY_TOP_K` (5)
- **Fallback** (if < 2 embedding matches): GPT-5.4 extracts entity names from query, then case-insensitive `CONTAINS` match against Neo4j `Entity` nodes

### Step 2 — Neo4j Traversal

- Cypher: `MATCH (e)-[r*1..2]-(related:Entity {status: 'active'})` — variable-length 1-2 hop traversal from seed entities
- **Hop weighting**: 1-hop = 1.0, 2-hop = 0.5 confidence
- Deduplicates `(seed_name, related_name)` pairs
- Returns related entities + `GraphPath` objects with entity names, relationship types, confidence

### Step 3 — Chunk Resolution

- Maps entities to chunks via `(source_document_ids[0], source_chunk_index)` pairs from Neo4j
- Fetches chunk content from Neon with `(document_id, chunk_index) IN (...)` batch query
- Scores chunks by hop distance weight
- Also resolves seed entity chunks (not just related entities)

**Never raises** — catches all exceptions, returns empty lists on failure.

---

## 7. Reranking

**Module**: `app/retrieval/reranker.py`

### Cohere Rerank

- **Model**: `rerank-v3.5` (`RERANK_MODEL`)
- **Client**: `cohere.AsyncClientV2`
- **Input**: Up to 20 chunks, content truncated to 4096 chars each
- **Output**: Top `RERANK_TOP_N` (10) chunks with `relevance_score`
- **Fallback**: If Cohere fails or `COHERE_API_KEY` is empty, returns original order truncated to `top_n`

### Freshness Boost

Applied multiplicatively after reranking:

```
freshness = max(0, 1 - days_since_ingestion / FRESHNESS_DECAY_DAYS)
final_score = relevance_score * (1.0 + FRESHNESS_WEIGHT * freshness)
```

| Setting | Default | Purpose |
|---------|---------|---------|
| `FRESHNESS_BOOST_ENABLED` | `True` | Enable/disable freshness boosting |
| `FRESHNESS_DECAY_DAYS` | `365` | Full decay period (1 year) |
| `FRESHNESS_WEIGHT` | `0.05` | Max +5% boost for brand-new documents |

---

## 8. Semantic Cache

**Module**: `app/retrieval/cache.py`

### Mechanism

1. Embed query at 256-dim (`text-embedding-3-large` with `dimensions=256`)
2. Cosine similarity search against `semantic_cache.query_embedding` (HNSW)
3. Hit if `similarity > CACHE_SIMILARITY_THRESHOLD` (0.95) and `expires_at > now()`
4. On miss: run full LangGraph pipeline, store result (fire-and-forget via `asyncio.create_task`)

### Cache Entry

Stored as `(query_text, query_embedding, query_type, answer, result_json, expires_at)`. The `result_json` JSONB column holds the full `QueryResult` payload (chunks, graph paths, conflicts, step timings) for faithful reconstruction on cache hit.

### Configuration

| Setting | Default | Purpose |
|---------|---------|---------|
| `CACHE_ENABLED` | `True` | Enable/disable semantic cache |
| `CACHE_SIMILARITY_THRESHOLD` | `0.95` | Cosine similarity threshold for cache hit |
| `CACHE_TTL_HOURS` | `24` | TTL before expiry |
| `CACHE_EMBEDDING_DIM` | `256` | Query embedding dimension |

### `semantic_cache` Table (migration 011)

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `UUID` | PK, `gen_random_uuid()` | Cache entry UUID |
| `query_text` | `TEXT` | NOT NULL | Original query text |
| `query_embedding` | `vector(256)` | NOT NULL | 256-dim query embedding |
| `query_type` | `TEXT` | NOT NULL | SIMPLE/FILTERED/GRAPH/ANALYTICAL |
| `answer` | `TEXT` | NOT NULL | Generated answer |
| `result_json` | `JSONB` | NOT NULL | Full result payload |
| `created_at` | `TIMESTAMPTZ` | NOT NULL, default `now()` | Creation time |
| `expires_at` | `TIMESTAMPTZ` | NOT NULL, default `now() + 24h` | Expiry time |

**Indexes:**
- `idx_semantic_cache_embedding` — HNSW on `query_embedding vector_cosine_ops` (m=16, ef_construction=64)
- `idx_semantic_cache_expires` — B-tree on `expires_at`

---

## 9. Conflict Resolution

**Module**: `app/retrieval/conflict.py`

### Detection

GPT-5.4 (`CONFLICT_MODEL`) analyzes up to 10 chunks (600 chars each) for contradictory claims. Returns pairs of `(chunk_a_id, chunk_b_id, claim_a, claim_b)`.

### Credibility Hierarchy

Deterministic resolution — no LLM involved:

| Priority | Rule | Reason |
|----------|------|--------|
| 1 | Higher `version` wins | Newer document version supersedes older |
| 2 | Newer `ingested_at` wins | More recently ingested content is fresher |
| 3 | Table content beats paragraph | Tables contain more precise/structured data |
| 4 | Higher `score` wins | Reranking relevance as tiebreaker |

### Output

```python
@dataclass
class ConflictResolution:
    claim_a: str            # short quote from chunk A
    claim_b: str            # short quote from chunk B
    resolution: str         # "Trusting chunk abc123..."
    winner_chunk_id: str    # full chunk UUID
    reason: str             # e.g. "version 2 > version 1"
```

---

## 10. Calculator

**Module**: `app/retrieval/calculator.py`

### Design

No `eval()` or `exec()` — numbers and operation are extracted by GPT-5.4 (`CALCULATOR_MODEL`) as structured JSON, then dispatched to whitelisted Python functions.

### Extraction Prompt

GPT-5.4 returns JSON with:
- `numbers`: array of floats from the context
- `operation`: one of 8 allowed operations
- `context`: description of what the numbers represent
- `applicable`: boolean — false if no calculation needed

### Allowed Operations

| Operation | Implementation |
|-----------|---------------|
| `sum` | `sum(numbers)` |
| `mean` | `statistics.mean(numbers)` |
| `difference` | `numbers[0] - numbers[1]` |
| `ratio` | `numbers[0] / numbers[1]` |
| `percentage_change` | `((numbers[1] - numbers[0]) / numbers[0]) * 100` |
| `min` | `min(numbers)` |
| `max` | `max(numbers)` |
| `count` | `len(numbers)` |

Results are rounded to 4 decimal places.

---

## 11. Iterative Refinement

**Module**: `app/retrieval/graph_builder.py` (`_route_after_summariser`)

After the summariser produces an answer, a coverage check determines whether to loop back:

### Coverage Heuristic

The answer is considered insufficient if any of:
- Answer contains "insufficient" (case-insensitive)
- Answer contains "could not find" (case-insensitive)
- Fewer than 2 chunks were retrieved

### Loop Control

- **Max passes**: `MAX_RETRIEVAL_PASSES` (2)
- **Re-entry point**: Vector agent (re-runs hybrid search, potentially with expanded queries from planner)
- **Toggle**: `COVERAGE_CHECK_ENABLED` (True by default)

On the 2nd pass, the vector agent uses `plan.expanded_queries` (reformulated queries from the planner) to broaden retrieval.

---

## 12. Configuration

All settings from `app/config.py`:

### Retrieval Engine

| Setting | Default | Purpose |
|---------|---------|---------|
| `RETRIEVAL_TOP_K_VECTOR` | `40` | Vector search top-K (pre-fusion) |
| `RETRIEVAL_TOP_K_BM25` | `40` | BM25 search top-K (pre-fusion) |
| `RETRIEVAL_TOP_K_FINAL` | `10` | Final top-K after RRF fusion |
| `RETRIEVAL_RRF_K` | `60` | RRF constant k |
| `RETRIEVAL_USE_HYBRID_EMBEDDING` | `True` | Use 768-dim hybrid vs 2000-dim text embedding |

### Query Classification

| Setting | Default | Purpose |
|---------|---------|---------|
| `QUERY_CLASSIFIER_MODEL` | `gpt-5.4` | LLM for query classification |

### Graph Search

| Setting | Default | Purpose |
|---------|---------|---------|
| `GRAPH_SEARCH_MAX_HOPS` | `2` | Max Neo4j traversal depth |
| `GRAPH_SEARCH_TOP_K` | `20` | Max chunks from graph search |
| `GRAPH_SEARCH_ENTITY_TOP_K` | `5` | Max seed entities from embedding match |

### Reranker (Cohere)

| Setting | Default | Purpose |
|---------|---------|---------|
| `COHERE_API_KEY` | (env) | `.env` — Cohere API key |
| `RERANK_MODEL` | `rerank-v3.5` | Cohere model |
| `RERANK_TOP_N` | `10` | Max reranked results |
| `RERANK_ENABLED` | `True` | Enable/disable Cohere reranking |

### Freshness Boost

| Setting | Default | Purpose |
|---------|---------|---------|
| `FRESHNESS_BOOST_ENABLED` | `True` | Enable/disable freshness boosting |
| `FRESHNESS_DECAY_DAYS` | `365` | Full decay period |
| `FRESHNESS_WEIGHT` | `0.05` | Max boost factor (+5%) |

### Summariser

| Setting | Default | Purpose |
|---------|---------|---------|
| `SUMMARISER_MODEL` | `gpt-5.4` | LLM for answer synthesis |
| `SUMMARISER_MAX_TOKENS` | `4096` | Max output tokens |

### Calculator

| Setting | Default | Purpose |
|---------|---------|---------|
| `CALCULATOR_MODEL` | `gpt-5.4` | LLM for number/operation extraction |

### Conflict Resolution

| Setting | Default | Purpose |
|---------|---------|---------|
| `CONFLICT_MODEL` | `gpt-5.4` | LLM for contradiction detection |

### Semantic Cache

| Setting | Default | Purpose |
|---------|---------|---------|
| `CACHE_ENABLED` | `True` | Enable/disable semantic cache |
| `CACHE_SIMILARITY_THRESHOLD` | `0.95` | Cosine threshold for cache hit |
| `CACHE_TTL_HOURS` | `24` | Cache TTL |
| `CACHE_EMBEDDING_DIM` | `256` | Query embedding dimension |

### Agent Retry

| Setting | Default | Purpose |
|---------|---------|---------|
| `AGENT_MAX_RETRIES` | `2` | Max retries per agent call |
| `AGENT_RETRY_BASE_MS` | `500` | Base delay for exponential backoff (ms) |

### Iterative Refinement

| Setting | Default | Purpose |
|---------|---------|---------|
| `COVERAGE_CHECK_ENABLED` | `True` | Enable/disable coverage loop |
| `MAX_RETRIEVAL_PASSES` | `2` | Max vector retrieval passes |

---

## 13. Database

### `semantic_cache` (migration 011)

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `UUID` | PK, `gen_random_uuid()` | Cache entry UUID |
| `query_text` | `TEXT` | NOT NULL | Original query |
| `query_embedding` | `vector(256)` | NOT NULL | 256-dim query embedding |
| `query_type` | `TEXT` | NOT NULL | Query classification |
| `answer` | `TEXT` | NOT NULL | Generated answer |
| `result_json` | `JSONB` | NOT NULL | Full result payload (chunks, paths, conflicts, timings) |
| `created_at` | `TIMESTAMPTZ` | NOT NULL, default `now()` | Creation time |
| `expires_at` | `TIMESTAMPTZ` | NOT NULL, default `now() + 24h` | Expiry time |

**Indexes:**
- `idx_semantic_cache_embedding` — HNSW on `query_embedding vector_cosine_ops` (m=16, ef_construction=64)
- `idx_semantic_cache_expires` — B-tree on `expires_at`

### `query_logs` (migration 012)

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `UUID` | PK, `gen_random_uuid()` | Log entry UUID |
| `query_text` | `TEXT` | NOT NULL | User query |
| `query_type` | `TEXT` | NOT NULL | SIMPLE/FILTERED/GRAPH/ANALYTICAL |
| `cached` | `BOOLEAN` | NOT NULL, default FALSE | Whether result was from cache |
| `chunks_retrieved` | `INTEGER` | | Number of chunks in result |
| `graph_paths` | `INTEGER` | default 0 | Number of graph paths |
| `conflicts` | `INTEGER` | default 0 | Number of conflicts detected |
| `step_timings` | `JSONB` | default `'{}'` | Per-agent timing (planner, vector, graph, conflict, calculator, summariser, total) |
| `error` | `TEXT` | | Error message if failed |
| `created_at` | `TIMESTAMPTZ` | NOT NULL, default `now()` | Query timestamp |

**Indexes:**
- `idx_query_logs_created` — B-tree on `created_at DESC`
- `idx_query_logs_type` — B-tree on `query_type`

---

## 14. Dependencies

### Python Libraries

| Package | Purpose |
|---------|---------|
| `langgraph` | StateGraph orchestration, conditional routing, parallel fan-out |
| `cohere` | AsyncClientV2 for Rerank v3.5 cross-encoder |
| `langchain-core` | Required by langgraph |
| `openai` | AsyncOpenAI for query classification, summarisation, entity extraction, calculator, conflict detection, embeddings |
| `asyncpg` | Async PostgreSQL (Neon) for vector search, BM25, cache, query logs |
| `neo4j` | Async Neo4j driver for graph traversal |

### External Services

| Service | Purpose | Config |
|---------|---------|--------|
| OpenAI API | GPT-5.4 (classification, summarisation, conflict, calculator), text-embedding-3-large (query embeddings) | `OPENAI_API_KEY` |
| Cohere API | Rerank v3.5 cross-encoder | `COHERE_API_KEY` |
| Neon PostgreSQL | Vector search (HNSW), BM25 (tsvector), semantic cache, query logs | `DATABASE_URL` |
| Neo4j | Entity embedding search, graph traversal | `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` |

---

## 15. Error Handling

Every agent function returns a partial state dict on failure — the pipeline always continues. No agent ever raises an exception to the graph.

| Scenario | Behavior |
|----------|----------|
| `query()` top-level exception | Returns `QueryResult(skipped=True, error=...)`, never raises |
| Planner classification fails | Defaults to `ExecutionPlan(query_type=SIMPLE)`, logs error |
| Heuristic regex matches | Bypasses LLM classification entirely, no API cost |
| Vector search fails | Returns empty `retrieved_chunks`, error appended to state |
| BM25 query has no matches | Returns empty list, RRF fusion uses vector results only |
| Graph search fails | Returns empty chunks + paths, error appended to state |
| Entity embedding match < 2 results | Fallback: GPT-5.4 entity extraction + fuzzy Neo4j match |
| Neo4j unreachable | `graph_search` catches, returns `([], [])` |
| Cohere rerank fails | Falls back to original order truncated to `top_n` |
| `COHERE_API_KEY` empty | Skips Cohere, uses original order |
| Freshness boost parse error | Returns 1.0 (no boost), no error |
| Conflict detection fails | Returns empty conflicts list |
| Calculator fails | Returns `None` calculation result, summariser proceeds without it |
| Calculator: no numbers or unknown operation | Returns `None` |
| Summariser fails | Returns error message as answer |
| Cache check fails | Returns `None` (miss), proceeds to full pipeline |
| Cache store fails | Logged as warning, result already returned to caller |
| Query logging fails | Logged as warning via fire-and-forget task |
| Agent retry exhaustion | Each agent retries `AGENT_MAX_RETRIES` (2) times with exponential backoff (`500ms * 2^attempt`), then falls back |
| Iterative refinement: insufficient coverage | Re-runs vector agent with expanded queries (max 2 total passes) |
| Expanded query embedding fails | That query's results skipped, other queries still contribute |

### Retry Hierarchy

1. **Agent-level retries**: `_retry()` wrapper in `agents.py` — 2 attempts with 500ms/1000ms backoff
2. **Agent-level fallback**: Each agent catches all exceptions, returns partial state with error
3. **LangGraph state merge**: `operator.add` reducers accumulate chunks/errors from parallel agents
4. **Top-level catch**: `query()` wraps entire pipeline in try/except, returns `QueryResult(skipped=True)` on failure

---

## 16. Data Models

```python
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
    metadata: dict
    source: str = "vector"     # "vector" | "bm25" | "graph" | "fused"
    filename: str = ""
    version: int = 1
    ingested_at: str = ""

@dataclass
class GraphPath:
    entities: list[str]        # [seed_entity, related_entity]
    relationships: list[str]   # [rel_type_1, rel_type_2]
    source_chunks: list[str]   # chunk IDs resolved from path
    confidence: float = 1.0    # 1.0 for 1-hop, 0.5 for 2-hop

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
    metadata_filters: dict
    expanded_queries: list[str]

@dataclass
class QueryResult:
    answer: str
    chunks_used: list[RetrievedChunk]
    graph_paths: list[GraphPath]
    conflicts: list[ConflictResolution]
    query_type: QueryType = QueryType.SIMPLE
    cached: bool = False
    step_timings: dict
    skipped: bool = False
    error: str | None = None
```

---

## 17. LangGraph State

The `GraphState` TypedDict uses annotated reducers for parallel fan-out merging:

| Field | Type | Reducer | Purpose |
|-------|------|---------|---------|
| `original_query` | `str` | overwrite | User query |
| `plan` | `ExecutionPlan \| None` | overwrite | Planner output |
| `retrieved_chunks` | `list[RetrievedChunk]` | `operator.add` | Merged chunks from vector + graph |
| `graph_paths` | `list[GraphPath]` | `operator.add` | Graph traversal paths |
| `conflicts` | `list[ConflictResolution]` | overwrite | Detected conflicts |
| `calculation_result` | `str \| None` | overwrite | Calculator output |
| `final_answer` | `str` | overwrite | Summariser output |
| `errors` | `list[str]` | `operator.add` | Accumulated errors from all agents |
| `step_timings` | `dict` | `_merge_dicts` | Per-agent timing in seconds |
| `pass_count` | `int` | overwrite | Iterative refinement counter |
| `query_embedding_256` | `list[float] \| None` | overwrite | 256-dim embedding (cache + entity search) |
| `query_embedding_768` | `list[float] \| None` | overwrite | 768-dim embedding (hybrid HNSW search) |

The `operator.add` reducer on `retrieved_chunks` is what enables parallel vector+graph fan-out — both agents append their chunks to the same list.

---

## Evaluation Fixes (v2)

Post-evaluation fixes addressing 5 failure modes across 10 test queries (Q8 and Q10 scored 0/5).

### Query Classifier Improvements
- Rewritten system prompt with 8 few-shot examples and explicit FILTERED vs ANALYTICAL disambiguation
- Added `_ANALYTICAL_PATTERN` and `_CONFLICT_PATTERN` heuristic regexes
- Conflict language (proviso, contradiction, reconcile) forces `activate_graph=True` regardless of classification
- Router uses `plan.activate_graph` flag instead of checking `query_type`

### Cache Validity Gate
- `is_cacheable()` in `cache.py` blocks caching of empty results, errors, and graceful declines
- Decline phrase detection: "could not find", "no relevant information", "failed to generate", etc.
- Failed queries logged but never cached — prevents cascading failures

### HyDE (Hypothetical Document Embeddings)
- `hyde.py`: GPT-4o-mini generates a 3-5 sentence hypothetical legal document passage
- Activated for FILTERED, GRAPH, and ANALYTICAL queries (skipped for SIMPLE)
- HyDE embedding used for vector search; raw query kept for BM25 and reranking
- Falls back to raw query on failure — zero degradation

### Chunk Type Filtering
- Enricher extended with INDEX (table-of-contents) and DEFINITION (explanation clauses, provisos) types
- Retrieval SQL excludes HEADING and INDEX chunks via `COALESCE(metadata->>'chunk_type', 'PARAGRAPH') NOT IN (...)`
- Summariser context assembly: 3-tier priority (graph → DEFINITION → other by score)

### Coverage Threshold Calibration
- `_route_after_summariser` checks failure phrases, query-type-aware minimum chunks, and short-answer heuristic
- ANALYTICAL queries require 3+ chunks; GRAPH/ANALYTICAL answers under 200 chars trigger retry
- On retry: metadata_filters cleared, vector agent runs unfiltered
- Summariser deduplicates chunks across passes

### Enumeration-Aware Chunking
- Lettered sub-clauses (a),(b),(c) kept together, up to 1.5x max_tokens overflow
- Sibling sub-sections (shared parent path) get doubled overlap (24% vs 12%)
- Heading-only trailing chunks merged into previous chunk
