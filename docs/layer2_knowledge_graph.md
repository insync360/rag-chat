# Layer 2: Knowledge Graph & Graph RAG

## Overview

Layer 2 adds entity/relationship extraction during document ingestion and persists them to a Neo4j knowledge graph. This enables multi-hop reasoning, relationship discovery, and causal chain analysis that vector-only retrieval cannot do.

The graph extraction step is **non-blocking** — if Neo4j is down or extraction fails, the ingestion pipeline continues normally.

## Extended Pipeline

```
File → [1. Parse] → [2. Version Track] → [3. Chunk] → [4. Enrich] → [5. Graph Extract] → [6. Save]
```

Step 5 was added by Layer 2. Steps 1–4 and 6 are from Layer 1.

## Architecture

### Data Flow

```
Chunks
  │
  ▼
Extractor (GPT-4o, 1 call per chunk)
  │
  ▼
Deduplicator (deterministic, no LLM)
  │
  ▼
Neo4j Store (MERGE upsert, batched)
```

### Graph Model

**Nodes** — `(:Entity {name, type, source_document_id, source_chunk_index, confidence, properties, created_at, updated_at})`

- Uniqueness constraint on `(name, type)`
- Indexes on `source_document_id`, `type`, `name`

**Relationships** — Dynamic type labels (e.g. `[:EMPLOYS]`, `[:REPORTS_TO]`) with properties: `source_document_id`, `source_chunk_index`, `confidence`, `properties`, `created_at`, `updated_at`

### Entity Types (extensible, not enum-restricted)

Person, Organization, Product, Policy, Date, Location, Metric, Event, Technology, Role, Document, Regulation

### Relationship Types (extensible, not enum-restricted)

REPORTS_TO, SUPERSEDES, REFERENCES, CAUSED_BY, APPLIES_TO, DEFINED_IN, MEMBER_OF, LOCATED_IN, PRODUCES, EMPLOYS

## Files

### New Module: `app/graph/`

| File | Purpose |
|------|---------|
| `__init__.py` | `extract_and_store_graph()` — single integration point called by pipeline |
| `models.py` | Dataclasses: `Entity`, `Relationship`, `GraphExtractionResult` |
| `extractor.py` | GPT-4o entity + relationship extraction (1 call per chunk) |
| `dedup.py` | Deterministic entity/relationship deduplication |
| `neo4j_client.py` | Async Neo4j driver singleton (mirrors `app/database.py`) |
| `store.py` | MERGE entities and relationships into Neo4j with batching |
| `schema.py` | Idempotent Neo4j constraint and index setup |

### Modified Files

| File | Change |
|------|--------|
| `app/config.py` | Added Neo4j connection settings + graph extraction settings |
| `app/ingestion/pipeline.py` | Added `graph_extract` step, `entity_count`/`relationship_count` in `FileResult`, `graph_extract_ms` in `StepTimings` |
| `requirements.txt` | Added `neo4j` |
| `Test Cases/test_pipeline.py` | Patched `extract_and_store_graph` in existing test fixture |

### Test Files

| File | Tests | Coverage |
|------|-------|----------|
| `Test Cases/test_graph_extractor.py` | 6 | LLM extraction success, tiny chunk skip, failure fallback, invalid JSON retry, concurrency, freeform types |
| `Test Cases/test_graph_dedup.py` | 6 | Exact duplicate merge, different types kept, highest confidence wins, property merge, empty input, relationship dedup |
| `Test Cases/test_graph_store.py` | 5 | Entity MERGE, relationship MERGE, clear document graph, batching, Neo4j down |
| `Test Cases/test_graph_integration.py` | 4 | Full flow success, Neo4j failure returns skipped, disabled flag skips, pipeline continues on failure |

**Total: 21 new tests, all passing.**

## Configuration

Added to `app/config.py` (via `.env` or defaults):

```python
# Neo4j connection
NEO4J_URI       = "bolt://localhost:7687"   # env: NEO4J_URI
NEO4J_USER      = "neo4j"                   # env: NEO4J_USER
NEO4J_PASSWORD  = ""                        # env: NEO4J_PASSWORD

# Graph extraction
GRAPH_EXTRACTION_MODEL       = "gpt-4o"
GRAPH_EXTRACTION_CONCURRENCY = 3            # max parallel LLM calls
GRAPH_EXTRACTION_ENABLED     = True         # set False to skip entirely
GRAPH_BATCH_SIZE             = 100          # Neo4j UNWIND batch size
```

## Key Design Decisions

### Non-blocking graph step
The `extract_and_store_graph()` function catches all exceptions and returns `GraphExtractionResult(skipped=True)`. The pipeline wraps it in an additional try/except for defense-in-depth. Ingestion never fails due to graph issues.

### Single LLM call per chunk
One GPT-4o call extracts both entities and relationships (not two separate calls). This halves the API cost and latency compared to separate extraction.

### Extensible types (not enums)
Entity types and relationship types are plain strings. The LLM prompt suggests common types but accepts any. No code changes needed to support new types.

### Deterministic deduplication
Entity dedup uses name normalization (lowercase, strip whitespace, remove trailing punctuation) and groups by `(normalized_name, type)`. The highest-confidence entity wins, and properties are merged across duplicates. No LLM calls in the dedup step.

### Re-ingestion safety
When a document is re-ingested (new version), `clear_document_graph()` runs `DETACH DELETE` on all entities for that `source_document_id` before inserting new ones. This prevents stale graph data.

### MERGE-based upserts
Entities use `MERGE` on `(name, type)` — if an entity already exists from another document, its confidence is updated to the higher value. Relationships use `MERGE` per type with dynamic Cypher labels.

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Neo4j unreachable | `extract_and_store_graph` catches, returns `skipped=True`, pipeline continues |
| GPT-4o rate limit / failure | 3 retries with exponential backoff per chunk, then empty result for that chunk |
| Invalid JSON from GPT-4o | Retry up to 3 times, then skip chunk |
| `GRAPH_EXTRACTION_ENABLED=False` | Step skipped immediately, no LLM or Neo4j calls |
| Re-ingestion (new version) | `clear_document_graph` removes old nodes first |
| Relationship target not in entity list | `MATCH` finds no node, relationship silently skipped |
| Chunk < 10 tokens | Skipped (no LLM call) |

## Dependencies

- `neo4j` (Python async driver) — added to `requirements.txt`
- Neo4j database instance (local or remote)
- OpenAI API (GPT-4o for extraction)
