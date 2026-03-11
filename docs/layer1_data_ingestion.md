# Layer 1: Data Ingestion Pipeline — Technical Documentation

> **Status**: Production-ready (Phase 1 complete)
> **Last updated**: 2026-03-11

---

## 1. Overview

Layer 1 is a 7-stage document ingestion pipeline that transforms raw files into enriched, structure-aware chunks with vector embeddings stored in Neon PostgreSQL (pgvector). Every stage is idempotent and failure-resilient.

**Per-file pipeline stages:**

```
File → [1. Parse] → [2. Version Track] → [3. Chunk] → [4. Enrich] → [5. Graph Extract] → [6. Save] → [7. Embed]
```

| Stage | Module | Purpose |
|-------|--------|---------|
| Parse | `parser.py` | LlamaParse agentic_plus extracts structured markdown |
| Version Track | `version_tracker.py` | content_hash dedup, version management |
| Chunk | `chunker.py` | Structure-aware splitting (headings, tables, code) |
| Enrich | `enricher.py` | GPT-4o-mini generates summaries, keywords, HyDE questions |
| Graph Extract | `graph/` | Entity/relationship extraction to Neo4j (non-blocking) |
| Save | `chunker.py` + `pipeline.py` | Persist chunks + audit logs to Neon |
| Embed | `embedder.py` | text-embedding-3-large vectors → chunk_embeddings (non-blocking) |

**Post-batch stages** (run once after all files are processed, if any file completed):

```
[Community Detection] → [Community Summary Embeddings] → [GraphSAGE] → [TransE] → [Hybrid Chunk-Entity Embeddings]
```

| Stage | Module | Purpose |
|-------|--------|---------|
| Community Detection | `graph/community.py` | Leiden clustering + LLM summaries |
| Community Summary Embeddings | `graph/community_embeddings.py` | Embed community summaries → Neon pgvector (512-dim) |
| GraphSAGE | `graph/embeddings.py` | Structural entity embeddings → Neon pgvector (128-dim) |
| TransE | `graph/transe.py` | Relation entity embeddings → Neon pgvector (128-dim) |
| Hybrid Chunk-Entity Embeddings | `graph/hybrid_embeddings.py` | Combined text + graph embeddings → Neon pgvector (768-dim) |

All post-batch stages are non-blocking — failures are logged but never crash the pipeline. See `docs/layer2_knowledge_graph.md` for details.

---

## 2. File Map

| File | Lines | Purpose |
|------|-------|---------|
| `app/config.py` | 93 | All settings (LlamaParse, OpenAI, chunking, Neo4j, graph, embeddings, hybrid) |
| `app/database.py` | 22 | Async connection pool (asyncpg, min=2, max=10) |
| `app/ingestion/__init__.py` | 8 | Exports `ingest_file`, `ingest_files` |
| `app/ingestion/parser.py` | 166 | LlamaParser class — parse files to markdown |
| `app/ingestion/version_tracker.py` | 140 | VersionTracker class — dedup + versioning |
| `app/ingestion/chunker.py` | 380 | 3-phase chunker + `save_chunks` DB writer |
| `app/ingestion/enricher.py` | 153 | LLM enrichment with concurrency control |
| `app/ingestion/embedder.py` | 99 | Chunk embedding generation (text-embedding-3-large) |
| `app/ingestion/pipeline.py` | 342 | Orchestrator with retry loop + audit logging + post-batch graph embeddings |
| `migrations/001_documents.sql` | 28 | Documents table + indexes |
| `migrations/002_chunks.sql` | 23 | Chunks table + tsvector index |
| `migrations/003_metadata_gin_index.sql` | 3 | GIN index on chunks.metadata |
| `migrations/004_ingestion_logs.sql` | 16 | Ingestion audit log table |
| `migrations/005_chunk_content_hash.sql` | 2 | `content_hash` column + index on chunks (incremental change detection) |
| `migrations/007_chunk_embeddings.sql` | 14 | Chunk embeddings table + HNSW index |
| `migrations/010_hybrid_chunk_embeddings.sql` | 6 | `hybrid_embedding` vector(768) column + HNSW index on chunk_embeddings |

---

## 3. Module Deep Dives

### 3.1 Parser (`app/ingestion/parser.py`)

**Class**: `LlamaParser`

**LlamaParse configuration:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `parse_mode` | `parse_page_with_agent` | Agentic_plus tier for complex docs |
| `model` | `anthropic-sonnet-4.0` | LLM used by LlamaParse |
| `result_type` | `markdown` | Output format |
| `high_res_ocr` | `True` | Better OCR for scanned docs |
| `adaptive_long_table` | `True` | Handle tables spanning pages |
| `outlined_table_extraction` | `True` | Detect table borders |
| `output_tables_as_HTML` | `False` | Keep tables as markdown |

**Agentic prompt** instructs the parser to preserve:
- Heading hierarchy (H1-H6)
- Tables in structured markdown
- Code blocks with language tags
- Lists with nesting
- Image descriptions
- Page numbers and section references

**File validation** (`_validate_file`):
- File must exist
- Extension must be in `SUPPORTED_FILE_TYPES`: `.pdf`, `.docx`, `.pptx`, `.txt`, `.md`, `.html`
- Size must be <= `MAX_FILE_SIZE_MB` (50 MB)

**Content hashing**: SHA-256 computed in 8KB chunks, used for idempotency downstream.

**Retry logic**: Up to `PARSE_MAX_RETRIES` (3) attempts with exponential backoff (2^attempt seconds).

**Data types:**

```python
@dataclass
class ParsedPage:
    page_number: int      # 1-indexed
    markdown: str         # per-page markdown

@dataclass
class ParsedDocument:
    filename: str
    content_hash: str     # SHA-256 hex
    pages: list[ParsedPage]
    full_markdown: str    # all pages joined with \n\n
    page_count: int
    metadata: dict        # tier, model, processing_time_seconds
```

**`parse_bytes`**: Convenience method for upload endpoints — writes bytes to a temp file, parses, then cleans up.

---

### 3.2 Version Tracker (`app/ingestion/version_tracker.py`)

**Class**: `VersionTracker`

**Deduplication algorithm** (executed in `track()`):

```
1. Query: SELECT * FROM documents WHERE content_hash = $hash
   → If found: return existing record, is_new=False (cache hit)

2. BEGIN TRANSACTION
   a. SELECT MAX(version) FROM documents WHERE filename = $filename
   b. UPDATE documents SET status='deprecated' WHERE filename=$filename AND status='active'
   c. INSERT new row with version = max + 1, status='active'
   COMMIT

3. Return new record, is_new=True
```

**Versioning rules:**
- Same `content_hash` → skip (idempotent), return existing
- Same `filename`, different hash → deprecate all active versions, insert next version
- New `filename` → insert as version 1

**Orphan detection**: When `is_new=False` but `chunk_count=0` in the chunks table, the pipeline recognizes a prior failure between version tracking and chunk saving, and resumes from the chunk step.

**Additional methods:**
- `get_active(filename)` — returns the current active version (or None)
- `get_history(filename)` — returns all versions, newest first

**Data type:**

```python
@dataclass
class DocumentRecord:
    id: str               # UUID as string
    filename: str
    content_hash: str
    version: int
    status: str           # 'active' | 'deprecated'
    page_count: int
    ingested_at: datetime
    metadata: dict
```

---

### 3.3 Chunker (`app/ingestion/chunker.py`)

**3-phase algorithm:**

#### Phase 1 — Parse Blocks (`_parse_blocks`)

Splits raw markdown into typed blocks using regex:

| Block Type | Detection Pattern |
|------------|------------------|
| `HEADING` | `^(#{1,6})\s+(.+)$` |
| `CODE` | Lines between `` ``` `` fences |
| `TABLE` | Contiguous lines matching `^\|.*\|` |
| `PARAGRAPH` | Contiguous non-empty lines not matching above |

Blank lines are skipped (not accumulated).

#### Phase 2 — Group Blocks (`_group_blocks`)

Groups blocks into chunks respecting token limits:

- **Heading stack**: Maintains a stack of `(level, text)` tuples. When a new heading arrives, pops headings of equal or higher level. Produces `section_path` like `"HR Policy > California > Parental Leave"`.
- **Flush rule**: Current chunk is flushed when a heading arrives *and* the chunk has body content (non-heading blocks).
- **Tables/code**: Never split. If adding would exceed `max_tokens` and current chunk has content, flush first. Oversized table/code becomes a standalone chunk.
- **Long paragraphs**: Split at sentence boundaries (`(?<=[.!?])\s+(?=[A-Z])`) into pieces <= `max_tokens`.
- **Normal paragraphs**: Appended to current chunk; flush if would exceed `max_tokens`.

#### Phase 3 — Apply Overlap (`_apply_overlap`)

For each chunk after the first:
1. Compute `max_overlap_tokens = int(prev.token_count * CHUNK_OVERLAP_PERCENT)`
2. Extract trailing sentences from previous chunk up to that token budget
3. Prepend overlap text to current chunk (separated by `\n\n`)
4. Record `overlap_tokens` on the chunk

**Token counting**: Uses `tiktoken` with `cl100k_base` encoding.

**Configuration:**

| Setting | Value |
|---------|-------|
| `CHUNK_MIN_TOKENS` | 256 |
| `CHUNK_MAX_TOKENS` | 512 |
| `CHUNK_OVERLAP_PERCENT` | 0.12 (12%) |

**Data types:**

```python
class BlockType(Enum):
    HEADING, TABLE, CODE, PARAGRAPH

@dataclass
class Block:
    type: BlockType
    content: str
    heading_level: int    # 1-6 for HEADING, 0 otherwise
    heading_text: str     # raw text without #

@dataclass
class Chunk:
    document_id: str
    chunk_index: int      # 0-indexed, sequential
    content: str
    token_count: int
    section_path: str     # "HR Policy > California > Parental Leave"
    has_table: bool
    has_code: bool
    overlap_tokens: int
    metadata: dict        # populated by enricher
```

**`save_chunks`**: Persists chunks to Neon inside a transaction. Idempotent — deletes existing chunks for the document first (`DELETE FROM chunks WHERE document_id = $1`), then inserts all new chunks.

---

### 3.4 Enricher (`app/ingestion/enricher.py`)

**Function**: `enrich_chunks(chunks, doc_record) -> list[Chunk]`

**LLM prompt** — system message instructs GPT-4o-mini to return JSON with:
- `summary` — 2-3 sentence summary
- `keywords` — 5-10 important terms
- `hypothetical_questions` — 3-5 questions the chunk can answer (HyDE)
- `entities` — `{people, organizations, dates, money}` arrays

**Concurrency**: Bounded by `asyncio.Semaphore(ENRICHMENT_CONCURRENCY)` (5 concurrent requests). All chunks processed via `asyncio.gather`.

**API call config:**
- Model: `gpt-4o-mini`
- `response_format`: `{"type": "json_object"}`
- `temperature`: 0
- `max_tokens`: 1024

**Chunk type classification** (`_classify_chunk_type`) — deterministic, no LLM:

| Priority | Condition | Type |
|----------|-----------|------|
| 1 | `has_table=True` | `TABLE` |
| 2 | `has_code=True` | `CODE` |
| 3 | Starts with `#` and < 20 tokens | `HEADING` |
| 4 | > 50% of lines match list pattern | `LIST` |
| 5 | Default | `PARAGRAPH` |

**Freshness score**: `max(0, 1 - (days_since_ingestion / 365))` — linearly decays from 1.0 to 0.0 over one year.

**Fallback strategy:**
- Tiny chunks (< 10 tokens): skip LLM entirely, only set base metadata
- API failures: retry 3 times with exponential backoff (1s, 2s, 4s)
- After 3 failures: log warning, use empty defaults for LLM fields (`summary=""`, `keywords=[]`, etc.)
- **Never raises** — failures produce fallback metadata, pipeline continues

**Final metadata per chunk:**

```json
{
  "summary": "...",
  "keywords": ["..."],
  "hypothetical_questions": ["..."],
  "entities": {"people": [], "organizations": [], "dates": [], "money": []},
  "chunk_type": "PARAGRAPH",
  "freshness_score": 0.97,
  "document_id": "uuid",
  "section_path": "HR Policy > California",
  "version": 1,
  "content_hash": "sha256...",
  "ingested_at": "ISO8601",
  "enriched_at": "ISO8601"
}
```

---

### 3.5 Embedder (`app/ingestion/embedder.py`)

**Functions:**
- `_build_embedding_text(chunk) -> str` — combines content + enriched metadata into embedding input
- `embed_chunks(chunk_ids, chunks) -> EmbeddingResult` — generates embeddings and upserts to DB

Generates `text-embedding-3-large` (2000-dim) embeddings from **content + enriched metadata** and upserts to the `chunk_embeddings` table.

**Embedding text format** (built by `_build_embedding_text`):

```
Section: HR Policy > California > Parental Leave
Summary: California employees are entitled to 12 weeks of parental leave...
Keywords: parental leave, California, FMLA, maternity, paternity
Questions: What is the parental leave policy in California? How many weeks of leave?

## Parental Leave
California employees are entitled to...
```

**Fields included** (semantically useful for retrieval):
- `section_path` — document hierarchy context
- `summary` — condensed semantic representation
- `keywords` — important terms joined with ", "
- `hypothetical_questions` — query-space coverage (embeds what queries this chunk answers)
- Raw `content` — the actual chunk text

**Fields excluded** (structural/temporal, not semantic):
- `chunk_type`, `freshness_score`, `document_id`, `version`, `content_hash`, `ingested_at`, `enriched_at`, `entities` (already in content)

**Fallback**: Chunks with empty/missing metadata (e.g. tiny chunks that skipped LLM enrichment) embed with content only.

**Design:**
- **Input**: `Chunk` objects (accesses `.content`, `.section_path`, and `.metadata`)
- **Batching**: Up to 2048 texts per OpenAI API call (typical document = single call)
- **Storage**: `ON CONFLICT (chunk_id) DO UPDATE` for idempotent upserts
- **Never raises**: All errors caught internally, returns `EmbeddingResult(skipped=True)` on failure

**Configuration:**

| Setting | Value | Purpose |
|---------|-------|---------|
| `CHUNK_EMBEDDING_MODEL` | `text-embedding-3-large` | OpenAI embedding model |
| `CHUNK_EMBEDDING_DIMENSIONS` | `2000` | Full resolution for 256-512 token chunks |
| `CHUNK_EMBEDDING_BATCH_SIZE` | `2048` | Max texts per API call |

**Return type:**

```python
@dataclass
class EmbeddingResult:
    total: int           # chunks received
    embedded: int        # chunks successfully embedded
    skipped: bool        # True if embedding failed
    error: str | None    # error message if failed
```

---

### 3.6 Pipeline Orchestrator (`app/ingestion/pipeline.py`)

**Entry points:**
- `ingest_file(file_path) -> FileResult` — single file convenience wrapper
- `ingest_files(file_paths) -> PipelineResult` — sequential batch processing

**Per-file retry loop** (`_ingest_one`):

```
For attempt 1..3:
    Create audit log row (status=pending)
    Try:
        1. Parse file → ParsedDocument
        2. Version track → (DocumentRecord, is_new)
           - If is_new=False AND chunks exist → SKIPPED (true duplicate)
           - If is_new=False AND chunks=0 → orphan, continue
        3. Chunk document → list[Chunk]
           3b. Change detection for incremental graph updates
        4. Enrich chunks → list[Chunk] (with metadata)
        5. Graph extract → entity/relationship extraction (non-blocking)
        6. Save chunks → list[UUID]
        7. Embed chunks → chunk_embeddings (non-blocking)
        Update log → completed
        Return FileResult(COMPLETED)
    Except:
        Update log → failed (with step + error)
        If more attempts: sleep(2^attempt), retry
Return FileResult(FAILED)
```

**Post-batch processing** (`ingest_files`, after all files):

```
If any file completed:
    1. Community Detection (Leiden clustering + LLM summaries)
    2. Community Summary Embeddings (OpenAI → pgvector 512-dim)
    3. GraphSAGE (structural entity embeddings → pgvector 128-dim)
    4. TransE (relation entity embeddings → pgvector 128-dim)
    5. Hybrid Chunk-Entity Embeddings (text + graph → pgvector 768-dim)
```

Each post-batch step is wrapped in its own try/except — failures are logged but never propagate.

**Audit logging**: Every attempt creates a row in `ingestion_logs` with:
- filename, attempt number, status, last step reached
- document_id (if known), chunk_count (if known)
- error_message (if failed)
- step_timings (JSON: `{parse_ms, version_track_ms, chunk_ms, enrich_ms, graph_extract_ms, save_ms, embed_ms}`)
- started_at, completed_at timestamps

**Step timing**: Uses `time.perf_counter()` for high-resolution timing of each stage.

**Data types:**

```python
class PipelineStatus(str, Enum):
    PENDING, PROCESSING, COMPLETED, FAILED, SKIPPED

@dataclass
class StepTimings:
    parse_ms, version_track_ms, chunk_ms, enrich_ms, graph_extract_ms, save_ms, embed_ms: float

@dataclass
class FileResult:
    filename: str
    status: PipelineStatus
    document_id: str | None
    version: int | None
    chunk_count: int | None
    error: str | None
    step: str | None           # last step reached
    timings: StepTimings
    log_id: str | None         # ingestion_logs UUID

@dataclass
class PipelineResult:
    total, completed, skipped, failed: int
    files: list[FileResult]
    total_duration_ms: float
```

---

## 4. Database Schema

### 4.1 `documents` (migration 001)

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `UUID` | PK, `gen_random_uuid()` | Document UUID |
| `filename` | `TEXT` | NOT NULL | Original file name |
| `content_hash` | `TEXT` | NOT NULL, UNIQUE | SHA-256 for dedup |
| `version` | `INTEGER` | NOT NULL, default 1 | Version number |
| `status` | `TEXT` | NOT NULL, CHECK `('active','deprecated')` | Lifecycle state |
| `page_count` | `INTEGER` | | Number of pages |
| `full_markdown` | `TEXT` | | Complete parsed markdown |
| `ingested_at` | `TIMESTAMPTZ` | NOT NULL, default `now()` | Ingestion timestamp |
| `deprecated_at` | `TIMESTAMPTZ` | | When deprecated |
| `metadata` | `JSONB` | default `'{}'` | Parser metadata |

**Indexes:**
- `idx_documents_content_hash` — UNIQUE on `content_hash` (idempotency check)
- `idx_documents_filename_status` — `(filename, status)` (find active version)
- `idx_documents_filename_version` — `(filename, version DESC)` (find latest)

### 4.2 `chunks` (migration 002)

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `UUID` | PK, `gen_random_uuid()` | Chunk UUID |
| `document_id` | `UUID` | NOT NULL, FK → documents(id) ON DELETE CASCADE | Parent document |
| `chunk_index` | `INTEGER` | NOT NULL | 0-indexed position |
| `content` | `TEXT` | NOT NULL | Chunk text |
| `token_count` | `INTEGER` | NOT NULL | cl100k_base token count |
| `section_path` | `TEXT` | NOT NULL, default `''` | Heading breadcrumb |
| `has_table` | `BOOLEAN` | NOT NULL, default FALSE | Contains table |
| `has_code` | `BOOLEAN` | NOT NULL, default FALSE | Contains code block |
| `overlap_tokens` | `INTEGER` | NOT NULL, default 0 | Overlap from previous chunk |
| `metadata` | `JSONB` | default `'{}'` | Enrichment metadata |
| `created_at` | `TIMESTAMPTZ` | NOT NULL, default `now()` | Creation time |

**Indexes:**
- `idx_chunks_document_index` — UNIQUE on `(document_id, chunk_index)`
- `idx_chunks_content_tsvector` — GIN on `to_tsvector('english', content)` (for BM25 keyword search)
- `idx_chunks_metadata_gin` — GIN on `metadata` (migration 003, for metadata queries)

### 4.3 `chunk_embeddings` (migrations 007, 010)

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `UUID` | PK, `gen_random_uuid()` | Embedding UUID |
| `chunk_id` | `UUID` | NOT NULL, FK → chunks(id) ON DELETE CASCADE | Parent chunk |
| `embedding` | `vector(2000)` | NOT NULL | text-embedding-3-large vector |
| `hybrid_embedding` | `vector(768)` | | Hybrid chunk-entity embedding (512 text + 128 GraphSAGE + 128 TransE) |
| `model` | `TEXT` | NOT NULL, default `'text-embedding-3-large'` | Model used |
| `created_at` | `TIMESTAMPTZ` | NOT NULL, default `now()` | Creation time |

**Indexes:**
- `idx_chunk_emb_chunk_id` — UNIQUE on `chunk_id` (one embedding per chunk, upsert via ON CONFLICT)
- `idx_chunk_emb_vector` — HNSW on `embedding vector_cosine_ops` (m=16, ef_construction=200)
- `idx_chunk_emb_hybrid` — HNSW on `hybrid_embedding vector_cosine_ops` (m=16, ef_construction=64)

**CASCADE behavior:** When chunks are deleted during re-ingestion (`DELETE FROM chunks WHERE document_id = $1`), embeddings are automatically deleted.

### 4.4 `ingestion_logs` (migration 004)

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `UUID` | PK, `gen_random_uuid()` | Log UUID |
| `filename` | `TEXT` | NOT NULL | File being ingested |
| `status` | `TEXT` | NOT NULL, CHECK `('pending','processing','completed','failed','skipped')` | Pipeline status |
| `document_id` | `UUID` | FK → documents(id) | Linked document |
| `attempt` | `INTEGER` | NOT NULL, default 1 | Retry attempt number |
| `step` | `TEXT` | | Last pipeline step reached |
| `chunk_count` | `INTEGER` | | Number of chunks produced |
| `error_message` | `TEXT` | | Error details if failed |
| `step_timings` | `JSONB` | default `'{}'` | Per-step timing in ms |
| `started_at` | `TIMESTAMPTZ` | NOT NULL, default `now()` | Start time |
| `completed_at` | `TIMESTAMPTZ` | | Completion time |

**Indexes:**
- `idx_ingestion_logs_status` — on `status`

---

## 5. Configuration Reference

All settings from `app/config.py`:

| Setting | Value | Source |
|---------|-------|--------|
| `DATABASE_URL` | (env) | `.env` |
| `LLAMA_CLOUD_API_KEY` | (env) | `.env` |
| `OPENAI_API_KEY` | (env) | `.env` |
| `LLAMAPARSE_MODE` | `parse_page_with_agent` | Hardcoded |
| `LLAMAPARSE_MODEL` | `anthropic-sonnet-4.0` | Hardcoded |
| `LLAMAPARSE_RESULT_TYPE` | `markdown` | Hardcoded |
| `ENRICHMENT_MODEL` | `gpt-4o-mini` | Hardcoded |
| `ENRICHMENT_CONCURRENCY` | `5` | Hardcoded |
| `SUPPORTED_FILE_TYPES` | `.pdf, .docx, .pptx, .txt, .md, .html` | Hardcoded |
| `MAX_FILE_SIZE_MB` | `50` | Hardcoded |
| `PARSE_TIMEOUT_SECONDS` | `300` (5 min) | Hardcoded |
| `PARSE_MAX_RETRIES` | `3` | Hardcoded |
| `CHUNK_MIN_TOKENS` | `256` | Hardcoded |
| `CHUNK_MAX_TOKENS` | `512` | Hardcoded |
| `CHUNK_OVERLAP_PERCENT` | `0.12` (12%) | Hardcoded |
| `CHUNK_EMBEDDING_MODEL` | `text-embedding-3-large` | Hardcoded |
| `CHUNK_EMBEDDING_DIMENSIONS` | `2000` | Hardcoded |
| `CHUNK_EMBEDDING_BATCH_SIZE` | `2048` | Hardcoded |

**Database pool** (`app/database.py`): `min_size=2`, `max_size=10`

**Pipeline constant** (`app/ingestion/pipeline.py`): `MAX_RETRIES = 3`

---

## 6. Data Flow Diagram

```
                        ┌─────────────────────────────────────────────────────────┐
                        │              PIPELINE ORCHESTRATOR (pipeline.py)        │
                        │         retry loop (3 attempts, 2^n backoff)            │
                        │         audit log per attempt (ingestion_logs)          │
                        └──────────────────────┬──────────────────────────────────┘
                                               │
     ┌─────────────────────────────────────────┼─────────────────────────────────────────┐
     │                                         │                                         │
     ▼                                         ▼                                         ▼
 ┌───────────┐    ParsedDocument    ┌──────────────────┐   (record,is_new)   ┌───────────────┐
 │  PARSER   │ ──────────────────►  │  VERSION TRACKER │ ──────────────────► │    CHUNKER    │
 │           │                      │                  │                     │               │
 │ LlamaParse│                      │ content_hash     │                     │ 3 phases:     │
 │ agentic+  │                      │ dedup check      │                     │ parse blocks  │
 │           │                      │ version mgmt     │                     │ group chunks  │
 │ Validate  │                      │ deprecate old    │                     │ add overlap   │
 │ Hash file │                      │ insert new       │                     │               │
 └───────────┘                      └──────────────────┘                     └───────┬───────┘
      │                                    │                                         │
      │                                    │                                   list[Chunk]
      │                                    ▼                                         │
      │                             ┌──────────────┐                                 ▼
      │                             │  NEON        │                        ┌─────────────────┐
      │                             │  PostgreSQL  │◄───── save_chunks ──── │    ENRICHER     │
      │                             │              │                        │                 │
      │                             │  documents   │                        │ GPT-4o-mini     │
      │                             │  chunks      │◄── embed_chunks ──┐    │ 5 concurrent    │
      │                             │  chunk_      │                   │    │ summary,        │
      │                             │  embeddings  │   ┌───────────┐   │    │ keywords, HyDE, │
      │                             │  entity_     │   │  EMBEDDER │───┘    │ entities, type  │
      │                             │  embeddings  │   │ text-emb- │        └─────────────────┘
      │                             │  relation_   │   │ 3-large   │
      │                             │  embeddings  │   │ 2000-dim  │        ┌─────────────────┐
      │                             │  community_  │   └───────────┘        │  GRAPH EXTRACT  │
      │                             │  summary_    │                        │                 │
      │                             │  embeddings  │◄── graph store ─────── │ GPT-4o per chunk│
      │                             │  ingestion_  │                        │ coref + dedup   │
      │                             │    logs      │                        │ → Neo4j MERGE   │
      │                             └──────────────┘                        └─────────────────┘
      │                                    ▲                                         │
      │                                    │                                         ▼
      │                             ┌──────────────┐                        ┌─────────────────┐
      │                             │  NEO4J       │◄─── community IDs ──── │  POST-BATCH     │
      │                             │              │                        │                 │
      │                             │  Entity      │    ┌───────────────┐   │ 1. Community    │
      │                             │  Community   │◄───│  GraphSAGE +  │   │ 2. Comm. Embeds │
      │                             │  nodes       │    │  TransE       │   │ 3. GraphSAGE    │
      │                             └──────────────┘    │  Hybrid       │──►│ 4. TransE       │
      │                                                 └───────────────┘   │ 5. Hybrid       │
      │                             ┌──────────────┐                        └─────────────────┘
      └────────────────────────────►│  LlamaParse  │
                                    │  Cloud API   │
                                    └──────────────┘
```

**Duplicate detection fast path:**

```
content_hash match found?
  ├─ YES → chunks exist?
  │         ├─ YES → SKIP (return existing, no further work)
  │         └─ NO  → ORPHAN (resume from chunk step)
  └─ NO  → new/updated → deprecate old versions → full pipeline
```

---

## 7. Error Handling Matrix

| Error | Where | Cause | Handling |
|-------|-------|-------|----------|
| `ParsingError("LLAMA_CLOUD_API_KEY is not set")` | `LlamaParser.__init__` | Missing env var | Raised immediately, pipeline cannot start |
| `ParsingError("File not found")` | `_validate_file` | File doesn't exist | Raised, caught by pipeline retry loop |
| `ParsingError("Unsupported file type")` | `_validate_file` | Extension not in allowed set | Raised, caught by pipeline retry loop |
| `ParsingError("File too large")` | `_validate_file` | File > 50 MB | Raised, caught by pipeline retry loop |
| `ParsingError("Parsing failed after N attempts")` | `LlamaParser.parse` | LlamaParse API errors | 3 retries with 2^n backoff inside parser, then raised |
| DB connection error | `VersionTracker.track` | Neon unavailable | Caught by pipeline retry loop |
| Transaction conflict | `VersionTracker.track` | Concurrent ingestion of same file | Transaction isolation handles it |
| LLM API failure | `_enrich_single` | OpenAI API error or bad JSON | 3 retries with backoff, then fallback to empty metadata |
| JSON decode error | `_enrich_single` | LLM returns invalid JSON | Same as LLM API failure |
| DB write error | `save_chunks` | Insert failure | Caught by pipeline retry loop |
| Embedding API failure | `embed_chunks` | OpenAI API error | Caught internally, returns `EmbeddingResult(skipped=True)`, pipeline continues |
| Any exception in pipeline step | `_ingest_one` | Unexpected error | Logged to `ingestion_logs`, retry with backoff (3 attempts) |

**Retry hierarchy:**
1. Parser-level retries: 3 attempts for LlamaParse API calls
2. Enricher-level retries: 3 attempts per chunk for OpenAI API calls
3. Pipeline-level retries: 3 attempts for the entire file (wrapping all stages)

---

## 8. Test Coverage

### Test files

| File | Lines | Tests | Framework | Type |
|------|-------|-------|-----------|------|
| `Test Cases/test_parser.py` | 34 | 1 | Script (manual) | Integration — parses real file via LlamaParse |
| `Test Cases/test_chunker.py` | 200 | 3 | Script (manual) | Unit + Integration — blocks, chunking, DB round-trip |
| `Test Cases/test_enricher.py` | 218 | 12 | pytest | Unit + Integration — classification, freshness, enrichment |
| `Test Cases/test_embedder.py` | 228 | 10 | pytest + pytest-asyncio | Unit — enriched text building, batching, upsert, error handling, pipeline integration |
| `Test Cases/test_pipeline.py` | 253 | 9 | pytest + pytest-asyncio | Unit — all pipeline paths with mocked deps |
| `Test Cases/test_pipeline_e2e.py` | 130 | 1 | Script (manual) | E2E — full pipeline against real APIs + Neon DB |

### Test scenarios

**test_parser.py:**
- Parse a file (default: blueprint PDF) and print results

**test_chunker.py:**
- `test_parse_blocks` — detects all 4 block types, correct heading levels, table content, code content
- `test_chunk_document` — sequential indices, table integrity, code integrity, section paths, overlap presence
- `test_db_roundtrip` — save to Neon, query back, verify all fields match, idempotent re-save

**test_enricher.py:**
- `TestClassifyChunkType` (7 tests): TABLE, CODE, HEADING, LIST (bullet + numbered), PARAGRAPH, table-over-code priority
- `TestComputeFreshnessScore` (5 tests): just now (~1.0), half year (~0.5), one year (0.0), older (0.0), naive datetime
- `TestEnrichChunks` (5 tests): all fields populated, tiny chunk skips LLM, empty list, multiple chunks, API failure fallback

**test_embedder.py:**
- `test_build_embedding_text_includes_all_metadata` — section_path, summary, keywords, questions all present
- `test_build_embedding_text_empty_metadata_falls_back_to_content` — empty metadata → content only
- `test_build_embedding_text_no_metadata_attr` — `metadata=None` → content only (graceful fallback)
- `test_build_embedding_text_partial_metadata` — only summary present, no keywords/questions in output
- `test_embed_chunks_success` — mocked OpenAI + asyncpg, correct count returned
- `test_embed_chunks_sends_enriched_text` — verifies enriched text (not raw content) sent to OpenAI
- `test_embed_chunks_empty_input` — empty input → `EmbeddingResult(total=0, embedded=0)`
- `test_embed_chunks_openai_failure` — API failure → `skipped=True`, pipeline unaffected
- `test_pipeline_calls_embed_with_chunks` — pipeline calls `embed_chunks` with chunk objects
- `test_pipeline_continues_if_embed_raises` — pipeline continues if embed raises

**test_pipeline.py:**
- `test_success_full_pipeline` — all 7 steps called, COMPLETED
- `test_skipped_duplicate` — hash match + chunks exist → SKIPPED, no chunk/enrich/save calls
- `test_orphan_resumes` — hash match + no chunks → COMPLETED via chunk/enrich/save
- `test_parse_failure_retries_then_succeeds` — 2 failures then success, 3 parse calls, 3 log rows
- `test_all_retries_exhausted` — always fails → FAILED with error message
- `test_batch_isolation` — 3 files (1 complete, 1 skip, 1 fail) processed independently
- `test_empty_list` — empty input → zero counts, empty files list
- `test_step_timings` — zero-valued timings excluded from dict
- `test_pipeline_result_counts` — completed + skipped + failed = total

**test_pipeline_e2e.py:**
- Full 6-step E2E: parse → version track → chunk → enrich → save → verify round-trip
- Validates all metadata fields present in DB (lineage + LLM fields)

---

## 9. Performance Profile

### Latency estimates (per document)

| Stage | Estimated Latency | Notes |
|-------|-------------------|-------|
| Parse | 30-120s | LlamaParse API call, depends on doc size/complexity |
| Version Track | 5-50ms | 1-3 DB queries (hash check + transaction) |
| Chunk | 10-100ms | Local CPU, depends on markdown length |
| Enrich | 5-30s | N chunks x 1 API call each, bounded by semaphore(5) |
| Save | 20-200ms | N sequential inserts in one transaction |
| Embed | 1-5s | 1 API call for typical doc (< 2048 chunks), single batch |
| **Total** | **~36-155s** | **Dominated by parse + enrich** |

### API call counts (per document, ingestion only)

| Service | Calls | Details |
|---------|-------|---------|
| LlamaParse | 1 | One parse call per file (up to 3 retries) |
| OpenAI (enrichment) | N | One per chunk (up to 3 retries each), skips tiny chunks |
| OpenAI (embedding) | ceil(N/2048) | Batched embedding calls, typically 1 per document |
| OpenAI (graph extraction) | N | One GPT-4o call per chunk for entity/relationship extraction |
| Neon PostgreSQL | 3 + 2N | Hash check, version ops, N chunk inserts + N embedding upserts + log writes |
| Neo4j | 2 + batch | Schema ensure, entity/relationship MERGE in batches of 100 |

**Post-batch steps** (run once per batch, not per document): Community detection (Neo4j read + igraph + LLM summaries), community summary embeddings (1 OpenAI call), GraphSAGE (1 OpenAI call for features + CPU training), TransE (CPU training only), hybrid chunk-entity embeddings (Neon reads + writes only, no API calls).

### Bottlenecks

1. **LlamaParse latency**: Largest single cost. Cloud API, no local control.
2. **OpenAI enrichment**: Scales linearly with chunk count. Bounded to 5 concurrent.
3. **Sequential batch processing**: `ingest_files` processes files one at a time.

---

## 10. Design Patterns

### Idempotency
- **Document level**: `content_hash` (SHA-256) uniquely identifies content. Re-ingesting the same file is a no-op.
- **Chunk level**: `save_chunks` deletes all existing chunks for a document before inserting. Same input produces same output.
- **DB level**: UNIQUE constraint on `content_hash`; UNIQUE constraint on `(document_id, chunk_index)`.

### Deprecate, Never Delete
- Old document versions get `status='deprecated'` and `deprecated_at` timestamp.
- Active content is always the latest version with `status='active'`.
- Historical versions remain queryable for audit.

### Exponential Backoff
- Parser: `sleep(2^attempt)` seconds between retries (2s, 4s).
- Enricher: `sleep(2^attempt)` seconds between retries (1s, 2s).
- Pipeline: `sleep(2^attempt)` seconds between full-pipeline retries (2s, 4s).

### Graceful Degradation
- Enricher never raises. LLM failures produce chunks with base metadata only.
- Pipeline catches all exceptions per file; batch continues even if one file fails.

### Structure-Aware Chunking
- Tables and code blocks are never split, ensuring structural integrity.
- Heading stack tracks section hierarchy for breadcrumb paths.
- Sentence-boundary splitting for long paragraphs preserves semantic coherence.

### Orphan Detection
- If a document record exists (same hash) but has no chunks, the pipeline infers a prior crash between version tracking and chunk saving, and resumes from the chunk step.

### Audit Trail
- Every pipeline attempt creates an `ingestion_logs` row with attempt number, status, last step reached, error message, and per-step timings in milliseconds.

---

## 11. Known Limitations & Future Work

### Current Limitations
- **Sequential file processing**: `ingest_files` processes one file at a time. No parallel file ingestion.
- **No streaming/progress callbacks**: No way to monitor pipeline progress from the API layer.
- **`PARSE_TIMEOUT_SECONDS` not enforced**: The 300s timeout is defined but not applied as an actual timeout on the parse call.
- **Enricher concurrency is global**: The semaphore bounds concurrent OpenAI calls per `enrich_chunks` invocation, not globally across multiple files.
- **No chunk deduplication across documents**: If two different documents contain identical text, chunks are stored separately.
- **Hybrid entity-chunk mapping V1**: Multi-document entities only enrich their first extraction chunk. Chunks in later documents won't get entity graph context from pre-existing entities. See Layer 2 docs for details.

### Completed (Phases 1–2)
- **Layer 1: Data Ingestion** — Parse, chunk, enrich, embed (this document)
- **Layer 2: Knowledge Graph** — Neo4j entity graph, community detection, GraphSAGE, TransE, hybrid embeddings (see `docs/layer2_knowledge_graph.md`)

### Future Work (Phases 3–6)
- **Layer 3: Query Processing Engine** — Query classification, semantic cache, query decomposition
- **Layer 4: Agentic Retrieval Engine** — LangGraph multi-agent (6 agents), hybrid vector+BM25+graph retrieval, Cohere Rerank 3
- **Layer 5: Validation & Guardrails** — Parallel faithfulness/relevance/coherence checks (GPT-4o-mini)
- **Layer 6: Evaluation & Observability** — RAGAS, LangSmith, online metrics, dashboards
- **Layer 7: Stress Testing & Security** — Prompt injection defense, ACL, PII, rate limiting
- **Parallel file ingestion**: Use `asyncio.gather` or a task queue for concurrent file processing
- **Webhook/SSE progress**: Stream pipeline status to API consumers
- **Cost tracking**: Log LlamaParse and OpenAI API costs per ingestion

---

## 12. Dependencies

### Python Libraries

| Package | Purpose |
|---------|---------|
| `llama-cloud-services` | LlamaParse document parsing API client |
| `python-dotenv` | Load `.env` file into environment |
| `asyncpg` | Async PostgreSQL driver for Neon |
| `openai` | AsyncOpenAI client for GPT-4o/GPT-4o-mini/text-embedding-3-large |
| `tiktoken` | Token counting (cl100k_base encoding) |
| `neo4j` | Async Neo4j driver (knowledge graph) |
| `fastcoref` | Coreference resolution (FCoref model) |
| `transformers` | Required by fastcoref |
| `rapidfuzz` | Fuzzy string matching for entity deduplication |
| `leidenalg` | Leiden community detection algorithm |
| `igraph` | Graph data structure for Leiden |
| `torch` | GraphSAGE + TransE model training (CPU-only) |

### External Services

| Service | Purpose | Config |
|---------|---------|--------|
| LlamaParse Cloud | Document parsing (agentic_plus tier) | `LLAMA_CLOUD_API_KEY` |
| OpenAI API | GPT-4o extraction, GPT-4o-mini enrichment/summaries, text-embedding-3-large embeddings | `OPENAI_API_KEY` |
| Neon PostgreSQL | Document, chunk, embedding storage (pgvector) | `DATABASE_URL` |
| Neo4j | Knowledge graph storage (entities, relationships, communities) | `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` |

### Dev/Test Dependencies (not in requirements.txt)

| Package | Purpose |
|---------|---------|
| `pytest` | Test runner |
| `pytest-asyncio` | Async test support |
