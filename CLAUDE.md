# RAG Chat — Production Agentic + Graph RAG

## Project Overview
Production-grade Agentic + Graph RAG system with 8 architectural layers. Accuracy-first design with hybrid retrieval, knowledge graph, multi-agent orchestration, and zero-hallucination guardrails.

## Architecture (8 Layers)
1. **Data Ingestion Pipeline** — Parse, structure-aware chunk, enrich documents ✅
2. **Knowledge Graph + Graph RAG** — Entity/relationship extraction, Neo4j graph, structural/relational embeddings ✅
3. **Database Layer** — Neon PostgreSQL (pgvector + tsvector) + Neo4j ✅
4. **Query Processing Engine** — Query classification, semantic cache, decomposition ✅
5. **Agentic Retrieval Engine** — LangGraph multi-agent (6 agents), hybrid retrieval, Cohere reranker ✅
6. **Validation & Guardrails** — Parallel faithfulness/relevance/coherence checks
7. **Evaluation & Observability** — RAGAS, LangSmith, online metrics
8. **Stress Testing & Security** — Prompt injection defense, ACL, rate limiting

## Technology Stack
- **Database**: Neon PostgreSQL v17 + pgvector (project: `icy-fire-24933610`, region: `aws-ap-southeast-1`)
- **Graph DB**: Neo4j (entities, relationships, communities)
- **LLM Provider**: OpenAI (GPT-4o extraction, GPT-4o-mini enrichment/community summaries, GPT-5.4 retrieval agents)
- **Embeddings**: OpenAI text-embedding-3-large (2000-dim chunks, 512-dim communities, 256-dim entities)
- **Graph Embeddings**: GraphSAGE 128-dim structural + TransE 128-dim relational (pure PyTorch, CPU)
- **Hybrid Embeddings**: 768-dim per chunk (512 text + 128 GraphSAGE + 128 TransE)
- **Document Parsing**: LlamaParse agentic_plus (Anthropic Sonnet 4.0)
- **Agent Orchestration**: LangGraph 6-agent StateGraph with parallel fan-out
- **Reranker**: Cohere Rerank v3.5 cross-encoder + freshness scoring

## Database
- **Neon project**: RAG_CHAT (`icy-fire-24933610`)
- **Connection**: stored in `.env` as `DATABASE_URL`
- **Extensions**: `vector` (pgvector), `pg_trgm`
- **Neon Tables**: documents, chunks, chunk_embeddings, entity_embeddings, relation_embeddings, community_summary_embeddings, ingestion_logs, semantic_cache, query_logs
- **Neo4j Nodes**: Entity (name, type, status, community_id), Community (summary)

## Implementation Phases
- **Phase 1** ✅: Foundation — ingestion, Neon schema, structure-aware chunking, LLM enrichment, chunk embeddings
- **Phase 2** ✅: Graph RAG — Neo4j entity extraction, 3-tier dedup, community detection, GraphSAGE, TransE, hybrid embeddings
- **Phase 3** ✅: Agentic Engine — LangGraph multi-agent, query classifier, semantic cache
- **Phase 4** ✅: Retrieval + Validation — hybrid retrieval (vector+BM25+graph), Cohere reranker, conflict resolution
- **Phase 5**: Observability — LangSmith, RAGAS, dashboards
- **Phase 6**: Security & Optimization — prompt injection, ACL, PII, rate limiting, A/B testing

## Project Structure
```
rag_chat/
├── app/
│   ├── config.py              # 140 lines — all settings (LlamaParse, OpenAI, Neo4j, graph, retrieval)
│   ├── database.py            # 22 lines — Neon async connection pool
│   ├── ingestion/             # Layer 1: parse, chunk, enrich, embed, pipeline
│   │   ├── parser.py          # 166 lines — LlamaParse agentic_plus → structured markdown
│   │   ├── version_tracker.py # 140 lines — content_hash dedup, version management
│   │   ├── chunker.py         # 380 lines — 3-phase structure-aware chunking
│   │   ├── enricher.py        # 153 lines — GPT-4o-mini metadata enrichment
│   │   ├── embedder.py        # 99 lines  — text-embedding-3-large (2000-dim) → pgvector
│   │   └── pipeline.py        # 342 lines — orchestrator with retry + post-batch graph steps
│   └── graph/                 # Layer 2: knowledge graph + graph embeddings
│       ├── __init__.py        # 150 lines — extract_and_store_graph() integration point
│       ├── models.py          # 88 lines  — 9 dataclasses (Entity, Relationship, results)
│       ├── extractor.py       # 134 lines — GPT-4o entity/relationship extraction
│       ├── coref.py           # 132 lines — fastcoref coreference resolution
│       ├── dedup.py           # 246 lines — 3-tier dedup (exact → fuzzy → embedding)
│       ├── schema.py          # 51 lines  — idempotent Neo4j constraints + indexes
│       ├── store.py           # 166 lines — MERGE entities/relationships into Neo4j
│       ├── community.py       # 299 lines — Leiden clustering + GPT-4o-mini summaries
│       ├── embeddings.py      # 424 lines — GraphSAGE structural embeddings (128-dim)
│       ├── transe.py          # 361 lines — TransE relation embeddings (128-dim)
│       ├── community_embeddings.py  # 233 lines — community summary embeddings (512-dim)
│       ├── hybrid_embeddings.py     # 254 lines — hybrid chunk-entity embeddings (768-dim)
│       └── neo4j_client.py    # 26 lines  — async Neo4j driver singleton
│   └── retrieval/             # Layer 5: agentic retrieval engine
│       ├── __init__.py        # 125 lines — query() entry point, cache check, query logging
│       ├── models.py          # 68 lines  — QueryType enum, dataclasses (RetrievedChunk, QueryResult, etc.)
│       ├── classifier.py      # 107 lines — GPT-5.4 query classification + heuristic pre-check
│       ├── cache.py           # 108 lines — semantic cache (256-dim embedding, cosine > 0.95)
│       ├── vector_search.py   # 242 lines — pgvector HNSW + BM25 tsvector + RRF fusion
│       ├── graph_search.py    # 309 lines — entity matching → Neo4j traversal → chunk retrieval
│       ├── reranker.py        # 80 lines  — Cohere Rerank v3.5 + freshness scoring
│       ├── summariser.py      # 95 lines  — GPT-5.4 chunk compression → final answer
│       ├── calculator.py      # 94 lines  — safe arithmetic (structured extraction, no eval)
│       ├── conflict.py        # 115 lines — contradiction detection + credibility resolution
│       ├── hyde.py            # 60 lines  — HyDE hypothetical document passage generation
│       ├── agents.py          # 255 lines — 6 LangGraph agent node functions with retry
│       └── graph_builder.py   # 139 lines — LangGraph StateGraph wiring + conditional routing
├── scripts/                   # Utility scripts
│   └── reset_and_ingest.py    # Clear all data + full end-to-end ingestion
├── migrations/                # 12 SQL migrations (001–012)
├── models/                    # Saved PyTorch weights (GraphSAGE + TransE)
├── docs/                      # Technical documentation (Layer 1 + Layer 2)
├── Test Cases/                # 19 test files (~3,850 lines)
├── requirements.txt
├── .env                       # DATABASE_URL, OPENAI_API_KEY, NEO4J_URI/USER/PASSWORD
└── .gitignore
```

## Key Design Decisions
- **Structure-aware chunking**: Heading boundaries, table integrity, 256-512 token range, 12% overlap
- **Metadata enrichment**: LLM-generated summaries, keywords, hypothetical questions (HyDE) per chunk
- **Idempotent ingestion**: content_hash deduplication, version tracking (deprecate, never delete)
- **3-tier entity dedup**: Exact → fuzzy (rapidfuzz 85%) → embedding (cosine 0.92) scoped by type
- **Incremental graph updates**: Only re-extract changed chunks on re-ingestion
- **Dual graph embeddings**: GraphSAGE (neighborhood structure) + TransE (typed relationships)
- **Hybrid chunk-entity embeddings**: 768-dim = 512 MRL-truncated text + 128 GraphSAGE + 128 TransE
- **Non-blocking graph pipeline**: All graph steps catch exceptions, pipeline never fails due to graph issues
- **No unnecessary dependencies**: GraphSAGE/TransE are pure PyTorch; hybrid embeddings use only `math` module
- **Query-adaptive routing**: SIMPLE→vector-only, GRAPH→parallel vector+graph, ANALYTICAL→all agents
- **RRF fusion**: Vector + BM25 results merged via Reciprocal Rank Fusion (k=60)
- **Never-raises agents**: Every agent catches exceptions, returns partial state, pipeline continues
- **Semantic cache**: 256-dim query embedding, cosine > 0.95, 24h TTL
- **Credibility hierarchy**: version > recency > content type > reranking score for conflict resolution
- **HyDE for non-SIMPLE queries**: GPT-4o-mini generates hypothetical legal document passage for vector search; raw query kept for BM25; Cohere reranks against raw query
- **Cache validity gate**: Only cache results with chunks, no errors, and no decline phrases
- **Chunk type filtering**: HEADING and INDEX chunks excluded from retrieval; DEFINITION chunks prioritized in summariser
- **Enumeration-aware chunking**: Lettered sub-clauses (a),(b),(c) kept together up to 1.5x max_tokens; sibling sub-sections get 2x overlap

## Documentation
- `docs/layer1_data_ingestion.md` — Layer 1 technical doc (pipeline, schema, config, error handling)
- `docs/layer2_knowledge_graph.md` — Layer 2 technical doc (graph extraction, embeddings, community detection)
- `docs/layer5_agentic_retrieval.md` — Layer 5 technical doc (retrieval engine, agents, routing, cache)

## Blueprint Reference
See `SAE-Steer_DSE316_COURSE_PROJECT_PROPOSAL.pdf` for the full 25-page architecture document.
