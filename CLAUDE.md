# RAG Chat — Production Agentic + Graph RAG

## Project Overview
Production-grade Agentic + Graph RAG system with 8 architectural layers. Accuracy-first design with hybrid retrieval, knowledge graph, multi-agent orchestration, and zero-hallucination guardrails.

## Architecture (8 Layers)
1. **Data Ingestion Pipeline** — Parse, structure-aware chunk, enrich documents ✅
2. **Knowledge Graph + Graph RAG** — Entity/relationship extraction, Neo4j graph, structural/relational embeddings ✅
3. **Database Layer** — Neon PostgreSQL (pgvector + tsvector) + Neo4j ✅
4. **Query Processing Engine** — Query classification, semantic cache, decomposition
5. **Agentic Retrieval Engine** — LangGraph multi-agent (6 agents)
6. **Validation & Guardrails** — Parallel faithfulness/relevance/coherence checks
7. **Evaluation & Observability** — RAGAS, LangSmith, online metrics
8. **Stress Testing & Security** — Prompt injection defense, ACL, rate limiting

## Technology Stack
- **Database**: Neon PostgreSQL v17 + pgvector (project: `icy-fire-24933610`, region: `aws-ap-southeast-1`)
- **Graph DB**: Neo4j (entities, relationships, communities)
- **LLM Provider**: OpenAI (GPT-4o extraction, GPT-4o-mini enrichment/summaries/validation)
- **Embeddings**: OpenAI text-embedding-3-large (2000-dim chunks, 512-dim communities, 256-dim entities)
- **Graph Embeddings**: GraphSAGE 128-dim structural + TransE 128-dim relational (pure PyTorch, CPU)
- **Hybrid Embeddings**: 768-dim per chunk (512 text + 128 GraphSAGE + 128 TransE)
- **Document Parsing**: LlamaParse agentic_plus (Anthropic Sonnet 4.0)
- **Agent Orchestration**: LangGraph (Phase 3)
- **Reranker**: Cohere Rerank 3 (Phase 4)

## Database
- **Neon project**: RAG_CHAT (`icy-fire-24933610`)
- **Connection**: stored in `.env` as `DATABASE_URL`
- **Extensions**: `vector` (pgvector), `pg_trgm`
- **Neon Tables**: documents, chunks, chunk_embeddings, entity_embeddings, relation_embeddings, community_summary_embeddings, ingestion_logs
- **Neo4j Nodes**: Entity (name, type, status, community_id), Community (summary)

## Implementation Phases
- **Phase 1** ✅: Foundation — ingestion, Neon schema, structure-aware chunking, LLM enrichment, chunk embeddings
- **Phase 2** ✅: Graph RAG — Neo4j entity extraction, 3-tier dedup, community detection, GraphSAGE, TransE, hybrid embeddings
- **Phase 3**: Agentic Engine — LangGraph multi-agent, query classifier, semantic cache
- **Phase 4**: Retrieval + Validation — hybrid retrieval (vector+BM25+graph), Cohere reranker, parallel validation
- **Phase 5**: Observability — LangSmith, RAGAS, dashboards
- **Phase 6**: Security & Optimization — prompt injection, ACL, PII, rate limiting, A/B testing

## Project Structure
```
rag_chat/
├── app/
│   ├── config.py              # 93 lines — all settings (LlamaParse, OpenAI, Neo4j, graph, embeddings)
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
├── migrations/                # 10 SQL migrations (001–010)
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

## Documentation
- `docs/layer1_data_ingestion.md` — Layer 1 technical doc (pipeline, schema, config, error handling)
- `docs/layer2_knowledge_graph.md` — Layer 2 technical doc (graph extraction, embeddings, community detection)

## Blueprint Reference
See `SAE-Steer_DSE316_COURSE_PROJECT_PROPOSAL.pdf` for the full 25-page architecture document.
