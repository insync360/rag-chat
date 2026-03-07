# RAG Chat — Production Agentic + Graph RAG

## Project Overview
Production-grade Agentic + Graph RAG system with 8 architectural layers. Accuracy-first design with hybrid retrieval, knowledge graph, multi-agent orchestration, and zero-hallucination guardrails.

## Architecture (8 Layers)
1. **Data Ingestion Pipeline** — Parse, structure-aware chunk, enrich documents
2. **Knowledge Graph + Graph RAG** — Entity/relationship extraction, Neo4j graph
3. **Database Layer** — Neon PostgreSQL (pgvector + tsvector) + Neo4j
4. **Query Processing Engine** — Query classification, semantic cache, decomposition
5. **Agentic Retrieval Engine** — LangGraph multi-agent (6 agents)
6. **Validation & Guardrails** — Parallel faithfulness/relevance/coherence checks
7. **Evaluation & Observability** — RAGAS, LangSmith, online metrics
8. **Stress Testing & Security** — Prompt injection defense, ACL, rate limiting

## Technology Stack
- **Database**: Neon PostgreSQL v17 + pgvector (project: `icy-fire-24933610`, region: `aws-ap-southeast-1`)
- **LLM Provider**: OpenAI (GPT-4o generation, GPT-4o-mini validation)
- **Embeddings**: OpenAI text-embedding-3-large (3072 dimensions)
- **API Framework**: FastAPI (async)
- **Graph DB**: Neo4j (Phase 2)
- **Agent Orchestration**: LangGraph (Phase 3)
- **Document Parsing**: Unstructured.io
- **Reranker**: Cohere Rerank 3

## Database
- **Neon project**: RAG_CHAT (`icy-fire-24933610`)
- **Connection**: stored in `.env` as `DATABASE_URL`
- **Extensions**: `vector` (pgvector), `pg_trgm`
- **Tables**: documents, chunks, chunk_embeddings, chunk_metadata, retrieval_logs, evaluation_logs

## Implementation Phases
- **Phase 1 (current)**: Foundation — ingestion, Neon schema, hybrid retrieval (vector+BM25), parallel validation
- **Phase 2**: Graph RAG — Neo4j, entity extraction, knowledge graph
- **Phase 3**: Agentic Engine — LangGraph multi-agent, query classifier, semantic cache
- **Phase 4**: Observability — LangSmith, RAGAS, dashboards
- **Phase 5**: Security — Red team, prompt injection, ACL, PII, rate limiting
- **Phase 6**: Optimization — A/B testing, cache tuning, cost optimization

## Project Structure
```
rag_chat/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Settings from .env
│   ├── database.py          # Neon async connection
│   ├── ingestion/           # Layer 1: parse, chunk, enrich, pipeline
│   ├── retrieval/           # Layer 5: vector_search, keyword_search, hybrid, reranker
│   ├── generation/          # Response generation with GPT-4o
│   ├── validation/          # Layer 6: faithfulness, relevance, parallel validator
│   └── api/                 # FastAPI routes: /ingest, /query
├── migrations/              # SQL migrations for Neon
├── requirements.txt
├── .env                     # DATABASE_URL, OPENAI_API_KEY
└── .gitignore
```

## Key Design Decisions
- **Hybrid retrieval**: Vector (pgvector cosine) + BM25 (tsvector) fused via Reciprocal Rank Fusion (k=60)
- **Structure-aware chunking**: Heading boundaries, table integrity, 256-512 token range, 10-15% overlap
- **Metadata enrichment**: LLM-generated summaries, keywords, hypothetical questions (HyDE) per chunk
- **Parallel validation**: Faithfulness (>0.85) + Relevance (>0.80) checked concurrently via GPT-4o-mini
- **Idempotent ingestion**: content_hash deduplication, version tracking (deprecate, never delete)

## Blueprint Reference
See `production_rag_blueprint (1).pdf` for the full 25-page architecture document.
