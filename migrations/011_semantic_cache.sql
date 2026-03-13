-- Semantic cache: embed queries at 256-dim, cosine similarity for cache hits

CREATE TABLE semantic_cache (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text      TEXT NOT NULL,
    query_embedding vector(256) NOT NULL,
    query_type      TEXT NOT NULL,
    answer          TEXT NOT NULL,
    result_json     JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at      TIMESTAMPTZ NOT NULL DEFAULT now() + interval '24 hours'
);

CREATE INDEX idx_semantic_cache_embedding ON semantic_cache
    USING hnsw (query_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX idx_semantic_cache_expires ON semantic_cache (expires_at);
