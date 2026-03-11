CREATE TABLE community_summary_embeddings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    community_id    INTEGER NOT NULL,
    summary_text    TEXT NOT NULL,
    embedding       vector(512) NOT NULL,
    model           TEXT NOT NULL DEFAULT 'text-embedding-3-large',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX idx_community_emb_cid ON community_summary_embeddings (community_id);
CREATE INDEX idx_community_emb_vector ON community_summary_embeddings
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
