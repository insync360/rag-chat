-- Chunk embeddings: text-embedding-3-large (2000-dim) for hybrid retrieval
-- 2000 dims: Neon pgvector HNSW limit; text-embedding-3-large supports reduced dims via API

CREATE TABLE chunk_embeddings (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id     UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    embedding    vector(2000) NOT NULL,
    model        TEXT NOT NULL DEFAULT 'text-embedding-3-large',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX idx_chunk_emb_chunk_id ON chunk_embeddings (chunk_id);
CREATE INDEX idx_chunk_emb_vector ON chunk_embeddings
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);
