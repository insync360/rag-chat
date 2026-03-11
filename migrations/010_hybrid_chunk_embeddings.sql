-- Hybrid chunk-entity embeddings: truncated text (512) + GraphSAGE (128) + TransE (128) = 768-dim

ALTER TABLE chunk_embeddings
    ADD COLUMN IF NOT EXISTS hybrid_embedding vector(768);
CREATE INDEX idx_chunk_emb_hybrid ON chunk_embeddings
    USING hnsw (hybrid_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
