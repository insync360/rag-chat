-- Entity embeddings: GraphSAGE structural (128-dim) + OpenAI text (256-dim)

CREATE TABLE entity_embeddings (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_name      TEXT NOT NULL,
    entity_type      TEXT NOT NULL,
    neo4j_id         TEXT,
    embedding        vector(128) NOT NULL,   -- GraphSAGE structural embedding
    text_embedding   vector(256),            -- OpenAI initial features (for text-based entity search)
    model_version    TEXT NOT NULL DEFAULT 'v1',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX idx_entity_emb_name_type ON entity_embeddings (entity_name, entity_type);
CREATE INDEX idx_entity_emb_structural ON entity_embeddings USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX idx_entity_emb_text ON entity_embeddings USING hnsw (text_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
