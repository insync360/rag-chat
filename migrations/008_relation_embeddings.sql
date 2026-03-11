-- Relation type embeddings (TransE) + entity TransE embedding column

CREATE TABLE relation_embeddings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    relation_type   TEXT NOT NULL,
    embedding       vector(128) NOT NULL,
    model_version   TEXT NOT NULL DEFAULT 'v1',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX idx_rel_emb_type ON relation_embeddings (relation_type);
CREATE INDEX idx_rel_emb_vector ON relation_embeddings
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

ALTER TABLE entity_embeddings
    ADD COLUMN IF NOT EXISTS transe_embedding vector(128);
CREATE INDEX idx_entity_emb_transe ON entity_embeddings
    USING hnsw (transe_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
