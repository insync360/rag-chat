-- Structure-aware chunks with FK to documents
-- Idempotent: delete + reinsert on re-chunk via CASCADE or explicit delete

CREATE TABLE IF NOT EXISTS chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    token_count     INTEGER NOT NULL,
    section_path    TEXT NOT NULL DEFAULT '',
    has_table       BOOLEAN NOT NULL DEFAULT FALSE,
    has_code        BOOLEAN NOT NULL DEFAULT FALSE,
    overlap_tokens  INTEGER NOT NULL DEFAULT 0,
    metadata        JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_document_index
    ON chunks (document_id, chunk_index);

-- GIN index on tsvector for future BM25 keyword search
CREATE INDEX IF NOT EXISTS idx_chunks_content_tsvector
    ON chunks USING GIN (to_tsvector('english', content));
