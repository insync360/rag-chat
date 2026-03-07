-- Version-tracked documents table
-- Design: content_hash deduplication, deprecate never delete

CREATE TABLE IF NOT EXISTS documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename        TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    version         INTEGER NOT NULL DEFAULT 1,
    status          TEXT NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'deprecated')),
    page_count      INTEGER,
    full_markdown   TEXT,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    deprecated_at   TIMESTAMPTZ,
    metadata        JSONB DEFAULT '{}'::jsonb
);

-- Fast lookup by hash (idempotency check)
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_content_hash
    ON documents (content_hash);

-- Find active version of a file
CREATE INDEX IF NOT EXISTS idx_documents_filename_status
    ON documents (filename, status);

-- Find latest version of a file
CREATE INDEX IF NOT EXISTS idx_documents_filename_version
    ON documents (filename, version DESC);
