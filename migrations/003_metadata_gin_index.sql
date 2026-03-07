-- GIN index on chunks.metadata JSONB for efficient metadata queries
CREATE INDEX IF NOT EXISTS idx_chunks_metadata_gin
    ON chunks USING GIN (metadata);
