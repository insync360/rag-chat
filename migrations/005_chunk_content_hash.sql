ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_hash TEXT;
CREATE INDEX IF NOT EXISTS idx_chunks_doc_hash ON chunks (document_id, content_hash);
