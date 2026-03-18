-- Migration 013: Enriched BM25 search via pre-computed tsvector
-- Combines content + hypothetical_questions + keywords + summary for full-text search

-- 1. Add search_tsvector column
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_tsvector tsvector;

-- 2. Backfill from existing metadata
UPDATE chunks SET search_tsvector = to_tsvector('english',
    content
    || ' ' || COALESCE(metadata->>'summary', '')
    || ' ' || COALESCE(
        (SELECT string_agg(q, ' ') FROM jsonb_array_elements_text(metadata->'hypothetical_questions') q),
        '')
    || ' ' || COALESCE(
        (SELECT string_agg(k, ' ') FROM jsonb_array_elements_text(metadata->'keywords') k),
        '')
);

-- 3. GIN index for fast BM25 search
CREATE INDEX IF NOT EXISTS idx_chunks_search_tsvector ON chunks USING GIN (search_tsvector);

-- 4. Trigger for future inserts/updates
CREATE OR REPLACE FUNCTION chunks_search_tsvector_trigger() RETURNS trigger AS $$
BEGIN
    NEW.search_tsvector := to_tsvector('english',
        NEW.content
        || ' ' || COALESCE(NEW.metadata->>'summary', '')
        || ' ' || COALESCE(
            (SELECT string_agg(q, ' ') FROM jsonb_array_elements_text(NEW.metadata->'hypothetical_questions') q),
            '')
        || ' ' || COALESCE(
            (SELECT string_agg(k, ' ') FROM jsonb_array_elements_text(NEW.metadata->'keywords') k),
            '')
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_chunks_search_tsvector
    BEFORE INSERT OR UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_search_tsvector_trigger();
