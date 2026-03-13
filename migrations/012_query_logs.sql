-- Query logs: track every retrieval query for observability

CREATE TABLE query_logs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text      TEXT NOT NULL,
    query_type      TEXT NOT NULL,
    cached          BOOLEAN NOT NULL DEFAULT FALSE,
    chunks_retrieved INTEGER,
    graph_paths     INTEGER DEFAULT 0,
    conflicts       INTEGER DEFAULT 0,
    step_timings    JSONB DEFAULT '{}'::jsonb,
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_query_logs_created ON query_logs (created_at DESC);
CREATE INDEX idx_query_logs_type ON query_logs (query_type);
