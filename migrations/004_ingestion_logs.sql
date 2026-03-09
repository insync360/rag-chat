CREATE TABLE IF NOT EXISTS ingestion_logs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename        TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending','processing','completed','failed','skipped')),
    document_id     UUID REFERENCES documents(id),
    attempt         INTEGER NOT NULL DEFAULT 1,
    step            TEXT,                        -- last step reached
    chunk_count     INTEGER,
    error_message   TEXT,
    step_timings    JSONB DEFAULT '{}'::jsonb,   -- {"parse_ms": 1200, ...}
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_ingestion_logs_status ON ingestion_logs (status);
