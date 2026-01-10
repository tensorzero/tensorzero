-- =============================================================================
-- Migration: Statistics and Config Tables
-- Description: Creates config_snapshot, deployment_id, and cumulative_usage tables.
-- =============================================================================

-- =============================================================================
-- CONFIG SNAPSHOT
-- =============================================================================

CREATE TABLE config_snapshot (
    hash BYTEA PRIMARY KEY,
    config TEXT NOT NULL,
    extra_templates JSONB NOT NULL DEFAULT '{}',
    tensorzero_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tags JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX idx_config_snapshot_last_used ON config_snapshot(last_used DESC);

-- =============================================================================
-- DEPLOYMENT ID (Singleton table)
-- =============================================================================

CREATE TABLE deployment_id (
    id SERIAL PRIMARY KEY,
    deployment_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT single_deployment_row CHECK (id = 1)
);

-- =============================================================================
-- CUMULATIVE USAGE
-- =============================================================================

CREATE TABLE cumulative_usage (
    type TEXT PRIMARY KEY,
    count BIGINT NOT NULL DEFAULT 0
);

-- Initialize counters
INSERT INTO cumulative_usage (type, count) VALUES
    ('input_tokens', 0),
    ('output_tokens', 0),
    ('model_inferences', 0)
ON CONFLICT (type) DO NOTHING;

-- =============================================================================
-- TRIGGER FOR CUMULATIVE USAGE UPDATE
-- Automatically increments counters when model_inference rows are inserted
-- =============================================================================

CREATE OR REPLACE FUNCTION increment_usage_counters()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE cumulative_usage SET count = count + COALESCE(NEW.input_tokens, 0) WHERE type = 'input_tokens';
    UPDATE cumulative_usage SET count = count + COALESCE(NEW.output_tokens, 0) WHERE type = 'output_tokens';
    UPDATE cumulative_usage SET count = count + 1 WHERE type = 'model_inferences';
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_cumulative_usage
    AFTER INSERT ON model_inference
    FOR EACH ROW EXECUTE FUNCTION increment_usage_counters();
