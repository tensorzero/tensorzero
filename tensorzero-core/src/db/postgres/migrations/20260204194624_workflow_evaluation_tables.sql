-- Workflow evaluation tables
-- These correspond to DynamicEvaluationRun and DynamicEvaluationRunEpisode in ClickHouse

-- Helper function to convert UInt128 (as NUMERIC) to UUID
-- ClickHouse stores UUIDs as UInt128, and fixtures export them as decimal strings.
-- This function converts the decimal representation back to UUID format.
CREATE OR REPLACE FUNCTION tensorzero.uint128_to_uuid(p_uint128 NUMERIC)
RETURNS UUID AS $$
DECLARE
    hex_str TEXT := '';
    remainder NUMERIC;
    temp NUMERIC;
    hex_digit TEXT;
    hex_chars TEXT := '0123456789abcdef';
BEGIN
    temp := p_uint128;
    -- Convert numeric to hex string
    WHILE temp > 0 LOOP
        remainder := temp % 16;
        hex_digit := substring(hex_chars FROM (remainder::INT + 1) FOR 1);
        hex_str := hex_digit || hex_str;
        temp := floor(temp / 16);
    END LOOP;
    -- Pad to 32 characters
    hex_str := lpad(hex_str, 32, '0');
    -- Format as UUID (8-4-4-4-12)
    RETURN (
        substring(hex_str FROM 1 FOR 8) || '-' ||
        substring(hex_str FROM 9 FOR 4) || '-' ||
        substring(hex_str FROM 13 FOR 4) || '-' ||
        substring(hex_str FROM 17 FOR 4) || '-' ||
        substring(hex_str FROM 21 FOR 12)
    )::UUID;
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT;

-- workflow_evaluation_runs: stores workflow evaluation runs
CREATE TABLE tensorzero.workflow_evaluation_runs (
    run_id UUID PRIMARY KEY,
    project_name TEXT,
    run_display_name TEXT,
    variant_pins JSONB NOT NULL DEFAULT '{}',
    tags JSONB NOT NULL DEFAULT '{}',
    snapshot_hash BYTEA,
    staled_at TIMESTAMPTZ,
    -- created_at should be derived from UUIDv7; do not set a default value.
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for workflow_evaluation_runs
CREATE INDEX idx_workflow_eval_runs_project
    ON tensorzero.workflow_evaluation_runs(project_name, run_id) WHERE staled_at IS NULL;
CREATE INDEX idx_workflow_eval_runs_created_at
    ON tensorzero.workflow_evaluation_runs(created_at DESC);
CREATE INDEX idx_workflow_eval_runs_updated_at
    ON tensorzero.workflow_evaluation_runs(updated_at DESC);

-- workflow_evaluation_run_episodes: stores workflow evaluation run episodes
CREATE TABLE tensorzero.workflow_evaluation_run_episodes (
    episode_id UUID PRIMARY KEY,
    run_id UUID NOT NULL,
    variant_pins JSONB NOT NULL DEFAULT '{}',
    datapoint_name TEXT,  -- externally called "task_name"
    tags JSONB NOT NULL DEFAULT '{}',
    snapshot_hash BYTEA,
    staled_at TIMESTAMPTZ,
    -- created_at should be derived from UUIDv7; do not set a default value.
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for workflow_evaluation_run_episodes
CREATE INDEX idx_workflow_eval_episodes_run
    ON tensorzero.workflow_evaluation_run_episodes(run_id, episode_id) WHERE staled_at IS NULL;
CREATE INDEX idx_workflow_eval_episodes_created_at
    ON tensorzero.workflow_evaluation_run_episodes(created_at DESC);
CREATE INDEX idx_workflow_eval_episodes_updated_at
    ON tensorzero.workflow_evaluation_run_episodes(updated_at DESC);
