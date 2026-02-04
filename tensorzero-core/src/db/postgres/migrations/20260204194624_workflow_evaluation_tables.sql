-- Workflow evaluation tables
-- These correspond to DynamicEvaluationRun and DynamicEvaluationRunEpisode in ClickHouse

-- Helper function to convert UInt128 (as NUMERIC) to UUID
-- ClickHouse stores UUIDs as UInt128, and fixtures export them as decimal strings.
-- This function converts the decimal representation back to UUID format.
--
-- Only used in fixture loading.
CREATE OR REPLACE FUNCTION tensorzero.uint128_to_uuid(p_uint128 NUMERIC)
RETURNS UUID AS $$
DECLARE
    bytes BYTEA;
    i INT;
    byte_val INT;
    temp NUMERIC;
BEGIN
    -- Initialize 16-byte array with zeros
    bytes := '\x00000000000000000000000000000000'::BYTEA;
    temp := trunc(p_uint128);  -- Ensure we have an integer value

    -- Build 16 bytes from least significant to most significant (big-endian output)
    -- Use mod() and div() for precise integer arithmetic on large NUMERIC values
    FOR i IN REVERSE 15..0 LOOP
        byte_val := mod(temp, 256)::INT;
        bytes := set_byte(bytes, i, byte_val);
        temp := div(temp, 256);
    END LOOP;

    -- Convert bytea to hex string and format as UUID
    RETURN encode(bytes, 'hex')::UUID;
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

-- workflow_evaluation_run_episodes: stores workflow evaluation run episodes
CREATE TABLE tensorzero.workflow_evaluation_run_episodes (
    episode_id UUID PRIMARY KEY,
    run_id UUID NOT NULL,
    variant_pins JSONB NOT NULL DEFAULT '{}',
    task_name TEXT,
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

-- Add indexes to support DISTINCT ON (target_id, metric_name) ORDER BY target_id, metric_name, created_at DESC
-- These indexes optimize queries that fetch the most recent feedback per (target_id, metric_name) pair.

CREATE INDEX idx_float_feedback_target_metric_created
    ON tensorzero.float_metric_feedback(target_id, metric_name, created_at DESC);

CREATE INDEX idx_boolean_feedback_target_metric_created
    ON tensorzero.boolean_metric_feedback(target_id, metric_name, created_at DESC);
