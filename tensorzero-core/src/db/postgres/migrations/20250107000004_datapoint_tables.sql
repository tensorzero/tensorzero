-- =============================================================================
-- Migration: Datapoint Tables
-- Description: Creates chat_inference_datapoint and json_inference_datapoint
--              tables with soft delete support.
-- =============================================================================

-- =============================================================================
-- HELPER FUNCTION FOR UPDATED_AT TRIGGER
-- =============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- CHAT INFERENCE DATAPOINT
-- =============================================================================

CREATE TABLE chat_inference_datapoint (
    id UUID NOT NULL,
    dataset_name TEXT NOT NULL,
    function_name TEXT NOT NULL,
    episode_id UUID,
    input JSONB NOT NULL,
    output JSONB,
    tool_params TEXT,
    tags JSONB NOT NULL DEFAULT '{}',
    extra_body JSONB,
    dynamic_tools JSONB NOT NULL DEFAULT '[]',
    dynamic_provider_tools JSONB NOT NULL DEFAULT '[]',
    allowed_tools TEXT,
    tool_choice TEXT,
    parallel_tool_calls BOOLEAN,
    source_inference_id UUID,
    name TEXT,
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    is_custom BOOLEAN NOT NULL DEFAULT FALSE,
    staled_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    snapshot_hash BYTEA,
    PRIMARY KEY (dataset_name, id)
);

-- Partial index for active datapoints (not deleted)
CREATE INDEX idx_chat_datapoint_active ON chat_inference_datapoint(dataset_name, function_name)
    WHERE NOT is_deleted;
CREATE INDEX idx_chat_datapoint_updated ON chat_inference_datapoint(updated_at DESC);
CREATE INDEX idx_chat_datapoint_source ON chat_inference_datapoint(source_inference_id)
    WHERE source_inference_id IS NOT NULL;

-- Trigger for auto-updating updated_at
CREATE TRIGGER update_chat_datapoint_updated_at
    BEFORE UPDATE ON chat_inference_datapoint
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- JSON INFERENCE DATAPOINT
-- =============================================================================

CREATE TABLE json_inference_datapoint (
    id UUID NOT NULL,
    dataset_name TEXT NOT NULL,
    function_name TEXT NOT NULL,
    episode_id UUID,
    input JSONB NOT NULL,
    output JSONB,
    output_schema TEXT,
    tags JSONB NOT NULL DEFAULT '{}',
    extra_body JSONB,
    auxiliary_content TEXT,
    source_inference_id UUID,
    name TEXT,
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    is_custom BOOLEAN NOT NULL DEFAULT FALSE,
    staled_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    snapshot_hash BYTEA,
    PRIMARY KEY (dataset_name, id)
);

-- Partial index for active datapoints (not deleted)
CREATE INDEX idx_json_datapoint_active ON json_inference_datapoint(dataset_name, function_name)
    WHERE NOT is_deleted;
CREATE INDEX idx_json_datapoint_updated ON json_inference_datapoint(updated_at DESC);
CREATE INDEX idx_json_datapoint_source ON json_inference_datapoint(source_inference_id)
    WHERE source_inference_id IS NOT NULL;

-- Trigger for auto-updating updated_at
CREATE TRIGGER update_json_datapoint_updated_at
    BEFORE UPDATE ON json_inference_datapoint
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
