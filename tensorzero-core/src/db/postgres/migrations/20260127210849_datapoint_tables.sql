-- Datapoint tables for storing training/evaluation data
-- These correspond to ChatInferenceDatapoint and JsonInferenceDatapoint in ClickHouse

-- chat_datapoints: stores chat inference datapoints for training/evaluation
CREATE TABLE tensorzero.chat_datapoints (
    id UUID PRIMARY KEY,
    dataset_name TEXT NOT NULL,
    function_name TEXT NOT NULL,
    episode_id UUID,
    input JSONB NOT NULL,
    output JSONB,
    dynamic_tools JSONB NOT NULL DEFAULT '[]',
    dynamic_provider_tools JSONB NOT NULL DEFAULT '[]',
    allowed_tools JSONB,
    tool_choice JSONB,
    parallel_tool_calls BOOLEAN,
    tags JSONB NOT NULL DEFAULT '{}',
    is_custom BOOLEAN NOT NULL DEFAULT false,
    source_inference_id UUID,
    name TEXT,
    snapshot_hash BYTEA,
    staled_at TIMESTAMPTZ,
    -- created_at should be derived from UUIDv7; do not set a default value.
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for chat_datapoints
CREATE INDEX idx_chat_datapoints_dataset_function
    ON tensorzero.chat_datapoints(dataset_name, function_name, id);
CREATE INDEX idx_chat_datapoints_dataset_not_staled
    ON tensorzero.chat_datapoints(dataset_name) WHERE staled_at IS NULL;
CREATE INDEX idx_chat_datapoints_updated_at
    ON tensorzero.chat_datapoints(updated_at DESC, id DESC);
CREATE INDEX idx_chat_datapoints_tags
    ON tensorzero.chat_datapoints USING GIN (tags);

-- json_datapoints: stores JSON inference datapoints for training/evaluation
CREATE TABLE tensorzero.json_datapoints (
    id UUID PRIMARY KEY,
    dataset_name TEXT NOT NULL,
    function_name TEXT NOT NULL,
    episode_id UUID,
    input JSONB NOT NULL,
    output JSONB,
    output_schema JSONB NOT NULL,
    tags JSONB NOT NULL DEFAULT '{}',
    is_custom BOOLEAN NOT NULL DEFAULT false,
    source_inference_id UUID,
    name TEXT,
    snapshot_hash BYTEA,
    staled_at TIMESTAMPTZ,
    -- created_at should be derived from UUIDv7; do not set a default value.
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for json_datapoints
CREATE INDEX idx_json_datapoints_dataset_function
    ON tensorzero.json_datapoints(dataset_name, function_name, id);
CREATE INDEX idx_json_datapoints_dataset_not_staled
    ON tensorzero.json_datapoints(dataset_name) WHERE staled_at IS NULL;
CREATE INDEX idx_json_datapoints_updated_at
    ON tensorzero.json_datapoints(updated_at DESC, id DESC);
CREATE INDEX idx_json_datapoints_tags
    ON tensorzero.json_datapoints USING GIN (tags);
