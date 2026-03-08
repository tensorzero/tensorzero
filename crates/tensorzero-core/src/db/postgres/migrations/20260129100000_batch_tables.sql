-- Batch processing tables for tracking batch inference jobs

-- batch_requests: stores batch request metadata
CREATE TABLE tensorzero.batch_requests (
    id UUID PRIMARY KEY,
    batch_id UUID NOT NULL,
    function_name TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_provider_name TEXT NOT NULL,
    batch_params JSONB NOT NULL,
    status TEXT NOT NULL,
    errors JSONB,
    raw_request TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    snapshot_hash BYTEA,
    -- created_at should be derived from UUIDv7; do not set a default value.
    created_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX idx_batch_requests_batch_id ON tensorzero.batch_requests(batch_id, id);
CREATE INDEX idx_batch_requests_created_at ON tensorzero.batch_requests(created_at);

-- batch_model_inferences: stores batch model inference metadata
CREATE TABLE tensorzero.batch_model_inferences (
    inference_id UUID PRIMARY KEY,
    batch_id UUID NOT NULL,
    function_name TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    episode_id UUID NOT NULL,
    model_name TEXT NOT NULL,
    model_provider_name TEXT NOT NULL,
    input JSONB NOT NULL,
    input_messages JSONB NOT NULL,
    system TEXT,
    inference_params JSONB NOT NULL,
    raw_request TEXT NOT NULL,
    output_schema JSONB,
    tags JSONB NOT NULL DEFAULT '{}',
    -- Decomposed tool params (no tool_params column)
    dynamic_tools JSONB NOT NULL DEFAULT '[]',
    dynamic_provider_tools JSONB NOT NULL DEFAULT '[]',
    allowed_tools JSONB,
    tool_choice JSONB,
    parallel_tool_calls BOOLEAN,
    snapshot_hash BYTEA,
    -- created_at should be derived from UUIDv7; do not set a default value.
    created_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX idx_batch_model_inferences_batch ON tensorzero.batch_model_inferences(batch_id, inference_id);
CREATE INDEX idx_batch_model_inferences_inference_id ON tensorzero.batch_model_inferences(inference_id);
CREATE INDEX idx_batch_model_inferences_created_at ON tensorzero.batch_model_inferences(created_at);
