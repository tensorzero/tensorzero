-- model_inferences table (partitioned by day)
-- Each row represents a call to a model provider. Each inference can have multiple
-- model inferences (e.g., for fallbacks or retries).
CREATE TABLE tensorzero.model_inferences (
    id UUID NOT NULL,
    inference_id UUID NOT NULL,
    raw_request TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    system TEXT,
    input_messages JSONB NOT NULL,
    output JSONB NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    response_time_ms INTEGER,
    model_name TEXT NOT NULL,
    model_provider_name TEXT NOT NULL,
    ttft_ms INTEGER,
    cached BOOLEAN NOT NULL DEFAULT false,
    finish_reason TEXT,  -- Stored as snake_case: stop, stop_sequence, length, tool_call, content_filter, unknown
    snapshot_hash BYTEA,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Indexes for model_inferences
-- Primary lookup pattern: get all model inferences for a given inference
CREATE INDEX idx_model_inferences_inference_id ON tensorzero.model_inferences(inference_id, id);
-- Secondary pattern: filter by model/provider
CREATE INDEX idx_model_inferences_model_provider ON tensorzero.model_inferences(model_name, model_provider_name, created_at);

-- Create initial partitions for the next 7 days
SELECT tensorzero.create_partitions('model_inferences');

-- Default partition for backfilling historical data
CREATE TABLE tensorzero.model_inferences_default PARTITION OF tensorzero.model_inferences DEFAULT;
