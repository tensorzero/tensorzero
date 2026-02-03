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

-- Schedule pg_cron jobs for partition management (if pg_cron is available)
DO $cron_setup$
BEGIN
    -- Check if pg_cron extension is available
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_cron') THEN
        -- Create future partitions daily at 00:05 UTC
        PERFORM cron.schedule(
            'tensorzero_create_model_inferences_partitions',
            '5 0 * * *',
            $$SELECT tensorzero.create_partitions('model_inferences');$$
        );

        -- Drop old partitions daily at 00:30 UTC (only acts if retention is configured)
        PERFORM cron.schedule(
            'tensorzero_drop_old_model_inferences_partitions',
            '30 0 * * *',
            $$SELECT tensorzero.drop_old_partitions('model_inferences', 'inference_retention_days');$$
        );

        RAISE NOTICE 'pg_cron jobs scheduled for model_inferences partition management';
    ELSE
        RAISE NOTICE 'pg_cron extension not available - partition management must be scheduled externally';
    END IF;
END $cron_setup$;

-- Default partition for backfilling historical data
CREATE TABLE tensorzero.model_inferences_default PARTITION OF tensorzero.model_inferences DEFAULT;
