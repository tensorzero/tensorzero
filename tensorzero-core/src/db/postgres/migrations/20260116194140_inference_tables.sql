-- chat_inferences (partitioned by day)
CREATE TABLE chat_inferences (
    id UUID NOT NULL,
    function_name TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    episode_id UUID NOT NULL,
    input TEXT NOT NULL,
    output TEXT NOT NULL,
    tool_params TEXT NOT NULL,
    inference_params TEXT NOT NULL,
    processing_time_ms INTEGER,
    ttft_ms INTEGER,
    tags JSONB DEFAULT '{}',
    extra_body TEXT,
    dynamic_tools TEXT[],
    dynamic_provider_tools TEXT[],
    allowed_tools TEXT,
    tool_choice TEXT,
    parallel_tool_calls BOOLEAN,
    snapshot_hash BYTEA,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Indexes for chat_inferences
CREATE INDEX idx_chat_inferences_function_variant_episode
    ON chat_inferences(function_name, variant_name, episode_id);
CREATE INDEX idx_chat_inferences_episode_id ON chat_inferences(episode_id);
CREATE INDEX idx_chat_inferences_function_id ON chat_inferences(function_name, id);
CREATE INDEX idx_chat_inferences_input_trgm ON chat_inferences USING GIN (input gin_trgm_ops);
CREATE INDEX idx_chat_inferences_output_trgm ON chat_inferences USING GIN (output gin_trgm_ops);
CREATE INDEX idx_chat_inferences_tags ON chat_inferences USING GIN (tags);

-- json_inferences (partitioned by day)
CREATE TABLE json_inferences (
    id UUID NOT NULL,
    function_name TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    episode_id UUID NOT NULL,
    input TEXT NOT NULL,
    output TEXT NOT NULL,
    output_schema TEXT NOT NULL,
    inference_params TEXT NOT NULL,
    processing_time_ms INTEGER,
    ttft_ms INTEGER,
    tags JSONB DEFAULT '{}',
    extra_body TEXT,
    auxiliary_content TEXT NOT NULL,
    snapshot_hash BYTEA,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Indexes for json_inferences
CREATE INDEX idx_json_inferences_function_variant_episode
    ON json_inferences(function_name, variant_name, episode_id);
CREATE INDEX idx_json_inferences_episode_id ON json_inferences(episode_id);
CREATE INDEX idx_json_inferences_function_id ON json_inferences(function_name, id);
CREATE INDEX idx_json_inferences_input_trgm ON json_inferences USING GIN (input gin_trgm_ops);
CREATE INDEX idx_json_inferences_output_trgm ON json_inferences USING GIN (output gin_trgm_ops);
CREATE INDEX idx_json_inferences_tags ON json_inferences USING GIN (tags);

-- Create initial partitions for the next 7 days
SELECT tensorzero_create_partitions('chat_inferences');
SELECT tensorzero_create_partitions('json_inferences');

-- Schedule pg_cron jobs for partition management (if pg_cron is available)
DO $$
BEGIN
    -- Check if pg_cron extension is available
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_cron') THEN
        -- Create future partitions daily at 00:05 UTC
        PERFORM cron.schedule(
            'tensorzero_create_inference_partitions',
            '5 0 * * *',
            $$SELECT tensorzero_create_partitions('chat_inferences'); SELECT tensorzero_create_partitions('json_inferences');$$
        );

        -- Drop old partitions daily at 00:30 UTC (only acts if retention is configured)
        PERFORM cron.schedule(
            'tensorzero_drop_old_inference_partitions',
            '30 0 * * *',
            $$SELECT tensorzero_drop_old_partitions('chat_inferences', 'inference_retention_days'); SELECT tensorzero_drop_old_partitions('json_inferences', 'inference_retention_days');$$
        );

        RAISE NOTICE 'pg_cron jobs scheduled for inference partition management';
    ELSE
        RAISE NOTICE 'pg_cron extension not available - partition management must be scheduled externally';
    END IF;
END $$;
