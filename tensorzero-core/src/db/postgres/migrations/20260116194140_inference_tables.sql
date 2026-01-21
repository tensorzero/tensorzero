-- TODO(#5691): Add indexes for substring search on input/output JSONB columns

-- Extract timestamp from UUIDv7
-- UUIDv7 stores milliseconds since Unix epoch in the first 48 bits
CREATE OR REPLACE FUNCTION tensorzero.uuid_v7_to_timestamp(p_uuid UUID)
RETURNS TIMESTAMPTZ AS $$
BEGIN
    RETURN to_timestamp(
        ('x' || substring(replace(p_uuid::text, '-', ''), 1, 12))::bit(48)::bigint / 1000.0
    ) AT TIME ZONE 'UTC';
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT;

-- chat_inferences (partitioned by day)
CREATE TABLE tensorzero.chat_inferences (
    id UUID NOT NULL,
    function_name TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    episode_id UUID NOT NULL,
    input JSONB NOT NULL,
    output JSONB NOT NULL,
    tool_params JSONB NOT NULL,
    inference_params JSONB NOT NULL,
    processing_time_ms INTEGER,
    ttft_ms INTEGER,
    tags JSONB NOT NULL DEFAULT '{}',
    extra_body JSONB NOT NULL DEFAULT '[]',
    dynamic_tools JSONB NOT NULL DEFAULT '[]',
    dynamic_provider_tools JSONB NOT NULL DEFAULT '[]',
    allowed_tools JSONB,
    tool_choice TEXT,
    parallel_tool_calls BOOLEAN,
    snapshot_hash BYTEA,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Indexes for chat_inferences
CREATE INDEX idx_chat_inferences_function_variant_episode
    ON tensorzero.chat_inferences(function_name, variant_name, episode_id);
CREATE INDEX idx_chat_inferences_episode_id ON tensorzero.chat_inferences(episode_id);
CREATE INDEX idx_chat_inferences_function_id ON tensorzero.chat_inferences(function_name, id);
CREATE INDEX idx_chat_inferences_tags ON tensorzero.chat_inferences USING GIN (tags);

-- json_inferences (partitioned by day)
CREATE TABLE tensorzero.json_inferences (
    id UUID NOT NULL,
    function_name TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    episode_id UUID NOT NULL,
    input JSONB NOT NULL,
    output JSONB NOT NULL,
    output_schema JSONB NOT NULL,
    inference_params JSONB NOT NULL,
    processing_time_ms INTEGER,
    ttft_ms INTEGER,
    tags JSONB NOT NULL DEFAULT '{}',
    extra_body JSONB NOT NULL DEFAULT '[]',
    auxiliary_content JSONB NOT NULL,
    snapshot_hash BYTEA,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Indexes for json_inferences
CREATE INDEX idx_json_inferences_function_variant_episode
    ON tensorzero.json_inferences(function_name, variant_name, episode_id);
CREATE INDEX idx_json_inferences_episode_id ON tensorzero.json_inferences(episode_id);
CREATE INDEX idx_json_inferences_function_id ON tensorzero.json_inferences(function_name, id);
CREATE INDEX idx_json_inferences_tags ON tensorzero.json_inferences USING GIN (tags);

-- Create initial partitions for the next 7 days
SELECT tensorzero.create_partitions('chat_inferences');
SELECT tensorzero.create_partitions('json_inferences');

-- Schedule pg_cron jobs for partition management (if pg_cron is available)
DO $cron_setup$
BEGIN
    -- Check if pg_cron extension is available
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_cron') THEN
        -- Create future partitions daily at 00:05 UTC
        PERFORM cron.schedule(
            'tensorzero_create_inference_partitions',
            '5 0 * * *',
            $$SELECT tensorzero.create_partitions('chat_inferences'); SELECT tensorzero.create_partitions('json_inferences');$$
        );

        -- Drop old partitions daily at 00:30 UTC (only acts if retention is configured)
        PERFORM cron.schedule(
            'tensorzero_drop_old_inference_partitions',
            '30 0 * * *',
            $$SELECT tensorzero.drop_old_partitions('chat_inferences', 'inference_retention_days'); SELECT tensorzero.drop_old_partitions('json_inferences', 'inference_retention_days');$$
        );

        RAISE NOTICE 'pg_cron jobs scheduled for inference partition management';
    ELSE
        RAISE NOTICE 'pg_cron extension not available - partition management must be scheduled externally';
    END IF;
END $cron_setup$;

-- Default partitions for backfilling historical data
-- These catch any rows that don't fit into the date-based partitions
CREATE TABLE tensorzero.chat_inferences_default PARTITION OF tensorzero.chat_inferences DEFAULT;
CREATE TABLE tensorzero.json_inferences_default PARTITION OF tensorzero.json_inferences DEFAULT;
