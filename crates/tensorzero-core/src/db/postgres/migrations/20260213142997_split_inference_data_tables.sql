-- Split inference tables into metadata and data tables.
-- Creates *_data tables (daily partitions) for large payload columns, then drops
-- those columns from the original tables.
-- Also converts the three original inference tables from daily to monthly partitions.

-- ============================================================================
-- Monthly partition management functions
-- ============================================================================

-- Create future monthly partitions for a given table.
-- Creates partitions for the current month and the next 3 months.
CREATE OR REPLACE FUNCTION tensorzero.create_monthly_partitions(p_table_name TEXT)
RETURNS void AS $$
DECLARE
    partition_start DATE;
    partition_end DATE;
    partition_name TEXT;
BEGIN
    FOR i IN 0..3 LOOP
        partition_start := date_trunc('month', CURRENT_DATE) + (i || ' months')::INTERVAL;
        partition_end := partition_start + INTERVAL '1 month';
        partition_name := p_table_name || '_' || to_char(partition_start, 'YYYY_MM');
        IF NOT EXISTS (
            SELECT 1 FROM pg_tables WHERE schemaname = 'tensorzero' AND tablename = partition_name
        ) THEN
            EXECUTE format(
                'CREATE TABLE IF NOT EXISTS tensorzero.%I PARTITION OF tensorzero.%I FOR VALUES FROM (%L) TO (%L)',
                partition_name,
                p_table_name,
                partition_start,
                partition_end
            );
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Drop old monthly partitions for a given table.
-- Reads retention in days from retention_config; drops a monthly partition only
-- when the entire month is older than the cutoff date.
CREATE OR REPLACE FUNCTION tensorzero.drop_old_monthly_partitions(p_table_name TEXT, p_retention_key TEXT)
RETURNS void AS $$
DECLARE
    retention_days INT;
    cutoff_date DATE;
    partition_record RECORD;
    pattern TEXT;
    partition_month_start DATE;
BEGIN
    SELECT value::INT INTO retention_days
    FROM tensorzero.retention_config
    WHERE key = p_retention_key;

    IF retention_days IS NULL THEN
        RAISE NOTICE '% not configured, skipping partition cleanup for %', p_retention_key, p_table_name;
        RETURN;
    END IF;

    cutoff_date := CURRENT_DATE - retention_days;
    pattern := '^' || p_table_name || '_\d{4}_\d{2}$';

    FOR partition_record IN
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'tensorzero' AND tablename ~ pattern
    LOOP
        partition_month_start := to_date(substring(partition_record.tablename from '\d{4}_\d{2}$'), 'YYYY_MM');
        -- Drop only when the entire month is before the cutoff
        IF partition_month_start + INTERVAL '1 month' <= cutoff_date THEN
            EXECUTE format('DROP TABLE tensorzero.%I', partition_record.tablename);
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- chat_inference_data (daily partitions)
-- ============================================================================

CREATE TABLE tensorzero.chat_inference_data (
    id UUID NOT NULL,
    input JSONB NOT NULL,
    output JSONB NOT NULL,
    inference_params JSONB NOT NULL,
    extra_body JSONB NOT NULL DEFAULT '[]',
    dynamic_tools JSONB NOT NULL DEFAULT '[]',
    dynamic_provider_tools JSONB NOT NULL DEFAULT '[]',
    allowed_tools JSONB,
    tool_choice JSONB,
    parallel_tool_calls BOOLEAN,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

SELECT tensorzero.create_partitions('chat_inference_data');

CREATE TABLE tensorzero.chat_inference_data_default
    PARTITION OF tensorzero.chat_inference_data DEFAULT;

-- ============================================================================
-- json_inference_data (daily partitions)
-- ============================================================================

CREATE TABLE tensorzero.json_inference_data (
    id UUID NOT NULL,
    input JSONB NOT NULL,
    output JSONB NOT NULL,
    output_schema JSONB NOT NULL,
    inference_params JSONB NOT NULL,
    extra_body JSONB NOT NULL DEFAULT '[]',
    auxiliary_content JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

SELECT tensorzero.create_partitions('json_inference_data');

CREATE TABLE tensorzero.json_inference_data_default
    PARTITION OF tensorzero.json_inference_data DEFAULT;

-- ============================================================================
-- model_inference_data (daily partitions)
-- ============================================================================

CREATE TABLE tensorzero.model_inference_data (
    id UUID NOT NULL,
    raw_request TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    system TEXT,
    input_messages JSONB NOT NULL,
    output JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

SELECT tensorzero.create_partitions('model_inference_data');

CREATE TABLE tensorzero.model_inference_data_default
    PARTITION OF tensorzero.model_inference_data DEFAULT;

-- ============================================================================
-- batch_model_inference_data (daily partitions)
-- ============================================================================
-- Splits large payload columns from batch_model_inferences into a separate
-- table with daily partitions for independent retention.

CREATE TABLE tensorzero.batch_model_inference_data (
    inference_id UUID NOT NULL,
    input JSONB NOT NULL,
    input_messages JSONB NOT NULL,
    system TEXT,
    inference_params JSONB NOT NULL,
    raw_request TEXT NOT NULL,
    output_schema JSONB,
    dynamic_tools JSONB NOT NULL DEFAULT '[]',
    dynamic_provider_tools JSONB NOT NULL DEFAULT '[]',
    allowed_tools JSONB,
    tool_choice JSONB,
    parallel_tool_calls BOOLEAN,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (inference_id, created_at)
) PARTITION BY RANGE (created_at);

SELECT tensorzero.create_partitions('batch_model_inference_data');

CREATE TABLE tensorzero.batch_model_inference_data_default
    PARTITION OF tensorzero.batch_model_inference_data DEFAULT;

-- ============================================================================
-- batch_request_data (daily partitions)
-- ============================================================================
-- Splits large payload columns (raw_request, raw_response) from batch_requests
-- into a separate table with daily partitions for independent retention.

CREATE TABLE tensorzero.batch_request_data (
    id UUID NOT NULL,
    raw_request TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    errors JSONB,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

SELECT tensorzero.create_partitions('batch_request_data');

CREATE TABLE tensorzero.batch_request_data_default
    PARTITION OF tensorzero.batch_request_data DEFAULT;


-- ============================================================================
-- Drop payload columns from existing tables
-- ============================================================================

ALTER TABLE tensorzero.chat_inferences
    DROP COLUMN input,
    DROP COLUMN output,
    DROP COLUMN inference_params,
    DROP COLUMN extra_body,
    DROP COLUMN dynamic_tools,
    DROP COLUMN dynamic_provider_tools,
    DROP COLUMN allowed_tools,
    DROP COLUMN tool_choice,
    DROP COLUMN parallel_tool_calls;

ALTER TABLE tensorzero.json_inferences
    DROP COLUMN input,
    DROP COLUMN output,
    DROP COLUMN output_schema,
    DROP COLUMN inference_params,
    DROP COLUMN extra_body,
    DROP COLUMN auxiliary_content;

ALTER TABLE tensorzero.model_inferences
    DROP COLUMN raw_request,
    DROP COLUMN raw_response,
    DROP COLUMN system,
    DROP COLUMN input_messages,
    DROP COLUMN output;

-- batch_requests and batch_model_inferences are dropped and recreated as
-- partitioned tables below, so no ALTER TABLE needed here.

-- ============================================================================
-- Convert existing inference tables from daily to monthly partitions
-- ============================================================================

-- Drop all existing daily partitions and default partitions.
DO $$
DECLARE
    partition_record RECORD;
BEGIN
    FOR partition_record IN
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'tensorzero'
        AND (
            tablename ~ '^chat_inferences_\d{4}_\d{2}_\d{2}$'
            OR tablename ~ '^json_inferences_\d{4}_\d{2}_\d{2}$'
            OR tablename ~ '^model_inferences_\d{4}_\d{2}_\d{2}$'
        )
    LOOP
        EXECUTE format('DROP TABLE tensorzero.%I', partition_record.tablename);
    END LOOP;
END $$;

DROP TABLE IF EXISTS tensorzero.chat_inferences_default;
DROP TABLE IF EXISTS tensorzero.json_inferences_default;
DROP TABLE IF EXISTS tensorzero.model_inferences_default;

-- Create monthly partitions for the current month + next 3 months
SELECT tensorzero.create_monthly_partitions('chat_inferences');
SELECT tensorzero.create_monthly_partitions('json_inferences');
SELECT tensorzero.create_monthly_partitions('model_inferences');

-- Default partitions for backfilling historical data
CREATE TABLE tensorzero.chat_inferences_default PARTITION OF tensorzero.chat_inferences DEFAULT;
CREATE TABLE tensorzero.json_inferences_default PARTITION OF tensorzero.json_inferences DEFAULT;
CREATE TABLE tensorzero.model_inferences_default PARTITION OF tensorzero.model_inferences DEFAULT;

-- ============================================================================
-- Convert batch tables from non-partitioned to monthly partitions
-- ============================================================================
-- These tables were not released, so we can drop and recreate them.

DROP TABLE tensorzero.batch_requests;

CREATE TABLE tensorzero.batch_requests (
    id UUID NOT NULL,
    batch_id UUID NOT NULL,
    function_name TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_provider_name TEXT NOT NULL,
    batch_params JSONB NOT NULL,
    status TEXT NOT NULL,
    snapshot_hash BYTEA,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

SELECT tensorzero.create_monthly_partitions('batch_requests');
CREATE TABLE tensorzero.batch_requests_default PARTITION OF tensorzero.batch_requests DEFAULT;

CREATE INDEX idx_batch_requests_batch_id ON tensorzero.batch_requests(batch_id, id);

DROP TABLE tensorzero.batch_model_inferences;

CREATE TABLE tensorzero.batch_model_inferences (
    inference_id UUID NOT NULL,
    batch_id UUID NOT NULL,
    function_name TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    episode_id UUID NOT NULL,
    model_name TEXT NOT NULL,
    model_provider_name TEXT NOT NULL,
    tags JSONB NOT NULL DEFAULT '{}',
    snapshot_hash BYTEA,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (inference_id, created_at)
) PARTITION BY RANGE (created_at);

SELECT tensorzero.create_monthly_partitions('batch_model_inferences');
CREATE TABLE tensorzero.batch_model_inferences_default PARTITION OF tensorzero.batch_model_inferences DEFAULT;

CREATE INDEX idx_batch_model_inferences_batch ON tensorzero.batch_model_inferences(batch_id, inference_id);
