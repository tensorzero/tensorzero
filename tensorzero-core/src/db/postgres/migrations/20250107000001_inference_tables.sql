-- =============================================================================
-- Migration: Core Inference Tables
-- Description: Creates chat_inference, json_inference, and model_inference tables
--              with appropriate indexes for the Postgres-only TensorZero deployment.
-- =============================================================================

-- =============================================================================
-- ENUM TYPES
-- =============================================================================

CREATE TYPE function_type_enum AS ENUM ('chat', 'json');
CREATE TYPE finish_reason_enum AS ENUM ('stop', 'length', 'tool_calls', 'content_filter', 'other');

-- =============================================================================
-- CHAT INFERENCE TABLE
-- =============================================================================

CREATE TABLE chat_inference (
    id UUID PRIMARY KEY,
    function_name TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    episode_id UUID NOT NULL,
    input JSONB NOT NULL,
    output JSONB NOT NULL,
    tool_params TEXT,
    inference_params JSONB,
    processing_time_ms INTEGER,
    -- Derived from UUIDv7 id (like ClickHouse's MATERIALIZED UUIDv7ToDateTime(id))
    timestamp TIMESTAMPTZ GENERATED ALWAYS AS (uuid_v7_to_timestamp(id)) STORED,
    tags JSONB NOT NULL DEFAULT '{}',
    extra_body JSONB,
    ttft_ms INTEGER,
    dynamic_tools JSONB NOT NULL DEFAULT '[]',
    dynamic_provider_tools JSONB NOT NULL DEFAULT '[]',
    allowed_tools TEXT,
    tool_choice TEXT,
    parallel_tool_calls BOOLEAN,
    snapshot_hash BYTEA
);

-- Indexes for common query patterns
CREATE INDEX idx_chat_inference_function_variant ON chat_inference(function_name, variant_name);
CREATE INDEX idx_chat_inference_episode ON chat_inference(episode_id);
CREATE INDEX idx_chat_inference_timestamp ON chat_inference(timestamp DESC);
CREATE INDEX idx_chat_inference_tags ON chat_inference USING GIN(tags);

-- Compound index for list queries with time filtering
-- Used by: inference_queries.rs:225 (WHERE function_name = X AND timestamp >= Y)
--          inference_count.rs:40,71 (WHERE function_name = {function_name:String})
-- Note: Since UUIDv7 is time-ordered, timestamp DESC provides equivalent ordering to id DESC
CREATE INDEX idx_chat_inference_function_ts
    ON chat_inference(function_name, timestamp DESC);

-- Episode queries with id-based ordering (no timestamp filter typical)
-- Used by: inference_queries.rs:248 (i.episode_id = {episode_id_param_placeholder})
--          inference_queries.rs:337 (episode_id = {episode_id:UUID})
--          test_helpers.rs:227 (SELECT * FROM ChatInference WHERE episode_id = ...)
-- Note: UUIDv7 id provides chronological ordering directly
CREATE INDEX idx_chat_inference_episode_id
    ON chat_inference(episode_id, id DESC);

-- Variant statistics optimization
-- Used by: inference_count.rs:71-73 (WHERE function_name = ... GROUP BY variant_name)
--          inference_count.rs:193-194 (GROUP BY variant_name with max(timestamp))
--          feedback.rs:695 (GROUP BY variant_name for metrics)
CREATE INDEX idx_chat_inference_function_variant_ts
    ON chat_inference(function_name, variant_name, timestamp DESC);

-- Text search on JSONB input/output
-- Used by: inference_queries.rs:275 (countSubstringsCaseInsensitiveUTF8(i.input, ...) + ...)
--          inference_queries.rs:864-867 (input_term_frequency, output_term_frequency)
--          dataset_queries.rs:283-284 (countSubstringsCaseInsensitiveUTF8 on input/output)
-- Note: GIN indices enable JSONB containment (@>) queries; full-text search may need pg_trgm
CREATE INDEX idx_chat_inference_input_gin
    ON chat_inference USING GIN(input jsonb_path_ops);
CREATE INDEX idx_chat_inference_output_gin
    ON chat_inference USING GIN(output jsonb_path_ops);

-- =============================================================================
-- JSON INFERENCE TABLE
-- =============================================================================

CREATE TABLE json_inference (
    id UUID PRIMARY KEY,
    function_name TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    episode_id UUID NOT NULL,
    input JSONB NOT NULL,
    output JSONB NOT NULL,
    output_schema TEXT,
    inference_params JSONB,
    processing_time_ms INTEGER,
    -- Derived from UUIDv7 id (like ClickHouse's MATERIALIZED UUIDv7ToDateTime(id))
    timestamp TIMESTAMPTZ GENERATED ALWAYS AS (uuid_v7_to_timestamp(id)) STORED,
    tags JSONB NOT NULL DEFAULT '{}',
    extra_body JSONB,
    ttft_ms INTEGER,
    auxiliary_content TEXT,
    snapshot_hash BYTEA
);

-- Indexes for common query patterns
CREATE INDEX idx_json_inference_function_variant ON json_inference(function_name, variant_name);
CREATE INDEX idx_json_inference_episode ON json_inference(episode_id);
CREATE INDEX idx_json_inference_timestamp ON json_inference(timestamp DESC);
CREATE INDEX idx_json_inference_tags ON json_inference USING GIN(tags);

-- Compound index for list queries with time filtering
-- Same query patterns as chat_inference (inference_queries.rs generates UNION ALL)
CREATE INDEX idx_json_inference_function_ts
    ON json_inference(function_name, timestamp DESC);

-- Episode queries with id-based ordering (no timestamp filter typical)
-- Same query patterns as chat_inference
CREATE INDEX idx_json_inference_episode_id
    ON json_inference(episode_id, id DESC);

-- Variant statistics optimization
-- Same query patterns as chat_inference
CREATE INDEX idx_json_inference_function_variant_ts
    ON json_inference(function_name, variant_name, timestamp DESC);

-- Text search on JSONB input/output
-- Same query patterns as chat_inference
CREATE INDEX idx_json_inference_input_gin
    ON json_inference USING GIN(input jsonb_path_ops);
CREATE INDEX idx_json_inference_output_gin
    ON json_inference USING GIN(output jsonb_path_ops);

-- =============================================================================
-- MODEL INFERENCE TABLE
-- =============================================================================

CREATE TABLE model_inference (
    id UUID PRIMARY KEY,
    inference_id UUID NOT NULL,
    raw_request TEXT,
    raw_response TEXT,
    model_name TEXT NOT NULL,
    model_provider_name TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    response_time_ms INTEGER,
    ttft_ms INTEGER,
    -- Derived from UUIDv7 id (like ClickHouse's MATERIALIZED UUIDv7ToDateTime(id))
    timestamp TIMESTAMPTZ GENERATED ALWAYS AS (uuid_v7_to_timestamp(id)) STORED,
    system TEXT,
    input_messages JSONB,
    output JSONB,
    cached BOOLEAN NOT NULL DEFAULT FALSE,
    finish_reason finish_reason_enum,
    snapshot_hash BYTEA
);

-- Indexes for common query patterns
CREATE INDEX idx_model_inference_inference_id ON model_inference(inference_id);
CREATE INDEX idx_model_inference_model ON model_inference(model_name, model_provider_name);
CREATE INDEX idx_model_inference_timestamp ON model_inference(timestamp DESC);

-- =============================================================================
-- UNIFIED INFERENCE LOOKUP VIEW
-- Replaces ClickHouse InferenceById materialized view
-- =============================================================================

CREATE VIEW inference_by_id AS
SELECT
    id,
    function_name,
    variant_name,
    episode_id,
    'chat'::function_type_enum AS function_type,
    snapshot_hash
FROM chat_inference
UNION ALL
SELECT
    id,
    function_name,
    variant_name,
    episode_id,
    'json'::function_type_enum AS function_type,
    snapshot_hash
FROM json_inference;
