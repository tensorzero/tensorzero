-- =============================================================================
-- Migration: Core Inference Tables
-- Description: Creates chat_inference, json_inference, and model_inference tables
--              with appropriate indexes for the Postgres-only TensorZero deployment.
-- =============================================================================

-- =============================================================================
-- SCHEMA
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS tensorzero;

-- =============================================================================
-- UTILITY FUNCTIONS
-- =============================================================================

-- Extracts the embedded Unix timestamp from a UUIDv7.
-- UUIDv7 stores milliseconds since Unix epoch in the first 48 bits.
CREATE OR REPLACE FUNCTION tensorzero.uuid_v7_to_timestamp(id UUID)
RETURNS TIMESTAMPTZ AS $$
DECLARE
    hex_str TEXT;
    ts_ms BIGINT;
BEGIN
    -- Remove hyphens and get the hex string
    hex_str := replace(id::text, '-', '');
    -- First 48 bits (12 hex chars) are Unix timestamp in milliseconds
    ts_ms := ('x' || substring(hex_str, 1, 12))::bit(48)::bigint;
    RETURN to_timestamp(ts_ms / 1000.0);
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

-- =============================================================================
-- CHAT INFERENCE TABLE
-- =============================================================================

CREATE TABLE tensorzero.chat_inference (
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
    timestamp TIMESTAMPTZ GENERATED ALWAYS AS (tensorzero.uuid_v7_to_timestamp(id)) STORED,
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

-- Basic indexes
CREATE INDEX idx_chat_inference_function_variant ON tensorzero.chat_inference(function_name, variant_name);
CREATE INDEX idx_chat_inference_episode ON tensorzero.chat_inference(episode_id);
CREATE INDEX idx_chat_inference_timestamp ON tensorzero.chat_inference(timestamp DESC);
CREATE INDEX idx_chat_inference_tags ON tensorzero.chat_inference USING GIN(tags);

-- Compound index for list queries with time filtering
-- Used by: WHERE function_name = X AND timestamp >= Y ORDER BY timestamp DESC
CREATE INDEX idx_chat_inference_function_ts
    ON tensorzero.chat_inference(function_name, timestamp DESC);

-- Episode queries with id-based ordering (no timestamp filter typical)
-- UUIDv7 id provides chronological ordering directly
CREATE INDEX idx_chat_inference_episode_id
    ON tensorzero.chat_inference(episode_id, id DESC);

-- Variant statistics optimization
CREATE INDEX idx_chat_inference_function_variant_ts
    ON tensorzero.chat_inference(function_name, variant_name, timestamp DESC);

-- GIN indices for JSONB containment queries
CREATE INDEX idx_chat_inference_input_gin
    ON tensorzero.chat_inference USING GIN(input jsonb_path_ops);
CREATE INDEX idx_chat_inference_output_gin
    ON tensorzero.chat_inference USING GIN(output jsonb_path_ops);

-- =============================================================================
-- JSON INFERENCE TABLE
-- =============================================================================

CREATE TABLE tensorzero.json_inference (
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
    timestamp TIMESTAMPTZ GENERATED ALWAYS AS (tensorzero.uuid_v7_to_timestamp(id)) STORED,
    tags JSONB NOT NULL DEFAULT '{}',
    extra_body JSONB,
    ttft_ms INTEGER,
    auxiliary_content TEXT,
    snapshot_hash BYTEA
);

-- Basic indexes
CREATE INDEX idx_json_inference_function_variant ON tensorzero.json_inference(function_name, variant_name);
CREATE INDEX idx_json_inference_episode ON tensorzero.json_inference(episode_id);
CREATE INDEX idx_json_inference_timestamp ON tensorzero.json_inference(timestamp DESC);
CREATE INDEX idx_json_inference_tags ON tensorzero.json_inference USING GIN(tags);

-- Compound index for list queries with time filtering
CREATE INDEX idx_json_inference_function_ts
    ON tensorzero.json_inference(function_name, timestamp DESC);

-- Episode queries with id-based ordering
CREATE INDEX idx_json_inference_episode_id
    ON tensorzero.json_inference(episode_id, id DESC);

-- Variant statistics optimization
CREATE INDEX idx_json_inference_function_variant_ts
    ON tensorzero.json_inference(function_name, variant_name, timestamp DESC);

-- GIN indices for JSONB containment queries
CREATE INDEX idx_json_inference_input_gin
    ON tensorzero.json_inference USING GIN(input jsonb_path_ops);
CREATE INDEX idx_json_inference_output_gin
    ON tensorzero.json_inference USING GIN(output jsonb_path_ops);

-- =============================================================================
-- MODEL INFERENCE TABLE
-- =============================================================================

CREATE TABLE tensorzero.model_inference (
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
    timestamp TIMESTAMPTZ GENERATED ALWAYS AS (tensorzero.uuid_v7_to_timestamp(id)) STORED,
    system TEXT,
    input_messages JSONB,
    output JSONB,
    cached BOOLEAN NOT NULL DEFAULT FALSE,
    -- Can be CHECK constrained to: 'stop', 'length', 'tool_calls', 'content_filter', 'other'
    finish_reason TEXT,
    snapshot_hash BYTEA
);

-- Basic indexes
CREATE INDEX idx_model_inference_inference_id ON tensorzero.model_inference(inference_id);
CREATE INDEX idx_model_inference_model ON tensorzero.model_inference(model_name, model_provider_name);
CREATE INDEX idx_model_inference_timestamp ON tensorzero.model_inference(timestamp DESC);
