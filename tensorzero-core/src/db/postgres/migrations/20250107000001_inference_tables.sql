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
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tags JSONB NOT NULL DEFAULT '{}',
    extra_body JSONB,
    ttft_ms INTEGER,
    dynamic_tools TEXT[] DEFAULT '{}',
    dynamic_provider_tools TEXT[] DEFAULT '{}',
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
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
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
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
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
