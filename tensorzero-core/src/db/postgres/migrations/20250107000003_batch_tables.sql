-- =============================================================================
-- Migration: Batch Processing Tables
-- Description: Creates batch_request and batch_model_inference tables
--              for batch inference processing.
-- =============================================================================

-- =============================================================================
-- ENUM TYPES
-- =============================================================================

CREATE TYPE batch_status_enum AS ENUM ('pending', 'completed', 'failed');

-- =============================================================================
-- BATCH REQUEST
-- =============================================================================

CREATE TABLE batch_request (
    batch_id UUID NOT NULL,
    id UUID PRIMARY KEY,
    batch_params JSONB,
    status batch_status_enum NOT NULL DEFAULT 'pending',
    errors JSONB,
    model_name TEXT NOT NULL,
    model_provider_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    raw_request TEXT,
    raw_response TEXT,
    function_name TEXT,
    variant_name TEXT,
    snapshot_hash BYTEA
);

CREATE INDEX idx_batch_request_batch_id ON batch_request(batch_id);
CREATE INDEX idx_batch_request_status ON batch_request(status);
CREATE INDEX idx_batch_request_created_at ON batch_request(created_at DESC);

-- =============================================================================
-- BATCH MODEL INFERENCE
-- =============================================================================

CREATE TABLE batch_model_inference (
    inference_id UUID PRIMARY KEY,
    batch_id UUID NOT NULL,
    function_name TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    episode_id UUID NOT NULL,
    input JSONB NOT NULL,
    input_messages JSONB,
    system TEXT,
    tool_params TEXT,
    inference_params JSONB,
    output_schema TEXT,
    tags JSONB NOT NULL DEFAULT '{}',
    extra_body JSONB,
    dynamic_tools JSONB NOT NULL DEFAULT '[]',
    dynamic_provider_tools JSONB NOT NULL DEFAULT '[]',
    allowed_tools TEXT,
    tool_choice TEXT,
    parallel_tool_calls BOOLEAN,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    snapshot_hash BYTEA
);

CREATE INDEX idx_batch_model_inference_batch_id ON batch_model_inference(batch_id);
CREATE INDEX idx_batch_model_inference_function ON batch_model_inference(function_name, variant_name);
CREATE INDEX idx_batch_model_inference_episode ON batch_model_inference(episode_id);
