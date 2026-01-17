-- =============================================================================
-- Migration: Cache and Embeddings Tables
-- Description: Creates model_inference_cache and dynamic_in_context_learning_example tables.
--              Requires pgvector extension for embeddings.
-- =============================================================================

-- =============================================================================
-- PGVECTOR EXTENSION
-- Note: This requires pgvector to be installed on the PostgreSQL server.
-- If pgvector is not available, comment out this line and the embedding-related
-- columns/indexes below.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- MODEL INFERENCE CACHE
-- TODO: Rethink whether Postgres is the right choice for caching.
--       Consider Redis or another caching layer for production.
-- =============================================================================

CREATE TABLE model_inference_cache (
    short_cache_key BIGINT NOT NULL,
    long_cache_key BYTEA NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    output JSONB,
    raw_request TEXT,
    raw_response TEXT,
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    input_tokens INTEGER,
    output_tokens INTEGER,
    finish_reason finish_reason_enum,
    PRIMARY KEY (short_cache_key, long_cache_key)
);

CREATE INDEX idx_cache_timestamp ON model_inference_cache(timestamp DESC);
CREATE INDEX idx_cache_not_deleted ON model_inference_cache(short_cache_key, long_cache_key)
    WHERE NOT is_deleted;

-- =============================================================================
-- DYNAMIC IN-CONTEXT LEARNING EXAMPLE
-- Stores examples with embeddings for dynamic few-shot learning.
-- =============================================================================

CREATE TABLE dynamic_in_context_learning_example (
    id UUID PRIMARY KEY,
    function_name TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    namespace TEXT NOT NULL,
    input JSONB NOT NULL,
    output JSONB NOT NULL,
    embedding VECTOR,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for lookup by function/variant/namespace
CREATE INDEX idx_icl_function_variant_namespace ON dynamic_in_context_learning_example(function_name, variant_name, namespace);

-- Index for similarity search (if using pgvector)
-- Note: ivfflat requires training data, so this index may need to be created
-- after data is loaded. For small datasets, consider using hnsw instead.
-- Uncomment and customize based on your embedding dimension:
-- CREATE INDEX idx_icl_embedding ON dynamic_in_context_learning_example
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
