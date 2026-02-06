-- Migration: DICL (Dynamic In-Context Learning) Examples Table
--
-- This table stores examples for dynamic in-context learning variants.
-- Uses pgvector for efficient similarity search on embeddings.

-- Enable pgvector extension (must be available in the Postgres instance)
CREATE EXTENSION IF NOT EXISTS vector;

-- DICL examples table
CREATE TABLE tensorzero.dicl_examples (
    id UUID PRIMARY KEY,
    function_name TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    namespace TEXT NOT NULL DEFAULT '',
    input TEXT NOT NULL,
    output TEXT NOT NULL,
    embedding vector NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for filtering by function/variant/namespace before similarity search
CREATE INDEX idx_dicl_examples_function_variant_namespace
    ON tensorzero.dicl_examples(function_name, variant_name, namespace);

-- Index for retention cleanup
CREATE INDEX idx_dicl_examples_created_at ON tensorzero.dicl_examples(created_at);

-- Note: Vector similarity index should be created AFTER initial data load for better performance.
-- The index type and parameters depend on the embedding model dimension and data volume:
--
-- For smaller datasets (<1M rows), consider HNSW:
--   CREATE INDEX idx_dicl_examples_embedding ON tensorzero.dicl_examples
--       USING hnsw (embedding vector_cosine_ops);
--
-- For larger datasets, consider IVFFlat with appropriate lists parameter:
--   CREATE INDEX idx_dicl_examples_embedding ON tensorzero.dicl_examples
--       USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
--
-- Rule of thumb for IVFFlat: lists = sqrt(row_count)
