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
