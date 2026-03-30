-- Stores function configuration as versioned JSONB blobs.
-- Each row is an immutable version identified by a UUIDv7.
-- The function type discriminator ("chat" or "json") lives on the row itself,
-- not inside the JSONB config blob.
CREATE TABLE IF NOT EXISTS tensorzero.function_versions (
    id UUID PRIMARY KEY,
    -- FK to the functions registry table (created in 20260206184730_config_and_deployment.sql)
    function_id UUID NOT NULL,
    -- The function type: "chat" or "json"
    function_type TEXT NOT NULL,
    -- Schema version for JSONB deserialization dispatch. Bumped on breaking schema changes.
    schema_version INT NOT NULL DEFAULT 1,
    -- The full function configuration as JSONB. Contains schemas, evaluators,
    -- experimentation, and variant references (as UUIDs pointing to variant_versions rows).
    config JSONB NOT NULL,
    -- Who created this version: "ui", "autopilot", "migration", etc.
    creation_source TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for looking up function versions by function_id (e.g. to find all versions of a function)
CREATE INDEX IF NOT EXISTS idx_function_versions_function_id ON tensorzero.function_versions (function_id);
