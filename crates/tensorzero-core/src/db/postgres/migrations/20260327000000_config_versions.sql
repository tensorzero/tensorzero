-- Stores the raw content of prompt templates (system_template, user_template, etc.)
-- Each row is an immutable version identified by a UUIDv7.
CREATE TABLE IF NOT EXISTS tensorzero.prompt_template_versions (
    id UUID PRIMARY KEY,
    -- A human-readable key for the template (e.g. the original file path or a synthetic key like "tensorzero::llm_judge/system")
    template_key TEXT NOT NULL,
    -- The raw template content
    source_body TEXT NOT NULL,
    -- Who created this version: "ui", "autopilot", "migration", etc.
    creation_source TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Tracks dependencies between prompt template versions (e.g. when one template includes another).
-- Not used in the initial variant-only scope, but the table is created now to avoid a future migration.
CREATE TABLE IF NOT EXISTS tensorzero.prompt_template_version_dependencies (
    parent_id UUID NOT NULL REFERENCES tensorzero.prompt_template_versions(id),
    child_id UUID NOT NULL REFERENCES tensorzero.prompt_template_versions(id),
    PRIMARY KEY (parent_id, child_id)
);

-- Stores variant configuration as versioned JSONB blobs.
-- Each row is an immutable version identified by a UUIDv7.
-- The variant type discriminator ("type") lives inside the JSONB config itself,
-- deserialized via serde's adjacently-tagged enum (`#[serde(tag = "type", content = "config")]`).
--
-- Function config stays on disk (TOML). The TOML references variants by name;
-- on startup the gateway loads the latest variant_versions row for each name
-- and rehydrates it into UninitializedVariantInfo.
CREATE TABLE IF NOT EXISTS tensorzero.variant_versions (
    id UUID PRIMARY KEY,
    -- The variant name (e.g. "gpt4o_variant")
    variant_name TEXT NOT NULL,
    -- Schema version for JSONB deserialization dispatch. Bumped on breaking schema changes.
    schema_version INT NOT NULL DEFAULT 1,
    -- The full variant configuration as JSONB. Contains "type" (variant discriminator),
    -- "config" (variant-specific fields), and optional "timeouts"/"namespace" fields.
    config JSONB NOT NULL,
    -- Who created this version: "ui", "autopilot", "migration", etc.
    creation_source TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for looking up variant versions by name (e.g. to find all versions of a given variant)
CREATE INDEX IF NOT EXISTS idx_variant_versions_variant_name ON tensorzero.variant_versions (variant_name);
