CREATE SCHEMA IF NOT EXISTS tensorzero;

CREATE TABLE IF NOT EXISTS tensorzero.function_configs (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    function_type TEXT NOT NULL,
    schema_revision INT NOT NULL,
    config JSONB NOT NULL,
    creation_source TEXT NOT NULL,
    source_autopilot_session_id UUID,
    deleted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_function_configs_name_created_at ON tensorzero.function_configs(name, created_at DESC);

CREATE TABLE IF NOT EXISTS tensorzero.variant_configs (
    id UUID PRIMARY KEY,
    function_name TEXT NOT NULL,
    variant_type TEXT NOT NULL,
    name TEXT NOT NULL,
    schema_revision INT NOT NULL,
    config JSONB NOT NULL,
    content_hash BYTEA NOT NULL,
    creation_source TEXT NOT NULL,
    source_autopilot_session_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_variant_configs_function_name ON tensorzero.variant_configs(function_name, name, created_at DESC);
CREATE INDEX idx_variant_configs_content_lookup ON tensorzero.variant_configs(function_name, name, content_hash);

CREATE TABLE IF NOT EXISTS tensorzero.prompt_template_configs (
    id UUID PRIMARY KEY,
    template_key TEXT NOT NULL,
    source_body TEXT NOT NULL,
    content_hash BYTEA NOT NULL,
    creation_source TEXT NOT NULL,
    source_autopilot_session_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_prompt_template_configs_content_lookup
    ON tensorzero.prompt_template_configs (template_key, content_hash);
