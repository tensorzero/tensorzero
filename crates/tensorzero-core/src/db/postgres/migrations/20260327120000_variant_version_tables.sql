-- Variant and prompt template version tables for config-in-database (flat tables approach).
-- These tables store immutable version rows for prompt templates and variants.

-- Prompt template versions: each row is an immutable snapshot of a template body.
CREATE TABLE IF NOT EXISTS tensorzero.prompt_template_versions (
    id UUID PRIMARY KEY, -- UUIDv7
    template_key TEXT NOT NULL,
    source_body TEXT NOT NULL,
    creation_source TEXT NOT NULL, -- 'ui', 'autopilot'
    source_autopilot_session_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Dependencies between prompt template versions (for template inheritance/composition).
CREATE TABLE IF NOT EXISTS tensorzero.prompt_template_version_dependencies (
    prompt_template_version_id UUID NOT NULL REFERENCES tensorzero.prompt_template_versions(id),
    dependency_prompt_template_version_id UUID NOT NULL REFERENCES tensorzero.prompt_template_versions(id),
    dependency_key TEXT NOT NULL,
    PRIMARY KEY (prompt_template_version_id, dependency_key)
);

-- Variant versions: common fields for all variant types.
CREATE TABLE IF NOT EXISTS tensorzero.variant_versions (
    id UUID PRIMARY KEY, -- UUIDv7
    variant_type TEXT NOT NULL CHECK (variant_type IN (
        'chat_completion', 'best_of_n_sampling', 'mixture_of_n', 'dicl', 'chain_of_thought'
    )),

    -- Identity: which function/variant this version belongs to.
    function_name TEXT,
    variant_name TEXT,

    -- Common fields from UninitializedVariantInfo wrapper
    weight DOUBLE PRECISION,
    timeouts_non_streaming_total_ms BIGINT,
    timeouts_streaming_ttft_ms BIGINT,
    timeouts_streaming_total_ms BIGINT,
    namespace TEXT,

    creation_source TEXT NOT NULL,
    source_autopilot_session_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for "latest version per (function_name, variant_name)" queries.
-- Uses UUIDv7 ordering (id DESC) to pick the most recent version.
CREATE INDEX IF NOT EXISTS idx_variant_versions_name_lookup
    ON tensorzero.variant_versions (function_name, variant_name, id DESC)
    WHERE function_name IS NOT NULL AND variant_name IS NOT NULL;

-- Chat completion variant config (also used for chain_of_thought, best_of_n evaluator, mixture_of_n fuser).
CREATE TABLE IF NOT EXISTS tensorzero.variant_chat_completion_configs (
    variant_version_id UUID PRIMARY KEY REFERENCES tensorzero.variant_versions(id),
    model TEXT NOT NULL,

    -- Legacy template references
    system_template_prompt_id UUID REFERENCES tensorzero.prompt_template_versions(id),
    system_template_key TEXT,
    user_template_prompt_id UUID REFERENCES tensorzero.prompt_template_versions(id),
    user_template_key TEXT,
    assistant_template_prompt_id UUID REFERENCES tensorzero.prompt_template_versions(id),
    assistant_template_key TEXT,

    -- Inference parameters
    temperature REAL,
    top_p REAL,
    max_tokens INT,
    presence_penalty REAL,
    frequency_penalty REAL,
    seed INT,
    stop_sequences TEXT[],
    reasoning_effort TEXT,
    service_tier TEXT,
    thinking_budget_tokens INT,
    verbosity TEXT,
    json_mode TEXT,
    num_retries INT NOT NULL DEFAULT 0,
    max_retry_delay_s REAL NOT NULL DEFAULT 10.0
    -- extra_body and extra_headers are stored in separate normalized tables
);

-- Extra body replacements for chat_completion variants.
-- Maps to ExtraBodyConfig { data: Vec<ExtraBodyReplacement> }
-- Each ExtraBodyReplacement has a `pointer` (JSON pointer string) and a `kind`:
--   Value(serde_json::Value) or Delete
CREATE TABLE IF NOT EXISTS tensorzero.variant_chat_completion_extra_body (
    variant_version_id UUID NOT NULL REFERENCES tensorzero.variant_chat_completion_configs(variant_version_id),
    position INT NOT NULL,
    pointer TEXT NOT NULL,
    -- kind: 'value' or 'delete'. If 'value', the replacement_value column holds the JSON value.
    kind TEXT NOT NULL CHECK (kind IN ('value', 'delete')),
    replacement_value JSONB, -- only set when kind = 'value'; this is a single leaf value, not a nested config structure
    PRIMARY KEY (variant_version_id, position)
);

-- Extra headers for chat_completion variants.
-- Maps to ExtraHeadersConfig { data: Vec<ExtraHeader> }
-- Each ExtraHeader has a `name` and a `kind`: Value(String) or Delete
CREATE TABLE IF NOT EXISTS tensorzero.variant_chat_completion_extra_headers (
    variant_version_id UUID NOT NULL REFERENCES tensorzero.variant_chat_completion_configs(variant_version_id),
    position INT NOT NULL,
    header_name TEXT NOT NULL,
    -- kind: 'value' or 'delete'. If 'value', header_value holds the string value.
    kind TEXT NOT NULL CHECK (kind IN ('value', 'delete')),
    header_value TEXT, -- only set when kind = 'value'
    PRIMARY KEY (variant_version_id, position)
);

-- Named templates for chat_completion variants (the `templates` HashMap).
CREATE TABLE IF NOT EXISTS tensorzero.variant_chat_completion_templates (
    variant_version_id UUID NOT NULL REFERENCES tensorzero.variant_chat_completion_configs(variant_version_id),
    template_name TEXT NOT NULL,
    prompt_template_version_id UUID NOT NULL REFERENCES tensorzero.prompt_template_versions(id),
    template_key TEXT NOT NULL,
    PRIMARY KEY (variant_version_id, template_name)
);

-- Best-of-N variant config.
-- The evaluator is stored as a separate variant_versions row (chat_completion type).
CREATE TABLE IF NOT EXISTS tensorzero.variant_best_of_n_configs (
    variant_version_id UUID PRIMARY KEY REFERENCES tensorzero.variant_versions(id),
    evaluator_variant_version_id UUID NOT NULL REFERENCES tensorzero.variant_versions(id)
);

-- Candidate variant names for best-of-N.
CREATE TABLE IF NOT EXISTS tensorzero.variant_best_of_n_candidates (
    variant_version_id UUID NOT NULL REFERENCES tensorzero.variant_best_of_n_configs(variant_version_id),
    candidate_name TEXT NOT NULL,
    position INT NOT NULL,
    PRIMARY KEY (variant_version_id, candidate_name)
);

-- Mixture-of-N variant config.
-- The fuser is stored as a separate variant_versions row (chat_completion type).
CREATE TABLE IF NOT EXISTS tensorzero.variant_mixture_of_n_configs (
    variant_version_id UUID PRIMARY KEY REFERENCES tensorzero.variant_versions(id),
    fuser_variant_version_id UUID NOT NULL REFERENCES tensorzero.variant_versions(id)
);

-- Candidate variant names for mixture-of-N.
CREATE TABLE IF NOT EXISTS tensorzero.variant_mixture_of_n_candidates (
    variant_version_id UUID NOT NULL REFERENCES tensorzero.variant_mixture_of_n_configs(variant_version_id),
    candidate_name TEXT NOT NULL,
    position INT NOT NULL,
    PRIMARY KEY (variant_version_id, candidate_name)
);

-- DICL (Dynamic In-Context Learning) variant config.
CREATE TABLE IF NOT EXISTS tensorzero.variant_dicl_configs (
    variant_version_id UUID PRIMARY KEY REFERENCES tensorzero.variant_versions(id),
    embedding_model TEXT NOT NULL,
    k INT NOT NULL,
    model TEXT NOT NULL,
    system_instructions_prompt_id UUID REFERENCES tensorzero.prompt_template_versions(id),
    system_instructions_key TEXT,
    temperature REAL,
    top_p REAL,
    stop_sequences TEXT[],
    presence_penalty REAL,
    frequency_penalty REAL,
    max_tokens INT,
    seed INT,
    reasoning_effort TEXT,
    thinking_budget_tokens INT,
    verbosity TEXT,
    json_mode TEXT,
    num_retries INT NOT NULL DEFAULT 0,
    max_retry_delay_s REAL NOT NULL DEFAULT 10.0,
    max_distance REAL
    -- extra_body and extra_headers are stored in separate normalized tables
);

-- Extra body replacements for DICL variants (same structure as chat_completion).
CREATE TABLE IF NOT EXISTS tensorzero.variant_dicl_extra_body (
    variant_version_id UUID NOT NULL REFERENCES tensorzero.variant_dicl_configs(variant_version_id),
    position INT NOT NULL,
    pointer TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('value', 'delete')),
    replacement_value JSONB, -- only set when kind = 'value'; a single leaf value
    PRIMARY KEY (variant_version_id, position)
);

-- Extra headers for DICL variants (same structure as chat_completion).
CREATE TABLE IF NOT EXISTS tensorzero.variant_dicl_extra_headers (
    variant_version_id UUID NOT NULL REFERENCES tensorzero.variant_dicl_configs(variant_version_id),
    position INT NOT NULL,
    header_name TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('value', 'delete')),
    header_value TEXT, -- only set when kind = 'value'
    PRIMARY KEY (variant_version_id, position)
);
