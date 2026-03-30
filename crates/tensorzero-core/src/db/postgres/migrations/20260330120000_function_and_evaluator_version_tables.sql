-- Function and evaluator version tables for config-in-database (flat tables approach).
-- Depends on: 20260327120000_variant_version_tables.sql (prompt_template_versions, variant_versions)

-- Functions registry: one row per named function.
CREATE TABLE IF NOT EXISTS tensorzero.functions (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    active_version_id UUID, -- FK added after function_versions is created
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);

-- Function versions: one row per immutable version.
CREATE TABLE IF NOT EXISTS tensorzero.function_versions (
    id UUID PRIMARY KEY, -- UUIDv7
    function_id UUID NOT NULL REFERENCES tensorzero.functions(id),
    function_type TEXT NOT NULL CHECK (function_type IN ('chat', 'json')),

    -- Schema references (to prompt_template_versions)
    system_schema_prompt_id UUID REFERENCES tensorzero.prompt_template_versions(id),
    system_schema_key TEXT,
    user_schema_prompt_id UUID REFERENCES tensorzero.prompt_template_versions(id),
    user_schema_key TEXT,
    assistant_schema_prompt_id UUID REFERENCES tensorzero.prompt_template_versions(id),
    assistant_schema_key TEXT,
    output_schema_prompt_id UUID REFERENCES tensorzero.prompt_template_versions(id), -- JSON functions only
    output_schema_key TEXT,

    -- Chat-function-only fields
    tools TEXT[] NOT NULL DEFAULT '{}',
    tool_choice TEXT NOT NULL DEFAULT 'auto',
    parallel_tool_calls BOOLEAN,

    -- Common
    description TEXT,

    creation_source TEXT NOT NULL, -- 'ui', 'autopilot'
    source_autopilot_session_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Add FK from functions.active_version_id -> function_versions.id
ALTER TABLE tensorzero.functions
    ADD CONSTRAINT fk_functions_active_version
    FOREIGN KEY (active_version_id) REFERENCES tensorzero.function_versions(id);

-- Named schemas (the `schemas` map on UninitializedFunctionConfig)
CREATE TABLE IF NOT EXISTS tensorzero.function_version_schemas (
    function_version_id UUID NOT NULL REFERENCES tensorzero.function_versions(id),
    schema_name TEXT NOT NULL,
    prompt_template_version_id UUID NOT NULL REFERENCES tensorzero.prompt_template_versions(id),
    template_key TEXT NOT NULL,
    PRIMARY KEY (function_version_id, schema_name)
);

-- Junction: which variant versions belong to a function version
CREATE TABLE IF NOT EXISTS tensorzero.function_version_variants (
    function_version_id UUID NOT NULL REFERENCES tensorzero.function_versions(id),
    variant_name TEXT NOT NULL,
    variant_version_id UUID NOT NULL REFERENCES tensorzero.variant_versions(id),
    PRIMARY KEY (function_version_id, variant_name)
);

-- ============================================================
-- EXPERIMENTATION TABLES
-- ============================================================
-- Each function version has at most one base experimentation config,
-- plus optional namespace-specific overrides.
-- Maps to UninitializedExperimentationConfigWithNamespaces {
--   base: UninitializedExperimentationConfig,
--   namespaces: HashMap<String, UninitializedExperimentationConfig>
-- }
-- Legacy types (StaticWeights, Uniform, TrackAndStop) are normalized
-- to Static/Adaptive on write.

CREATE TABLE IF NOT EXISTS tensorzero.function_version_experimentation (
    id UUID PRIMARY KEY, -- UUIDv7
    function_version_id UUID NOT NULL REFERENCES tensorzero.function_versions(id),
    namespace TEXT, -- NULL = base config; non-NULL = namespace-specific override
    experimentation_type TEXT NOT NULL CHECK (experimentation_type IN ('static', 'adaptive')),
    UNIQUE (function_version_id, namespace)
);

-- Weighted variant assignments for 'static' experimentation
CREATE TABLE IF NOT EXISTS tensorzero.experimentation_static_variants (
    experimentation_id UUID NOT NULL REFERENCES tensorzero.function_version_experimentation(id),
    variant_name TEXT NOT NULL,
    weight DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (experimentation_id, variant_name)
);

-- Fallback variants for 'static' experimentation (ordered)
CREATE TABLE IF NOT EXISTS tensorzero.experimentation_static_fallbacks (
    experimentation_id UUID NOT NULL REFERENCES tensorzero.function_version_experimentation(id),
    variant_name TEXT NOT NULL,
    position INT NOT NULL,
    PRIMARY KEY (experimentation_id, variant_name)
);

-- Adaptive experimentation config (currently only TrackAndStop algorithm)
CREATE TABLE IF NOT EXISTS tensorzero.experimentation_adaptive_configs (
    experimentation_id UUID PRIMARY KEY REFERENCES tensorzero.function_version_experimentation(id),
    algorithm TEXT NOT NULL DEFAULT 'track_and_stop' CHECK (algorithm IN ('track_and_stop')),
    metric TEXT NOT NULL,
    min_samples_per_variant BIGINT NOT NULL DEFAULT 10,
    delta DOUBLE PRECISION NOT NULL DEFAULT 0.05,
    epsilon DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    update_period_s BIGINT NOT NULL DEFAULT 300,
    min_prob DOUBLE PRECISION,
    max_samples_per_variant BIGINT
);

-- Candidate variants for adaptive experimentation (ordered)
CREATE TABLE IF NOT EXISTS tensorzero.experimentation_adaptive_candidates (
    experimentation_id UUID NOT NULL REFERENCES tensorzero.experimentation_adaptive_configs(experimentation_id),
    variant_name TEXT NOT NULL,
    position INT NOT NULL,
    PRIMARY KEY (experimentation_id, variant_name)
);

-- Fallback variants for adaptive experimentation (ordered)
CREATE TABLE IF NOT EXISTS tensorzero.experimentation_adaptive_fallbacks (
    experimentation_id UUID NOT NULL REFERENCES tensorzero.experimentation_adaptive_configs(experimentation_id),
    variant_name TEXT NOT NULL,
    position INT NOT NULL,
    PRIMARY KEY (experimentation_id, variant_name)
);

-- ============================================================
-- EVALUATOR TABLES
-- ============================================================
-- Each function version has 0..N evaluators.
-- Maps to function config's `evaluators: HashMap<String, UninitializedEvaluatorConfig>`.

-- Evaluators base table (discriminated by evaluator_type)
CREATE TABLE IF NOT EXISTS tensorzero.function_version_evaluators (
    id UUID PRIMARY KEY, -- UUIDv7
    function_version_id UUID NOT NULL REFERENCES tensorzero.function_versions(id),
    evaluator_name TEXT NOT NULL,
    evaluator_type TEXT NOT NULL CHECK (evaluator_type IN ('exact_match', 'llm_judge', 'tool_use', 'regex')),
    UNIQUE (function_version_id, evaluator_name)
);

-- ExactMatch evaluator details
CREATE TABLE IF NOT EXISTS tensorzero.evaluator_exact_match_configs (
    evaluator_id UUID PRIMARY KEY REFERENCES tensorzero.function_version_evaluators(id),
    cutoff REAL -- deprecated but stored for backward compat
);

-- Regex evaluator details
CREATE TABLE IF NOT EXISTS tensorzero.evaluator_regex_configs (
    evaluator_id UUID PRIMARY KEY REFERENCES tensorzero.function_version_evaluators(id),
    must_match TEXT,
    must_not_match TEXT
);

-- ToolUse evaluator details
CREATE TABLE IF NOT EXISTS tensorzero.evaluator_tool_use_configs (
    evaluator_id UUID PRIMARY KEY REFERENCES tensorzero.function_version_evaluators(id),
    tool_use_type TEXT NOT NULL CHECK (tool_use_type IN ('none', 'none_of', 'any', 'any_of', 'all_of'))
);

-- Tools list for ToolUse evaluators (NoneOf, AnyOf, AllOf)
CREATE TABLE IF NOT EXISTS tensorzero.evaluator_tool_use_tools (
    evaluator_id UUID NOT NULL REFERENCES tensorzero.evaluator_tool_use_configs(evaluator_id),
    tool_name TEXT NOT NULL,
    PRIMARY KEY (evaluator_id, tool_name)
);

-- LLM Judge evaluator details
CREATE TABLE IF NOT EXISTS tensorzero.evaluator_llm_judge_configs (
    evaluator_id UUID PRIMARY KEY REFERENCES tensorzero.function_version_evaluators(id),
    input_format TEXT NOT NULL DEFAULT 'serialized' CHECK (input_format IN ('serialized', 'messages')),
    output_type TEXT NOT NULL CHECK (output_type IN ('float', 'boolean')),
    optimize TEXT NOT NULL CHECK (optimize IN ('min', 'max')),
    include_reference_output BOOLEAN NOT NULL DEFAULT false,
    cutoff REAL, -- deprecated
    description TEXT
);

-- LLM Judge variant configs (base table for all judge variant types)
CREATE TABLE IF NOT EXISTS tensorzero.evaluator_llm_judge_variants (
    id UUID PRIMARY KEY, -- UUIDv7
    evaluator_id UUID NOT NULL REFERENCES tensorzero.evaluator_llm_judge_configs(evaluator_id),
    variant_name TEXT NOT NULL,
    variant_type TEXT NOT NULL DEFAULT 'chat_completion' CHECK (variant_type IN (
        'chat_completion', 'experimental_best_of_n_sampling',
        'experimental_mixture_of_n', 'experimental_dynamic_in_context_learning',
        'experimental_chain_of_thought'
    )),
    timeouts_non_streaming_total_ms BIGINT,
    timeouts_streaming_ttft_ms BIGINT,
    timeouts_streaming_total_ms BIGINT,
    UNIQUE (evaluator_id, variant_name)
);

-- LLM Judge ChatCompletion variant details
CREATE TABLE IF NOT EXISTS tensorzero.evaluator_llm_judge_cc_configs (
    judge_variant_id UUID PRIMARY KEY REFERENCES tensorzero.evaluator_llm_judge_variants(id),
    active BOOLEAN DEFAULT true,
    model TEXT NOT NULL,
    system_instructions_prompt_id UUID NOT NULL REFERENCES tensorzero.prompt_template_versions(id),
    system_instructions_key TEXT NOT NULL,
    temperature REAL,
    top_p REAL,
    max_tokens INT,
    presence_penalty REAL,
    frequency_penalty REAL,
    seed INT,
    json_mode TEXT NOT NULL, -- required for LLM judge (it's a JSON function)
    stop_sequences TEXT[],
    reasoning_effort TEXT,
    service_tier TEXT,
    thinking_budget_tokens INT,
    verbosity TEXT,
    num_retries INT NOT NULL DEFAULT 0,
    max_retry_delay_s REAL NOT NULL DEFAULT 10.0
);

-- Extra body for LLM judge CC variants (same structure as variant extra_body tables)
CREATE TABLE IF NOT EXISTS tensorzero.evaluator_llm_judge_cc_extra_body (
    judge_variant_id UUID NOT NULL REFERENCES tensorzero.evaluator_llm_judge_cc_configs(judge_variant_id),
    position INT NOT NULL,
    pointer TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('value', 'delete')),
    replacement_value JSONB, -- only set when kind = 'value'
    PRIMARY KEY (judge_variant_id, position)
);

-- Extra headers for LLM judge CC variants (same structure as variant extra_headers tables)
CREATE TABLE IF NOT EXISTS tensorzero.evaluator_llm_judge_cc_extra_headers (
    judge_variant_id UUID NOT NULL REFERENCES tensorzero.evaluator_llm_judge_cc_configs(judge_variant_id),
    position INT NOT NULL,
    header_name TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('value', 'delete')),
    header_value TEXT, -- only set when kind = 'value'
    PRIMARY KEY (judge_variant_id, position)
);
