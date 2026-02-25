-- Inference evaluation runs metadata table.
-- This table stores run-level metadata so we can query evaluation runs directly
-- without scanning inference tags across full inference tables.

CREATE TABLE tensorzero.inference_evaluation_runs (
    run_id UUID PRIMARY KEY,
    evaluation_name TEXT NOT NULL,
    function_name TEXT NOT NULL,
    function_type TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    variant_names JSONB NOT NULL DEFAULT '[]'::JSONB,
    metrics JSONB NOT NULL DEFAULT '[]'::JSONB,
    source TEXT NOT NULL,
    snapshot_hash BYTEA,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (jsonb_typeof(variant_names) = 'array'),
    CHECK (jsonb_typeof(metrics) = 'array')
);

CREATE INDEX idx_inference_eval_runs_name
    ON tensorzero.inference_evaluation_runs(evaluation_name, run_id DESC);

CREATE INDEX idx_inference_eval_runs_function
    ON tensorzero.inference_evaluation_runs(function_name, run_id DESC);
