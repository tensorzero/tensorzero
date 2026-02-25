-- Inference evaluation runs metadata table.
-- This table stores run-level metadata so we can query evaluation runs directly
-- without scanning inference tags across full inference tables.

CREATE TABLE tensorzero.inference_evaluation_runs (
    run_id UUID PRIMARY KEY,
    evaluation_name TEXT NOT NULL,
    function_name TEXT NOT NULL,
    function_type TEXT NOT NULL CHECK (function_type IN ('chat', 'json')),
    dataset_name TEXT NOT NULL,
    variant_names JSONB NOT NULL DEFAULT '[]'::JSONB,
    metrics JSONB NOT NULL DEFAULT '[]'::JSONB,
    source TEXT NOT NULL CHECK (source IN ('dataset_name', 'datapoint_ids')),
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

-- Best-effort backfill from existing evaluation-tagged inferences and feedback.
-- `metrics.optimize` is unknown for historical runs and is therefore set to NULL.
WITH evaluation_inferences AS (
    SELECT
        CASE
            WHEN (tags->>'tensorzero::evaluation_run_id') ~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
                THEN (tags->>'tensorzero::evaluation_run_id')::UUID
            ELSE NULL
        END AS run_id,
        tags->>'tensorzero::evaluation_name' AS evaluation_name,
        tags->>'tensorzero::dataset_name' AS dataset_name,
        function_name,
        'chat'::TEXT AS function_type,
        variant_name,
        snapshot_hash,
        created_at,
        id AS inference_id
    FROM tensorzero.chat_inferences
    WHERE
        tags ? 'tensorzero::evaluation_run_id'
        AND tags ? 'tensorzero::evaluation_name'
        AND tags ? 'tensorzero::dataset_name'
        AND NOT function_name LIKE 'tensorzero::%'

    UNION ALL

    SELECT
        CASE
            WHEN (tags->>'tensorzero::evaluation_run_id') ~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
                THEN (tags->>'tensorzero::evaluation_run_id')::UUID
            ELSE NULL
        END AS run_id,
        tags->>'tensorzero::evaluation_name' AS evaluation_name,
        tags->>'tensorzero::dataset_name' AS dataset_name,
        function_name,
        'json'::TEXT AS function_type,
        variant_name,
        snapshot_hash,
        created_at,
        id AS inference_id
    FROM tensorzero.json_inferences
    WHERE
        tags ? 'tensorzero::evaluation_run_id'
        AND tags ? 'tensorzero::evaluation_name'
        AND tags ? 'tensorzero::dataset_name'
        AND NOT function_name LIKE 'tensorzero::%'
),
filtered_runs AS (
    SELECT
        run_id,
        MIN(evaluation_name) AS evaluation_name,
        MIN(function_name) AS function_name,
        MIN(function_type) AS function_type,
        MIN(dataset_name) AS dataset_name,
        to_jsonb(ARRAY_AGG(DISTINCT variant_name ORDER BY variant_name)) AS variant_names,
        MIN(snapshot_hash) AS snapshot_hash,
        MIN(created_at) AS created_at
    FROM evaluation_inferences
    WHERE
        run_id IS NOT NULL
        AND COALESCE(evaluation_name, '') <> ''
        AND COALESCE(dataset_name, '') <> ''
    GROUP BY run_id
),
metrics_flat AS (
    SELECT
        ei.run_id,
        b.metric_name,
        'boolean'::TEXT AS value_type
    FROM tensorzero.boolean_metric_feedback b
    INNER JOIN evaluation_inferences ei ON ei.inference_id = b.target_id
    WHERE ei.run_id IS NOT NULL

    UNION ALL

    SELECT
        ei.run_id,
        f.metric_name,
        'float'::TEXT AS value_type
    FROM tensorzero.float_metric_feedback f
    INNER JOIN evaluation_inferences ei ON ei.inference_id = f.target_id
    WHERE ei.run_id IS NOT NULL
),
metrics_by_run AS (
    SELECT
        run_id,
        JSONB_AGG(
            JSONB_BUILD_OBJECT(
                'name', metric_name,
                'evaluator_name', CASE
                    WHEN POSITION('::evaluator_name::' IN metric_name) > 0
                        THEN SPLIT_PART(metric_name, '::evaluator_name::', 2)
                    ELSE NULL
                END,
                'value_type', value_type,
                'optimize', NULL
            )
            ORDER BY metric_name
        ) AS metrics
    FROM (
        SELECT DISTINCT run_id, metric_name, value_type
        FROM metrics_flat
    ) deduped
    GROUP BY run_id
)
INSERT INTO tensorzero.inference_evaluation_runs (
    run_id,
    evaluation_name,
    function_name,
    function_type,
    dataset_name,
    variant_names,
    metrics,
    source,
    snapshot_hash,
    created_at
)
SELECT
    fr.run_id,
    fr.evaluation_name,
    fr.function_name,
    fr.function_type,
    fr.dataset_name,
    fr.variant_names,
    COALESCE(mbr.metrics, '[]'::JSONB) AS metrics,
    'dataset_name' AS source,
    fr.snapshot_hash,
    fr.created_at
FROM filtered_runs fr
LEFT JOIN metrics_by_run mbr ON mbr.run_id = fr.run_id
ON CONFLICT (run_id) DO UPDATE SET
    evaluation_name = EXCLUDED.evaluation_name,
    function_name = EXCLUDED.function_name,
    function_type = EXCLUDED.function_type,
    dataset_name = EXCLUDED.dataset_name,
    variant_names = EXCLUDED.variant_names,
    metrics = EXCLUDED.metrics,
    source = EXCLUDED.source,
    snapshot_hash = EXCLUDED.snapshot_hash,
    updated_at = NOW();
