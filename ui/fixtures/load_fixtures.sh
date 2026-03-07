#!/bin/bash
set -euxo pipefail

DATABASE_NAME="${1:-tensorzero_ui_fixtures}"
if [ -f /load_complete.marker ]; then
  echo "Fixtures already loaded; this script will now exit with status 0"
  exit 0
fi

CLICKHOUSE_HOST_VAR="${CLICKHOUSE_HOST}"
CLICKHOUSE_USER_VAR="${CLICKHOUSE_USER:-chuser}"
CLICKHOUSE_PASSWORD_VAR="${CLICKHOUSE_PASSWORD:-chpassword}"
CLICKHOUSE_SECURE_FLAG=""
if [ "${CLICKHOUSE_SECURE:-0}" = "1" ]; then
  CLICKHOUSE_SECURE_FLAG="--secure"
fi

# Truncate all tables before loading fixtures (dynamically query all tables, excluding views, internal .inner.* tables, migration tracking, and config snapshots)
echo "Truncating all tables before loading fixtures..."
tables=$(clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG \
    --query "SELECT name FROM system.tables WHERE database = '$DATABASE_NAME' AND engine NOT LIKE '%View%' AND name NOT LIKE '.inner%' AND name NOT IN ('TensorZeroMigration', 'ConfigSnapshot')")

# Build a single query with all TRUNCATEs for efficiency
truncate_query=""
for table in $tables; do
    truncate_query+="TRUNCATE TABLE \`$table\`; "
done

if [ -n "$truncate_query" ]; then
    echo "Truncating tables:"
    for table in $tables; do
        echo "  - $table"
    done
    clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG \
        --database "$DATABASE_NAME" --multiquery --query "$truncate_query"
fi

# Download JSONL fixtures from R2
if [ "${TENSORZERO_DOWNLOAD_FIXTURES_WITHOUT_CREDENTIALS:-}" = "1" ]; then
    uv run ./download-small-fixtures-http.py
else
    uv run ./download-small-fixtures.py
fi

clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO JsonInference FORMAT JSONEachRow" < ./small-fixtures/json_inference_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO ChatInference FORMAT JSONEachRow" < ./small-fixtures/chat_inference_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO BooleanMetricFeedback FORMAT JSONEachRow" < ./small-fixtures/boolean_metric_feedback_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO FloatMetricFeedback FORMAT JSONEachRow" < ./small-fixtures/float_metric_feedback_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO FloatMetricFeedback FORMAT JSONEachRow" < ./small-fixtures/jaro_winkler_similarity_feedback.jsonl
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO CommentFeedback FORMAT JSONEachRow" < ./small-fixtures/comment_feedback_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO DemonstrationFeedback FORMAT JSONEachRow" < ./small-fixtures/demonstration_feedback_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO ModelInference FORMAT JSONEachRow" < ./small-fixtures/model_inference_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO ChatInferenceDatapoint FORMAT JSONEachRow" < ./small-fixtures/chat_inference_datapoint_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO JsonInferenceDatapoint FORMAT JSONEachRow" < ./small-fixtures/json_inference_datapoint_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO DynamicEvaluationRun FORMAT JSONEachRow" < ./small-fixtures/dynamic_evaluation_run_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO DynamicEvaluationRunEpisode FORMAT JSONEachRow" < ./small-fixtures/dynamic_evaluation_run_episode_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO DeploymentID VALUES ('fixture', 0, 0, 4294967295)"

# Backfill InferenceEvaluationRuns from inference tags and feedback.
# This mirrors the backfill in migration_0049, which is skipped on clean_start=true (fresh databases).
echo "Backfilling InferenceEvaluationRuns from inference tags..."
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "
INSERT INTO InferenceEvaluationRuns
(
    run_id_uint,
    evaluation_name,
    function_name,
    function_type,
    dataset_name,
    variant_names,
    metrics,
    source,
    snapshot_hash,
    created_at,
    updated_at
)
WITH
evaluation_inferences AS (
    SELECT
        toUUIDOrNull(maxIf(value, key = 'tensorzero::evaluation_run_id')) AS run_id,
        maxIf(value, key = 'tensorzero::evaluation_name') AS evaluation_name,
        maxIf(value, key = 'tensorzero::dataset_name') AS dataset_name,
        any(function_name) AS function_name,
        any(variant_name) AS variant_name,
        any(snapshot_hash) AS snapshot_hash,
        inference_id,
        UUIDv7ToDateTime(inference_id) AS inference_timestamp
    FROM TagInference
    WHERE key IN ('tensorzero::evaluation_run_id', 'tensorzero::evaluation_name', 'tensorzero::dataset_name')
    GROUP BY inference_id
    HAVING
        run_id IS NOT NULL
        AND evaluation_name != ''
        AND dataset_name != ''
        AND NOT startsWith(function_name, 'tensorzero::')
),
metrics_flat AS (
    SELECT
        ei.run_id AS run_id,
        b.metric_name AS metric_name,
        'boolean' AS value_type
    FROM BooleanMetricFeedback b
    INNER JOIN evaluation_inferences ei ON b.target_id = ei.inference_id

    UNION ALL

    SELECT
        ei.run_id AS run_id,
        f.metric_name AS metric_name,
        'float' AS value_type
    FROM FloatMetricFeedback f
    INNER JOIN evaluation_inferences ei ON f.target_id = ei.inference_id
),
metrics_by_run AS (
    SELECT
        run_id,
        concat(
            '[',
            arrayStringConcat(
                groupUniqArray(
                    concat(
                        '{\"name\":', toJSONString(metric_name),
                        ',\"evaluator_name\":',
                        if(
                            position(metric_name, '::evaluator_name::') > 0,
                            toJSONString(arrayElement(splitByString('::evaluator_name::', metric_name), 2)),
                            'null'
                        ),
                        ',\"value_type\":', toJSONString(value_type),
                        ',\"optimize\":null}'
                    )
                ),
                ','
            ),
            ']'
        ) AS metrics
    FROM metrics_flat
    GROUP BY run_id
),
run_function_types AS (
    SELECT
        ei.run_id AS run_id,
        if(
            countIf(ji.id != toUUID('00000000-0000-0000-0000-000000000000')) > 0,
            'json',
            'chat'
        ) AS function_type
    FROM evaluation_inferences ei
    LEFT JOIN JsonInference ji ON ji.id = ei.inference_id
    GROUP BY ei.run_id
)
SELECT
    toUInt128(ei.run_id) AS run_id_uint,
    any(ei.evaluation_name) AS evaluation_name,
    any(ei.function_name) AS function_name,
    any(rft.function_type) AS function_type,
    any(ei.dataset_name) AS dataset_name,
    arrayDistinct(groupArray(ei.variant_name)) AS variant_names,
    coalesce(any(mbr.metrics), '[]') AS metrics,
    'dataset_name' AS source,
    if(isNull(any(ei.snapshot_hash)), NULL, lower(hex(any(ei.snapshot_hash)))) AS snapshot_hash,
    min(ei.inference_timestamp) AS created_at,
    now64(3) AS updated_at
FROM evaluation_inferences ei
LEFT JOIN metrics_by_run mbr ON ei.run_id = mbr.run_id
LEFT JOIN run_function_types rft ON ei.run_id = rft.run_id
GROUP BY ei.run_id
"

# If TENSORZERO_SKIP_LARGE_FIXTURES equals 1, exit
if [ "${TENSORZERO_SKIP_LARGE_FIXTURES:-}" = "1" ]; then
    echo "TENSORZERO_SKIP_LARGE_FIXTURES is set to 1 - exiting without loading large fixtures"
    touch /load_complete.marker
    exit 0
fi

uv run python --version
if [ "${TENSORZERO_DOWNLOAD_FIXTURES_WITHOUT_CREDENTIALS:-}" = "1" ]; then
    uv run ./download-large-fixtures-http.py
else
    uv run ./download-large-fixtures.py
fi
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO ChatInference FROM INFILE './large-fixtures/large_chat_inference_v2.native.lz4' FORMAT Native"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO JsonInference FROM INFILE './large-fixtures/large_json_inference_v2.native.lz4' FORMAT Native"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO ModelInference FROM INFILE './large-fixtures/large_chat_model_inference_v2.native.lz4' FORMAT Native"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO ModelInference FROM INFILE './large-fixtures/large_json_model_inference_v2.native.lz4' FORMAT Native"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO BooleanMetricFeedback FROM INFILE './large-fixtures/large_chat_boolean_feedback.native.lz4' FORMAT Native"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO BooleanMetricFeedback FROM INFILE './large-fixtures/large_json_boolean_feedback.native.lz4' FORMAT Native"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO FloatMetricFeedback FROM INFILE './large-fixtures/large_chat_float_feedback.native.lz4' FORMAT Native"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO FloatMetricFeedback FROM INFILE './large-fixtures/large_json_float_feedback.native.lz4' FORMAT Native"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO CommentFeedback FROM INFILE './large-fixtures/large_chat_comment_feedback.native.lz4' FORMAT Native"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO CommentFeedback FROM INFILE './large-fixtures/large_json_comment_feedback.native.lz4' FORMAT Native"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO DemonstrationFeedback FROM INFILE './large-fixtures/large_chat_demonstration_feedback.native.lz4' FORMAT Native"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO DemonstrationFeedback FROM INFILE './large-fixtures/large_json_demonstration_feedback.native.lz4' FORMAT Native"
# Give ClickHouse some time to make the writes visible
sleep 2

./check-fixtures.sh "$DATABASE_NAME"

# Create the marker file to signal completion for the healthcheck
# Write it to an ephemeral location to make sure that we don't see a marker file
# from a previous container run.
touch /load_complete.marker
echo "Fixtures loaded; this script will now exit with status 0"
