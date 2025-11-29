#!/bin/bash
set -euo pipefail

DATABASE_NAME="${1:-tensorzero_ui_fixtures}"
if [ -f /load_complete.marker ]; then
  echo "Fixtures already loaded; this script will now exit with status 0"
  exit 0
fi

CLICKHOUSE_HOST_VAR="${CLICKHOUSE_HOST}"
# Determine credentials based on environment
if command -v buildkite-agent >/dev/null 2>&1; then
  # Running on Buildkite - use secrets (fail if not available)
  CLICKHOUSE_USER_VAR=$(buildkite-agent secret get CLICKHOUSE_CLOUD_INSERT_USERNAME)
  CLICKHOUSE_PASSWORD_VAR=$(buildkite-agent secret get CLICKHOUSE_CLOUD_INSERT_PASSWORD)
  CLICKHOUSE_SECURE_FLAG="--secure"
else
  # Not on Buildkite - use environment variables with defaults
  CLICKHOUSE_USER_VAR="${CLICKHOUSE_USER:-chuser}"
  CLICKHOUSE_PASSWORD_VAR="${CLICKHOUSE_PASSWORD:-chpassword}"
  CLICKHOUSE_SECURE_FLAG=""
fi

# Truncate all tables before inserting new data
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "TRUNCATE TABLE JsonInference"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "TRUNCATE TABLE BooleanMetricFeedback"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "TRUNCATE TABLE FloatMetricFeedback"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "TRUNCATE TABLE CommentFeedback"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "TRUNCATE TABLE DemonstrationFeedback"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "TRUNCATE TABLE ChatInference"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "TRUNCATE TABLE ModelInference"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "TRUNCATE TABLE ChatInferenceDatapoint"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "TRUNCATE TABLE JsonInferenceDatapoint"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "TRUNCATE TABLE DynamicEvaluationRun"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "TRUNCATE TABLE DynamicEvaluationRunEpisode"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "TRUNCATE TABLE ModelInferenceCache"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "TRUNCATE TABLE DeploymentID"




# Truncate all tables first to ensure clean loading
echo "Truncating all tables before loading fixtures..."
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "TRUNCATE TABLE JsonInference"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "TRUNCATE TABLE BooleanMetricFeedback"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "TRUNCATE TABLE FloatMetricFeedback"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "TRUNCATE TABLE CommentFeedback"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "TRUNCATE TABLE DemonstrationFeedback"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "TRUNCATE TABLE ChatInference"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "TRUNCATE TABLE ModelInference"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "TRUNCATE TABLE ChatInferenceDatapoint"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "TRUNCATE TABLE JsonInferenceDatapoint"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "TRUNCATE TABLE DynamicEvaluationRun"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "TRUNCATE TABLE DynamicEvaluationRunEpisode"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "TRUNCATE TABLE ModelInferenceCache"

clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO JsonInference FORMAT JSONEachRow" < json_inference_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO ChatInference FORMAT JSONEachRow" < chat_inference_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO BooleanMetricFeedback FORMAT JSONEachRow" < boolean_metric_feedback_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO FloatMetricFeedback FORMAT JSONEachRow" < float_metric_feedback_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO FloatMetricFeedback FORMAT JSONEachRow" < jaro_winkler_similarity_feedback.jsonl
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO CommentFeedback FORMAT JSONEachRow" < comment_feedback_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO DemonstrationFeedback FORMAT JSONEachRow" < demonstration_feedback_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO ModelInference FORMAT JSONEachRow" < model_inference_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO ChatInferenceDatapoint FORMAT JSONEachRow" < chat_inference_datapoint_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO JsonInferenceDatapoint FORMAT JSONEachRow" < json_inference_datapoint_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO DynamicEvaluationRun FORMAT JSONEachRow" < dynamic_evaluation_run_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO DynamicEvaluationRunEpisode FORMAT JSONEachRow" < dynamic_evaluation_run_episode_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO ModelInferenceCache FORMAT JSONEachRow" < model_inference_cache_e2e.jsonl
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO DeploymentID VALUES ('fixture', 0, 0, 4294967295)"

# If TENSORZERO_SKIP_LARGE_FIXTURES equals 1, exit
if [ "${TENSORZERO_SKIP_LARGE_FIXTURES:-}" = "1" ]; then
    echo "TENSORZERO_SKIP_LARGE_FIXTURES is set to 1 - exiting without loading large fixtures"
    touch /load_complete.marker
    exit 0
fi

uv run python --version
uv run ./download-fixtures.py
df -h
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO ChatInference FROM INFILE './s3-fixtures/large_chat_inference_v2.parquet' SETTINGS input_format_parquet_use_native_reader_v3=0 FORMAT Parquet"
df -h
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO JsonInference FROM INFILE './s3-fixtures/large_json_inference_v2.parquet' SETTINGS input_format_parquet_use_native_reader_v3=0 FORMAT Parquet"
df -h
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO ModelInference FROM INFILE './s3-fixtures/large_chat_model_inference_v2.parquet' SETTINGS input_format_parquet_use_native_reader_v3=0 FORMAT Parquet"
df -h
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO ModelInference FROM INFILE './s3-fixtures/large_json_model_inference_v2.parquet' SETTINGS input_format_parquet_use_native_reader_v3=0 FORMAT Parquet"
df -h
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO BooleanMetricFeedback FROM INFILE './s3-fixtures/large_chat_boolean_feedback.parquet' SETTINGS input_format_parquet_use_native_reader_v3=0 FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO BooleanMetricFeedback FROM INFILE './s3-fixtures/large_json_boolean_feedback.parquet' SETTINGS input_format_parquet_use_native_reader_v3=0 FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO FloatMetricFeedback FROM INFILE './s3-fixtures/large_chat_float_feedback.parquet' SETTINGS input_format_parquet_use_native_reader_v3=0 FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO FloatMetricFeedback FROM INFILE './s3-fixtures/large_json_float_feedback.parquet' SETTINGS input_format_parquet_use_native_reader_v3=0 FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO CommentFeedback FROM INFILE './s3-fixtures/large_chat_comment_feedback.parquet' SETTINGS input_format_parquet_use_native_reader_v3=0 FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO CommentFeedback FROM INFILE './s3-fixtures/large_json_comment_feedback.parquet' SETTINGS input_format_parquet_use_native_reader_v3=0 FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO DemonstrationFeedback FROM INFILE './s3-fixtures/large_chat_demonstration_feedback.parquet' SETTINGS input_format_parquet_use_native_reader_v3=0 FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG --database "$DATABASE_NAME" --query "INSERT INTO DemonstrationFeedback FROM INFILE './s3-fixtures/large_json_demonstration_feedback.parquet' SETTINGS input_format_parquet_use_native_reader_v3=0 FORMAT Parquet"
# Give ClickHouse some time to make the writes visible
sleep 2

./check-fixtures.sh "$DATABASE_NAME"

# Create the marker file to signal completion for the healthcheck
# Write it to an ephemeral location to make sure that we don't see a marker file
# from a previous container run.
touch /load_complete.marker
echo "Fixtures loaded; this script will now exit with status 0"
