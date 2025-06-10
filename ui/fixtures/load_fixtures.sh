#!/bin/bash
set -euxo pipefail

DATABASE_NAME="${1:-tensorzero_ui_fixtures}"
if [ -f /load_complete.marker ]; then
  echo "Fixtures already loaded; this script will now exit with status 0"
  exit 0
fi

clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword  "SELECT * FROM system.disks;"

clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database "$DATABASE_NAME" --query "INSERT INTO JsonInference FORMAT JSONEachRow" < json_inference_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database "$DATABASE_NAME" --query "INSERT INTO BooleanMetricFeedback FORMAT JSONEachRow" < boolean_metric_feedback_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database "$DATABASE_NAME" --query "INSERT INTO FloatMetricFeedback FORMAT JSONEachRow" < float_metric_feedback_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database "$DATABASE_NAME" --query "INSERT INTO CommentFeedback FORMAT JSONEachRow" < comment_feedback_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database "$DATABASE_NAME" --query "INSERT INTO DemonstrationFeedback FORMAT JSONEachRow" < demonstration_feedback_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database "$DATABASE_NAME" --query "INSERT INTO ChatInference FORMAT JSONEachRow" < chat_inference_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database "$DATABASE_NAME" --query "INSERT INTO ModelInference FORMAT JSONEachRow" < model_inference_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database "$DATABASE_NAME" --query "INSERT INTO ChatInferenceDatapoint FORMAT JSONEachRow" < chat_inference_datapoint_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database "$DATABASE_NAME" --query "INSERT INTO JsonInferenceDatapoint FORMAT JSONEachRow" < json_inference_datapoint_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database "$DATABASE_NAME" --query "INSERT INTO ModelInferenceCache FORMAT JSONEachRow" < model_inference_cache_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database "$DATABASE_NAME" --query "INSERT INTO DynamicEvaluationRun FORMAT JSONEachRow" < dynamic_evaluation_run_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database "$DATABASE_NAME" --query "INSERT INTO DynamicEvaluationRunEpisode FORMAT JSONEachRow" < dynamic_evaluation_run_episode_examples.jsonl

# If TENSORZERO_SKIP_LARGE_FIXTURES equals 1, exit
if [ "${TENSORZERO_SKIP_LARGE_FIXTURES:-}" = "1" ]; then
    echo "TENSORZERO_SKIP_LARGE_FIXTURES is set to 1 - exiting without loading large fixtures"
    touch /tmp/load_complete.marker
    exit 0
fi

uv run python --version
uv run ./download-fixtures.py
df -h
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database "$DATABASE_NAME" --query "INSERT INTO ChatInference FROM INFILE './s3-fixtures/large_chat_inference_v2.parquet' FORMAT Parquet"
df -h
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database "$DATABASE_NAME" --query "INSERT INTO JsonInference FROM INFILE './s3-fixtures/large_json_inference_v2.parquet' FORMAT Parquet"
df -h
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database "$DATABASE_NAME" --query "INSERT INTO ModelInference FROM INFILE './s3-fixtures/large_chat_model_inference_v2.parquet' FORMAT Parquet"
df -h
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database "$DATABASE_NAME" --query "INSERT INTO ModelInference FROM INFILE './s3-fixtures/large_json_model_inference_v2.parquet' FORMAT Parquet"
df -h
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database "$DATABASE_NAME" --query "INSERT INTO BooleanMetricFeedback FROM INFILE './s3-fixtures/large_chat_boolean_feedback.parquet' FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database "$DATABASE_NAME" --query "INSERT INTO BooleanMetricFeedback FROM INFILE './s3-fixtures/large_json_boolean_feedback.parquet' FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database "$DATABASE_NAME" --query "INSERT INTO FloatMetricFeedback FROM INFILE './s3-fixtures/large_chat_float_feedback.parquet' FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database "$DATABASE_NAME" --query "INSERT INTO FloatMetricFeedback FROM INFILE './s3-fixtures/large_json_float_feedback.parquet' FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database "$DATABASE_NAME" --query "INSERT INTO CommentFeedback FROM INFILE './s3-fixtures/large_chat_comment_feedback.parquet' FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database "$DATABASE_NAME" --query "INSERT INTO CommentFeedback FROM INFILE './s3-fixtures/large_json_comment_feedback.parquet' FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database "$DATABASE_NAME" --query "INSERT INTO DemonstrationFeedback FROM INFILE './s3-fixtures/large_chat_demonstration_feedback.parquet' FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database "$DATABASE_NAME" --query "INSERT INTO DemonstrationFeedback FROM INFILE './s3-fixtures/large_json_demonstration_feedback.parquet' FORMAT Parquet"

# Give ClickHouse some time to make the writes visible
sleep 2

./check-fixtures.sh "$DATABASE_NAME"

# Create the marker file to signal completion for the healthcheck
# Write it to an ephemeral location to make sure that we don't see a marker file
# from a previous container run.
touch /load_complete.marker
echo "Fixtures loaded; this script will now exit with status 0"
