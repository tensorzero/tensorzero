#!/bin/bash
set -euxo pipefail

# Download the fixtures from R2
uv run ./download-fixtures.py

# Load the fixtures into ClickHouse
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO JsonInference FROM INFILE './s3-fixtures/small_json_inference.parquet' FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO BooleanMetricFeedback FROM INFILE './s3-fixtures/small_boolean_metric_feedback.parquet' FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO FloatMetricFeedback FROM INFILE './s3-fixtures/small_float_metric_feedback.parquet' FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO CommentFeedback FROM INFILE './s3-fixtures/small_comment_feedback.parquet' FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO DemonstrationFeedback FROM INFILE './s3-fixtures/small_demonstration_feedback.parquet' FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO ChatInference FROM INFILE './s3-fixtures/small_chat_inference.parquet' FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO ModelInference FROM INFILE './s3-fixtures/small_model_inference.parquet' FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO ChatInferenceDatapoint FROM INFILE './s3-fixtures/small_chat_inference_datapoint.parquet' FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO JsonInferenceDatapoint FROM INFILE './s3-fixtures/small_json_inference_datapoint.parquet' FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO ModelInferenceCache FROM INFILE './s3-fixtures/small_model_inference_cache.parquet' FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO DynamicEvaluationRun FROM INFILE './s3-fixtures/small_dynamic_evaluation_run.parquet' FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO DynamicEvaluationRunEpisode FROM INFILE './s3-fixtures/small_dynamic_evaluation_run_episode.parquet' FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO ChatInference FROM INFILE './s3-fixtures/large_chat_inference.parquet' FORMAT Parquet"
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO ModelInference FROM INFILE './s3-fixtures/large_model_inference.parquet' FORMAT Parquet"

# Give ClickHouse some time to make the writes visible
sleep 2

./check-fixtures.sh

# Create the marker file to signal completion for the healthcheck
# Write it to an ephemeral location to make sure that we don't see a marker file
# from a previous container run.
touch /tmp/load_complete.marker
echo "Fixtures loaded; this script will now exit with status 0"
