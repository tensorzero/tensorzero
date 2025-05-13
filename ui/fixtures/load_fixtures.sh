#!/bin/bash
set -euxo pipefail

clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword "SELECT * FROM system.disks;"

clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO JsonInference FORMAT JSONEachRow" < json_inference_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO BooleanMetricFeedback FORMAT JSONEachRow" < boolean_metric_feedback_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO FloatMetricFeedback FORMAT JSONEachRow" < float_metric_feedback_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO CommentFeedback FORMAT JSONEachRow" < comment_feedback_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO DemonstrationFeedback FORMAT JSONEachRow" < demonstration_feedback_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO ChatInference FORMAT JSONEachRow" < chat_inference_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO ModelInference FORMAT JSONEachRow" < model_inference_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO ChatInferenceDatapoint FORMAT JSONEachRow" < chat_inference_datapoint_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO JsonInferenceDatapoint FORMAT JSONEachRow" < json_inference_datapoint_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO ModelInferenceCache FORMAT JSONEachRow" < model_inference_cache_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO DynamicEvaluationRun FORMAT JSONEachRow" < dynamic_evaluation_run_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO DynamicEvaluationRunEpisode FORMAT JSONEachRow" < dynamic_evaluation_run_episode_examples.jsonl

uv run ./download-fixtures.py
df -h
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO ChatInference FROM INFILE './s3-fixtures/large_chat_inference.parquet' FORMAT Parquet"
df -h
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO JsonInference FROM INFILE './s3-fixtures/large_json_inference.parquet' FORMAT Parquet"
df -h
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "OPTIMIZE TABLE ChatInference"
df -h
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "OPTIMIZE TABLE JsonInference"
df -h
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO ModelInference FROM INFILE './s3-fixtures/large_model_inference.parquet' FORMAT Parquet"
df -h
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO ModelInference FROM INFILE './s3-fixtures/large_json_model_inference.parquet' FORMAT Parquet"
df -h

# Give ClickHouse some time to make the writes visible
sleep 2

./check-fixtures.sh

# Create the marker file to signal completion for the healthcheck
# Write it to an ephemeral location to make sure that we don't see a marker file
# from a previous container run.
touch /tmp/load_complete.marker
echo "Fixtures loaded; this script will now exit with status 0"
