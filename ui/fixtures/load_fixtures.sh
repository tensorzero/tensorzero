#!/bin/bash
set -euxo pipefail

clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO JsonInference FORMAT JSONEachRow" < json_inference_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO BooleanMetricFeedback FORMAT JSONEachRow" < boolean_metric_feedback_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO FloatMetricFeedback FORMAT JSONEachRow" < float_metric_feedback_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO CommentFeedback FORMAT JSONEachRow" < comment_feedback_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO DemonstrationFeedback FORMAT JSONEachRow" < demonstration_feedback_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO ChatInference FORMAT JSONEachRow" < chat_inference_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO ModelInference FORMAT JSONEachRow" < model_inference_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO ChatInferenceDatapoint FORMAT JSONEachRow" < chat_inference_datapoint_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO ChatInferenceDatapoint FORMAT JSONEachRow" < chat_inference_many_datasets_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO JsonInferenceDatapoint FORMAT JSONEachRow" < json_inference_datapoint_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO JsonInferenceDatapoint FORMAT JSONEachRow" < json_inference_many_datasets_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO ModelInferenceCache FORMAT JSONEachRow" < model_inference_cache_examples.jsonl


# Give ClickHouse some time to make the writes visible
sleep 1

./check-fixtures.sh

# Create the marker file to signal completion for the healthcheck
# Write it to an ephemeral location to make sure that we don't see a marker file
# from a previous container run.
touch /tmp/load_complete.marker
echo "Fixtures loaded; this script will now exit with status 0"
