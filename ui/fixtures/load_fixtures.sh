#!/bin/bash
set -e
clickhouse-client --host $CLICKHOUSE_HOST --database tensorzero --query "INSERT INTO JsonInference FORMAT CSV" < json_inference_examples.csv
clickhouse-client --host $CLICKHOUSE_HOST --database tensorzero --query "INSERT INTO BooleanMetricFeedback FORMAT CSV" < boolean_metric_feedback_examples.csv
clickhouse-client --host $CLICKHOUSE_HOST --database tensorzero --query "INSERT INTO FloatMetricFeedback FORMAT CSV" < float_metric_feedback_examples.csv
clickhouse-client --host $CLICKHOUSE_HOST --database tensorzero --query "INSERT INTO CommentFeedback FORMAT CSV" < comment_feedback_examples.csv
clickhouse-client --host $CLICKHOUSE_HOST --database tensorzero --query "INSERT INTO DemonstrationFeedback FORMAT CSV" < demonstration_feedback_examples.csv
clickhouse-client --host $CLICKHOUSE_HOST --database tensorzero --query "INSERT INTO ChatInference FORMAT CSV" < chat_inference_examples.csv
clickhouse-client --host $CLICKHOUSE_HOST --database tensorzero --query "INSERT INTO ModelInference FORMAT CSV" < model_inference_examples.csv
echo "Fixtures loaded; this container will now exit with status 0"
