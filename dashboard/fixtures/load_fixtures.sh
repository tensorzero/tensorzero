#!/bin/bash
set -e
clickhouse-client --host $CLICKHOUSE_HOST --database tensorzero --query "INSERT INTO JsonInference FORMAT CSV" < json_inference_examples.csv
clickhouse-client --host $CLICKHOUSE_HOST --database tensorzero --query "INSERT INTO BooleanMetricFeedback FORMAT CSV" < boolean_metric_feedback_examples.csv
echo "Fixtures loaded; this container will now exit with status 0"
