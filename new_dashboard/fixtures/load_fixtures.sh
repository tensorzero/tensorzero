#!/bin/bash
clickhouse-client --host $CLICKHOUSE_HOST --database tensorzero --query "INSERT INTO JsonInference FORMAT CSV" < json_inference_examples.csv
clickhouse-client --host $CLICKHOUSE_HOST --database tensorzero --query "INSERT INTO BooleanMetricFeedback FORMAT CSV" < boolean_metric_feedback_examples.csv
