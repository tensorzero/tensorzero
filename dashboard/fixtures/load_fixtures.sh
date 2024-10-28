clickhouse-client --host $CLICKHOUSE_HOST --database tensorzero --query "INSERT INTO JsonInference FORMAT CSV" < /fixtures/json_inference_examples.csv
clickhouse-client --host $CLICKHOUSE_HOST --database tensorzero --query "INSERT INTO BooleanMetricFeedback FORMAT CSV" < /fixtures/boolean_metric_feedback_examples.csv
