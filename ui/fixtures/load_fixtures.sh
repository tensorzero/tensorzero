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
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO JsonInferenceDatapoint FORMAT JSONEachRow" < json_inference_datapoint_examples.jsonl
clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword --database tensorzero_ui_fixtures --query "INSERT INTO ModelInferenceCache FORMAT JSONEachRow" < model_inference_cache_examples.jsonl


# Give ClickHouse some time to make the writes visible
sleep 1

./check-fixtures.sh

# -----

DATABASE="tensorzero_ui_fixtures"

echo "Fetching all materialized views in database $DATABASE..."

# Get all materialized view names
VIEWS=$(clickhouse-client --host "$HOST" --user "$USER" --password "$PASSWORD" --database "$DATABASE" --query "
SELECT name 
FROM system.tables 
WHERE database = '$DATABASE' AND engine = 'MaterializedView'
FORMAT CSV")

if [ -z "$VIEWS" ]; then
  echo "No materialized views found in database $DATABASE"
  exit 0
fi

echo "Found materialized views. Optimizing each one..."

# For each view, get its target table and optimize it
for VIEW in $VIEWS; do
  # Remove quotes if present
  VIEW=$(echo $VIEW | tr -d '"')
  
  # Get the target table for this materialized view
  TARGET_TABLE=$(clickhouse-client --host "$HOST" --user "$USER" --password "$PASSWORD" --database "$DATABASE" --query "
  SELECT table 
  FROM system.tables 
  WHERE database = '$DATABASE' AND name = '$VIEW'
  FORMAT CSV" | tr -d '"')
  
  if [ -z "$TARGET_TABLE" ]; then
    echo "Could not determine target table for view $VIEW, skipping..."
    continue
  fi
  
  echo "Optimizing target table $TARGET_TABLE for view $VIEW..."
  clickhouse-client --host "$HOST" --user "$USER" --password "$PASSWORD" --database "$DATABASE" --query "OPTIMIZE TABLE $TARGET_TABLE FINAL"
  echo "Optimization of $TARGET_TABLE complete."
done

echo "All materialized views optimized successfully."

# -----



# Create the marker file to signal completion for the healthcheck
# Write it to an ephemeral location to make sure that we don't see a marker file
# from a previous container run.
touch /tmp/load_complete.marker
echo "Fixtures loaded; this script will now exit with status 0"
