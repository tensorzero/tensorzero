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

# --------

HOST="clickhouse"
USER="chuser"
PASSWORD="chpassword"
DATABASE="tensorzero_ui_fixtures"

echo "Fetching all materialized views in database $DATABASE..."

echo "Fetching all materialized views in database $DATABASE..."

# First, get all materialized view names
VIEWS=$(clickhouse-client --host "$HOST" --user "$USER" --password "$PASSWORD" --database "$DATABASE" --query "
SELECT name 
FROM system.tables 
WHERE database = '$DATABASE' AND engine = 'MaterializedView'
FORMAT CSV" | tr -d '"')

if [ -z "$VIEWS" ]; then
  echo "No materialized views found in database $DATABASE"
  exit 0
fi

echo "Found materialized views. Optimizing each one..."

# For each view, extract the target table from its definition
for VIEW in $VIEWS; do
  echo "Processing view $VIEW..."
  
  # Get the view definition
  DEFINITION=$(clickhouse-client --host "$HOST" --user "$USER" --password "$PASSWORD" --database "$DATABASE" --query "
  SHOW CREATE TABLE ${VIEW}
  FORMAT TabSeparatedRaw")
  
  # Extract the target table - typically follows "TO tablename" in view definition
  TARGET_TABLE=$(echo "$DEFINITION" | grep -oP "TO\\s+\\K\\w+")
  
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
