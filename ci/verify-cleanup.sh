#!/bin/bash
set -euo pipefail

DATABASE_NAME="${1:-tensorzero_ui_fixtures}"

echo "Verifying database cleanup for $DATABASE_NAME..."

# Determine credentials based on environment
if command -v buildkite-agent >/dev/null 2>&1; then
  # Running on Buildkite - use secrets (fail if not available)
  if [ "${TENSORZERO_CLICKHOUSE_FAST_CHANNEL:-}" = "1" ]; then
    CLICKHOUSE_HOST_VAR=$(buildkite-agent secret get CLICKHOUSE_HOST_FAST_CHANNEL)
  else
    CLICKHOUSE_HOST_VAR=$(buildkite-agent secret get CLICKHOUSE_HOST)
  fi
  CLICKHOUSE_USER_VAR=$(buildkite-agent secret get CLICKHOUSE_CLOUD_INSERT_USERNAME)
  CLICKHOUSE_PASSWORD_VAR=$(buildkite-agent secret get CLICKHOUSE_CLOUD_INSERT_PASSWORD)
  CLICKHOUSE_SECURE_FLAG="--secure"
else
  # Not on Buildkite - use environment variables with defaults
  CLICKHOUSE_HOST_VAR="${CLICKHOUSE_HOST}"
  CLICKHOUSE_USER_VAR="${CLICKHOUSE_USER:-chuser}"
  CLICKHOUSE_PASSWORD_VAR="${CLICKHOUSE_PASSWORD:-chpassword}"
  CLICKHOUSE_SECURE_FLAG=""
fi

# Test database connection
echo "Testing database connection..."
error_file=$(mktemp)
connection_test=$(clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG \
                   --query "SELECT 1" 2>"$error_file" || echo "ERROR")

if [[ "$connection_test" == "ERROR" ]]; then
  echo "✗ Failed to connect to ClickHouse database"
  echo "  Host: $CLICKHOUSE_HOST_VAR"
  echo "  User: $CLICKHOUSE_USER_VAR"
  echo "  Database: $DATABASE_NAME"
  echo "  Error details:"
  sed 's/^/    /' "$error_file"
  rm -f "$error_file"
  exit 1
elif [[ "$connection_test" == "1" ]]; then
  echo "✓ Database connection successful"
  rm -f "$error_file"
else
  echo "✗ Unexpected response from database connection test: $connection_test"
  if [ -s "$error_file" ]; then
    echo "  Error details:"
    sed 's/^/    /' "$error_file"
  fi
  rm -f "$error_file"
  exit 1
fi

# Check if database exists first
echo "Checking if database $DATABASE_NAME exists..."
error_file=$(mktemp)
database_exists=$(clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG \
                   --query "SELECT count() FROM system.databases WHERE name = '$DATABASE_NAME'" 2>"$error_file" || echo "ERROR")

if [[ "$database_exists" == "ERROR" ]]; then
  echo "✗ Failed to check if database exists"
  echo "  Error details:"
  sed 's/^/    /' "$error_file"
  rm -f "$error_file"
  exit 1
elif [ "$database_exists" -eq 0 ]; then
  echo "✓ Database $DATABASE_NAME does not exist (acceptable for cleanup)"
  rm -f "$error_file"
  exit 0
else
  echo "✓ Database $DATABASE_NAME exists, checking tables..."
  rm -f "$error_file"
fi

# Define all tables that should be empty after cleanup
tables=(
  "JsonInference"
  "BooleanMetricFeedback"
  "FloatMetricFeedback"
  "CommentFeedback"
  "DemonstrationFeedback"
  "ChatInference"
  "ModelInference"
  "ChatInferenceDatapoint"
  "JsonInferenceDatapoint"
  "DynamicEvaluationRun"
  "DynamicEvaluationRunEpisode"
  "ModelInferenceCache"
)

cleanup_verified=1

for table in "${tables[@]}"; do
  # First check if table exists
  error_file=$(mktemp)
  table_exists=$(clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG \
                  --database "$DATABASE_NAME" --query "SELECT count() FROM system.tables WHERE database = '$DATABASE_NAME' AND name = '$table'" 2>"$error_file" || echo "ERROR")

  if [[ "$table_exists" == "ERROR" ]]; then
    echo "  ERROR: Could not check if table $table exists"
    echo "    Error details:"
    sed 's/^/      /' "$error_file"
    cleanup_verified=0
  elif [ "$table_exists" -eq 0 ]; then
    echo "  ✓ $table: does not exist (acceptable for cleanup)"
  else
    # Table exists, check if it's empty
    count_error_file=$(mktemp)
    count=$(clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG \
              --database "$DATABASE_NAME" --query "SELECT count() FROM $table" 2>"$count_error_file" || echo "ERROR")

    if [[ "$count" == "ERROR" ]]; then
      echo "  ERROR: Could not access table $table"
      echo "    Error details:"
      sed 's/^/      /' "$count_error_file"
      cleanup_verified=0
    elif [ "$count" -eq 0 ]; then
      echo "  ✓ $table: empty ($count rows)"
    else
      echo "  ✗ $table: NOT empty ($count rows)"
      cleanup_verified=0
    fi
    rm -f "$count_error_file"
  fi
  rm -f "$error_file"
done

if [ $cleanup_verified -eq 1 ]; then
  echo "✓ Database cleanup verified - all tables are empty"
  exit 0
else
  echo "✗ Database cleanup failed - some tables still contain data"
  exit 1
fi
