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
  count=$(clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG \
            --database "$DATABASE_NAME" --query "SELECT count() FROM $table" 2>/dev/null || echo "ERROR")

  if [[ "$count" == "ERROR" ]]; then
    echo "  ERROR: Could not access table $table"
    cleanup_verified=0
  elif [ "$count" -eq 0 ]; then
    echo "  ✓ $table: empty ($count rows)"
  else
    echo "  ✗ $table: NOT empty ($count rows)"
    cleanup_verified=0
  fi
done

if [ $cleanup_verified -eq 1 ]; then
  echo "✓ Database cleanup verified - all tables are empty"
  exit 0
else
  echo "✗ Database cleanup failed - some tables still contain data"
  exit 1
fi
