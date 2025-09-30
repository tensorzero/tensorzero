#!/bin/bash
set -e

# Check if TENSORZERO_CLICKHOUSE_URL environment variable is set
if [ -z "$TENSORZERO_CLICKHOUSE_URL" ]; then
  echo "Error: TENSORZERO_CLICKHOUSE_URL environment variable is not set."
  exit 1
fi

# Extract the base URL without path from TENSORZERO_CLICKHOUSE_URL
BASE_URL=$(echo "$TENSORZERO_CLICKHOUSE_URL" | sed -E 's#(https?://[^/]+).*#\1#')
# Extract the base URL without path and credentials from TENSORZERO_CLICKHOUSE_URL
# This is used for display purposes only
DISPLAY_BASE_URL=$(
  echo "$TENSORZERO_CLICKHOUSE_URL" |
  sed -E 's#(https?://)([^@/]*@)?([^/?]+).*#\1\3#'
)


# Run a simple SELECT 1 query to check if ClickHouse is healthy and credentials are valid
# This is better than /ping because /ping doesn't test authentication
echo "Checking ClickHouse connectivity at $DISPLAY_BASE_URL..."
if ! curl -s --connect-timeout 5 -X POST -d "SELECT 1" "$TENSORZERO_CLICKHOUSE_URL" > /dev/null; then
  echo "Error: Failed to connect to ClickHouse at $DISPLAY_BASE_URL"
  exit 1
fi

# Check if evaluations binary is available and executable
if ! command -v evaluations &> /dev/null; then
  echo "Error: 'evaluations' command not found. Make sure it's properly installed."
  exit 1
fi

# Make sure `evaluations -h` runs successfully
if ! evaluations -h &> /dev/null; then
  echo "Error: 'evaluations' help command failed to run."
  exit 1
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --default-config)
      export TENSORZERO_UI_DEFAULT_CONFIG=1
      shift
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

cd /app/ui

# Launch React Router
exec ./node_modules/.bin/react-router-serve ./build/server/index.js
