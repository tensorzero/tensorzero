#!/bin/bash
set -e

# Check if TENSORZERO_CLICKHOUSE_URL environment variable is set
if [ -z "$TENSORZERO_CLICKHOUSE_URL" ]; then
  echo "Error: TENSORZERO_CLICKHOUSE_URL environment variable is not set."
  exit 1
fi

# Extract the base URL without path from TENSORZERO_CLICKHOUSE_URL
BASE_URL=$(echo "$TENSORZERO_CLICKHOUSE_URL" | sed -E 's#(https?://[^/]+).*#\1#')

# Attempt to ping ClickHouse and check for OK response
echo "Pinging ClickHouse at $BASE_URL/ping to verify connectivity..."
if ! curl -s --connect-timeout 5 "$BASE_URL/ping" > /dev/null; then
  echo "Error: Failed to connect to ClickHouse at $BASE_URL/ping"
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

pnpm run start
