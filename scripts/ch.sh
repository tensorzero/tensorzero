#!/usr/bin/env bash
# Query ClickHouse e2e database. Usage: ./scripts/ch.sh "SELECT ..."
# Automatically appends FORMAT JSONEachRow if no FORMAT specified.
QUERY="$1"
if ! echo "$QUERY" | grep -qi "FORMAT"; then
  QUERY="$QUERY FORMAT JSONEachRow"
fi
curl -s "http://localhost:8123/?user=chuser&password=chpassword&database=tensorzero_e2e_tests" \
  --data-binary "$QUERY" | jq .
