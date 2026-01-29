#!/usr/bin/env bash
set -euo pipefail

# Extract base URL without database path (in case TENSORZERO_CLICKHOUSE_URL includes /database_name)
BASE_URL=$(echo "$TENSORZERO_CLICKHOUSE_URL" | sed 's|/[^/]*$||')

echo "Fetching databases"
# Get all 'tensorzero*' databases older than 3 hours (to avoid deleting dbs used by currently running tests)
databases=$(curl -X POST \
    --retry 10 --retry-delay 5 --retry-max-time 300 --retry-all-errors --max-time 30 \
    "$BASE_URL" \
    --data-binary "select database from system.tables where database LIKE 'tensorzero%' group by database having max(metadata_modification_time) < subtractHours(now(), 3);")

echo "The following databases will be deleted:"
echo "$databases"

# Delete each database
echo "$databases" | while read db; do
    if [ -n "$db" ]; then
        echo "Dropping database: $db"
        curl -X POST \
            --retry 10 --retry-delay 5 --retry-max-time 300 --retry-all-errors --max-time 30 \
            "${BASE_URL%/}/?param_target=$db" \
            --data-binary "DROP DATABASE IF EXISTS {target:Identifier}"
        echo "Database $db dropped."
    fi
done

echo "All matching databases have been deleted."
