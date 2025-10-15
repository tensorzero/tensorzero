#!/usr/bin/env bash
set -euo pipefail

# Extract base URL without database path (in case TENSORZERO_CLICKHOUSE_URL includes /database_name)
BASE_URL=$(echo "$TENSORZERO_CLICKHOUSE_URL" | sed 's|/[^/]*$||')

echo "Fetching databases"
# Get all 'tensorzero_e2e_tests_migration_manager' databases older than 1 hour (to avoid deleting dbs used by currently running tests)
# We use the 'metadata_modification_time' of a known table ('ChatInference') to get a lower bound on the database age.
databases=$(curl -X POST "$BASE_URL" \
    --data-binary "select database from system.tables where name = 'ChatInference' and startsWith(database, 'tensorzero_e2e_tests_migration_manager') and metadata_modification_time < subtractHours(now(), 1);")

echo "The following databases will be deleted:"
echo "$databases"

# Delete each database
echo "$databases" | while read db; do
    if [ -n "$db" ]; then
        echo "Dropping database: $db"
        curl -X POST "${BASE_URL%/}/?param_target=$db" \
            --data-binary "DROP DATABASE IF EXISTS {target:Identifier}"
        echo "Database $db dropped."
    fi
done

echo "All matching databases have been deleted."
