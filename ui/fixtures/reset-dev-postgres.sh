#!/bin/bash
set -e

# Development database reset script
# Drops and recreates the dev database, then runs migrations

DB_NAME="${POSTGRES_DEV_DB:-tensorzero-e2e-tests}"
DB_HOST="${POSTGRES_HOST:-localhost}"
DB_PORT="${POSTGRES_PORT:-5432}"
DB_USER="${POSTGRES_USER:-postgres}"
DB_PASSWORD="${POSTGRES_PASSWORD:-postgres}"
DEV_DATABASE_URL="postgres://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME"

if [[ -z "${TENSORZERO_POSTGRES_URL:-}" ]]; then
  echo "Error: TENSORZERO_POSTGRES_URL is not set (expected $DEV_DATABASE_URL)." >&2
  exit 1
fi

if [[ "$TENSORZERO_POSTGRES_URL" != "$DEV_DATABASE_URL" ]]; then
  echo "Error: TENSORZERO_POSTGRES_URL does not match dev database." >&2
  echo "Expected: $DEV_DATABASE_URL" >&2
  echo "Found:    $TENSORZERO_POSTGRES_URL" >&2
  echo "To override the database name, set POSTGRES_DEV_DB=<new_database_name>." >&2
  exit 1
fi

echo "==> Resetting dev database: $DB_NAME"

# Terminate existing connections and drop/recreate the database
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres <<EOF
-- Terminate existing connections
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = '$DB_NAME' AND pid <> pg_backend_pid();

DROP DATABASE IF EXISTS "$DB_NAME";
CREATE DATABASE "$DB_NAME";
EOF

echo "==> Database recreated"

# Run migrations
export TENSORZERO_POSTGRES_URL="postgres://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME"
echo "==> Running migrations..."

SQLX_OFFLINE=1 cargo run --package gateway --bin gateway -- --run-postgres-migrations

# TENSORZERO_USE_SERVER_COPY=0
./ui/fixtures/load_fixtures_postgres.sh

echo "==> Done! Database $DB_NAME is ready."
echo "    Connection: $TENSORZERO_POSTGRES_URL"
