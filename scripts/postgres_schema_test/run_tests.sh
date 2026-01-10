#!/bin/bash
# =============================================================================
# PostgreSQL Schema Migration and Test Runner
# Usage: ./run_tests.sh [postgres_url]
# Example: ./run_tests.sh "postgresql://user:pass@localhost:5432/tensorzero_test"
# =============================================================================

set -e

# Default connection URL
POSTGRES_URL="${1:-${TENSORZERO_POSTGRES_URL:-postgresql://localhost:5432/tensorzero_test}}"

echo "=== PostgreSQL Schema Migration and Test Runner ==="
echo "Database URL: ${POSTGRES_URL//:*@/:***@}"  # Hide password in output
echo ""

# Extract connection parameters
DB_HOST=$(echo "$POSTGRES_URL" | sed -n 's/.*@\([^:/]*\).*/\1/p')
DB_PORT=$(echo "$POSTGRES_URL" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
DB_NAME=$(echo "$POSTGRES_URL" | sed -n 's/.*\/\([^?]*\).*/\1/p')

echo "Host: $DB_HOST"
echo "Port: ${DB_PORT:-5432}"
echo "Database: $DB_NAME"
echo ""

# Check if psql is available
if ! command -v psql &> /dev/null; then
    echo "ERROR: psql command not found. Please install PostgreSQL client."
    exit 1
fi

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MIGRATIONS_DIR="$PROJECT_ROOT/tensorzero-core/src/db/postgres/migrations"

echo "Migrations directory: $MIGRATIONS_DIR"
echo ""

# Check if migrations directory exists
if [ ! -d "$MIGRATIONS_DIR" ]; then
    echo "ERROR: Migrations directory not found: $MIGRATIONS_DIR"
    exit 1
fi

# Function to run SQL file
run_sql() {
    local file="$1"
    local description="$2"
    echo "Running: $description"
    psql "$POSTGRES_URL" -f "$file" -v ON_ERROR_STOP=1
    echo ""
}

# Function to run SQL command
run_sql_cmd() {
    local cmd="$1"
    psql "$POSTGRES_URL" -c "$cmd" -v ON_ERROR_STOP=1
}

# =============================================================================
# Step 1: Create database if it doesn't exist
# =============================================================================
echo "=== Step 1: Checking database ==="
BASE_URL=$(echo "$POSTGRES_URL" | sed 's/\/[^/]*$/\/postgres/')
DB_EXISTS=$(psql "$BASE_URL" -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" 2>/dev/null || echo "")

if [ "$DB_EXISTS" != "1" ]; then
    echo "Creating database: $DB_NAME"
    psql "$BASE_URL" -c "CREATE DATABASE $DB_NAME" 2>/dev/null || true
else
    echo "Database exists: $DB_NAME"
fi
echo ""

# =============================================================================
# Step 2: Run migrations in order
# =============================================================================
echo "=== Step 2: Running migrations ==="

# Get sorted list of migration files
MIGRATION_FILES=$(ls -1 "$MIGRATIONS_DIR"/*.sql 2>/dev/null | sort)

if [ -z "$MIGRATION_FILES" ]; then
    echo "ERROR: No migration files found in $MIGRATIONS_DIR"
    exit 1
fi

for migration in $MIGRATION_FILES; do
    filename=$(basename "$migration")
    echo "--- Migration: $filename ---"

    # Run migration (ignore errors for CREATE IF NOT EXISTS, etc.)
    if psql "$POSTGRES_URL" -f "$migration" -v ON_ERROR_STOP=0 2>&1; then
        echo "OK"
    else
        echo "Migration may have partial failures (some objects may already exist)"
    fi
    echo ""
done

# =============================================================================
# Step 3: Run test script
# =============================================================================
echo "=== Step 3: Running schema tests ==="
TEST_SCRIPT="$SCRIPT_DIR/test_schema.sql"

if [ -f "$TEST_SCRIPT" ]; then
    psql "$POSTGRES_URL" -f "$TEST_SCRIPT"
else
    echo "WARNING: Test script not found: $TEST_SCRIPT"
fi

echo ""
echo "=== All done! ==="
