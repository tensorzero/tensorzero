#!/bin/bash
set -euo pipefail

# Load small fixtures into Postgres inference tables.
# Only loads chat_inferences and json_inferences (Step 1 tables).
#
# Usage:
#   ./ui/fixtures/load_fixtures_postgres.sh
#
# Environment variables:
#   TENSORZERO_POSTGRES_URL - Postgres connection URL (default: postgres://postgres:postgres@localhost:5432/tensorzero_ui_fixtures)
#   TENSORZERO_SKIP_TRUNCATE - Set to 1 to skip truncating tables before loading

POSTGRES_URL="${TENSORZERO_POSTGRES_URL:-postgres://postgres:postgres@localhost:5432/tensorzero_ui_fixtures}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

# Helper function to load JSONL into a table via temp TEXT table
load_jsonl() {
    local file="$1"
    local table="$2"
    local insert_sql="$3"

    if [ ! -f "$file" ]; then
        echo "Warning: $file not found, skipping"
        return
    fi

    echo "Loading $file into $table..."

    psql -q "$POSTGRES_URL" <<EOF
-- Create temp table for raw text (each line is a JSON string)
CREATE TEMP TABLE tmp_jsonl (data TEXT);

-- Load JSONL data as text using CSV format to preserve backslash escapes
\copy tmp_jsonl (data) FROM '$file' WITH (FORMAT csv, QUOTE E'\x01', DELIMITER E'\x02')

-- Insert into target table (cast text to jsonb for parsing)
$insert_sql

DROP TABLE tmp_jsonl;
EOF

    echo "  Done"
}

echo "Loading fixtures into PostgreSQL..."
echo "  URL: $POSTGRES_URL"

# Truncate tables first (unless skipped)
if [ "${TENSORZERO_SKIP_TRUNCATE:-}" = "1" ]; then
    echo "TENSORZERO_SKIP_TRUNCATE is set - skipping truncate"
else
    echo "Truncating tables..."
    psql -q "$POSTGRES_URL" <<EOF
TRUNCATE TABLE tensorzero.chat_inferences CASCADE;
TRUNCATE TABLE tensorzero.json_inferences CASCADE;
EOF
    echo "  Done"
fi

# Download JSONL fixtures from R2 if not present
if [ ! -f "small-fixtures/chat_inference_examples.jsonl" ] || [ ! -f "small-fixtures/json_inference_examples.jsonl" ]; then
    echo "Downloading small fixtures..."
    uv run ./download-small-fixtures.py
fi

# Chat Inferences
# Note: input, output, tool_params, inference_params are JSONB in our schema
# created_at is derived from the UUIDv7 id using tensorzero.uuid_v7_to_timestamp()
load_jsonl "small-fixtures/chat_inference_examples.jsonl" "tensorzero.chat_inferences" "
INSERT INTO tensorzero.chat_inferences (
    id, function_name, variant_name, episode_id,
    input, output, tool_params, inference_params,
    processing_time_ms, ttft_ms, tags, extra_body,
    dynamic_tools, dynamic_provider_tools, allowed_tools, tool_choice,
    parallel_tool_calls, created_at
)
SELECT
    (j->>'id')::uuid,
    j->>'function_name',
    j->>'variant_name',
    (j->>'episode_id')::uuid,
    COALESCE(j->'input', '{}')::jsonb,
    COALESCE(j->'output', '{}')::jsonb,
    COALESCE(j->'tool_params', '{}')::jsonb,
    COALESCE(j->'inference_params', '{}')::jsonb,
    (j->>'processing_time_ms')::integer,
    (j->>'ttft_ms')::integer,
    COALESCE(j->'tags', '{}')::jsonb,
    COALESCE(j->'extra_body', '[]')::jsonb,
    COALESCE(j->'dynamic_tools', '[]')::jsonb,
    COALESCE(j->'dynamic_provider_tools', '[]')::jsonb,
    j->'allowed_tools',
    j->>'tool_choice',
    (j->>'parallel_tool_calls')::boolean,
    tensorzero.uuid_v7_to_timestamp((j->>'id')::uuid)
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (id, created_at) DO NOTHING;
"

# JSON Inferences
# Note: input, output, output_schema, inference_params, auxiliary_content are JSONB in our schema
# created_at is derived from the UUIDv7 id using tensorzero.uuid_v7_to_timestamp()
load_jsonl "small-fixtures/json_inference_examples.jsonl" "tensorzero.json_inferences" "
INSERT INTO tensorzero.json_inferences (
    id, function_name, variant_name, episode_id,
    input, output, output_schema, inference_params,
    processing_time_ms, ttft_ms, tags, extra_body, auxiliary_content, created_at
)
SELECT
    (j->>'id')::uuid,
    j->>'function_name',
    j->>'variant_name',
    (j->>'episode_id')::uuid,
    COALESCE(j->'input', '{}')::jsonb,
    COALESCE(j->'output', '{}')::jsonb,
    COALESCE(j->'output_schema', '{}')::jsonb,
    COALESCE(j->'inference_params', '{}')::jsonb,
    (j->>'processing_time_ms')::integer,
    (j->>'ttft_ms')::integer,
    COALESCE(j->'tags', '{}')::jsonb,
    COALESCE(j->'extra_body', '[]')::jsonb,
    COALESCE(j->'auxiliary_content', '{}')::jsonb,
    tensorzero.uuid_v7_to_timestamp((j->>'id')::uuid)
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (id, created_at) DO NOTHING;
"

echo ""
echo "All fixtures loaded successfully!"

# Print row counts
echo ""
echo "Row counts:"
psql -q "$POSTGRES_URL" <<EOF
SELECT 'chat_inferences' as table_name, count(*) as count FROM tensorzero.chat_inferences
UNION ALL
SELECT 'json_inferences', count(*) FROM tensorzero.json_inferences
ORDER BY table_name;
EOF
