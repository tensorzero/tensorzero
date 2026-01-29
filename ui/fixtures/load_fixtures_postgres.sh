#!/bin/bash
set -euo pipefail

# Load small fixtures into Postgres tables.
# Loads inference tables (Step 1) and feedback tables (Step 2).
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
-- Inference tables (Step 1)
TRUNCATE TABLE tensorzero.chat_inferences CASCADE;
TRUNCATE TABLE tensorzero.json_inferences CASCADE;
-- Feedback tables (Step 2)
TRUNCATE TABLE tensorzero.boolean_metric_feedback CASCADE;
TRUNCATE TABLE tensorzero.float_metric_feedback CASCADE;
TRUNCATE TABLE tensorzero.comment_feedback CASCADE;
TRUNCATE TABLE tensorzero.demonstration_feedback CASCADE;
EOF
    echo "  Done"
fi

# Download JSONL fixtures from R2 if not present
if [ ! -f "small-fixtures/chat_inference_examples.jsonl" ] || [ ! -f "small-fixtures/json_inference_examples.jsonl" ] || \
   [ ! -f "small-fixtures/boolean_metric_feedback_examples.jsonl" ] || [ ! -f "small-fixtures/float_metric_feedback_examples.jsonl" ] || \
   [ ! -f "small-fixtures/comment_feedback_examples.jsonl" ] || [ ! -f "small-fixtures/demonstration_feedback_examples.jsonl" ]; then
    echo "Downloading small fixtures..."
    if [ "${TENSORZERO_DOWNLOAD_FIXTURES_WITHOUT_CREDENTIALS:-}" = "1" ]; then
        uv run ./download-small-fixtures-http.py
    else
        uv run ./download-small-fixtures.py
    fi
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

# =====================================================================
# Feedback Tables (Step 2)
# =====================================================================

# Boolean Metric Feedback
# created_at is derived from the UUIDv7 id using tensorzero.uuid_v7_to_timestamp()
load_jsonl "small-fixtures/boolean_metric_feedback_examples.jsonl" "tensorzero.boolean_metric_feedback" "
INSERT INTO tensorzero.boolean_metric_feedback (
    id, target_id, metric_name, value, tags, created_at
)
SELECT
    (j->>'id')::uuid,
    (j->>'target_id')::uuid,
    j->>'metric_name',
    (j->>'value')::boolean,
    COALESCE(j->'tags', '{}')::jsonb,
    tensorzero.uuid_v7_to_timestamp((j->>'id')::uuid)
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (id) DO NOTHING;
"

# Float Metric Feedback
# created_at is derived from the UUIDv7 id using tensorzero.uuid_v7_to_timestamp()
load_jsonl "small-fixtures/float_metric_feedback_examples.jsonl" "tensorzero.float_metric_feedback" "
INSERT INTO tensorzero.float_metric_feedback (
    id, target_id, metric_name, value, tags, created_at
)
SELECT
    (j->>'id')::uuid,
    (j->>'target_id')::uuid,
    j->>'metric_name',
    (j->>'value')::double precision,
    COALESCE(j->'tags', '{}')::jsonb,
    tensorzero.uuid_v7_to_timestamp((j->>'id')::uuid)
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (id) DO NOTHING;
"

# Also load jaro_winkler_similarity_feedback into float_metric_feedback
load_jsonl "small-fixtures/jaro_winkler_similarity_feedback.jsonl" "tensorzero.float_metric_feedback" "
INSERT INTO tensorzero.float_metric_feedback (
    id, target_id, metric_name, value, tags, created_at
)
SELECT
    (j->>'id')::uuid,
    (j->>'target_id')::uuid,
    j->>'metric_name',
    (j->>'value')::double precision,
    COALESCE(j->'tags', '{}')::jsonb,
    tensorzero.uuid_v7_to_timestamp((j->>'id')::uuid)
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (id) DO NOTHING;
"

# Comment Feedback
# created_at is derived from the UUIDv7 id using tensorzero.uuid_v7_to_timestamp()
load_jsonl "small-fixtures/comment_feedback_examples.jsonl" "tensorzero.comment_feedback" "
INSERT INTO tensorzero.comment_feedback (
    id, target_id, target_type, value, tags, created_at
)
SELECT
    (j->>'id')::uuid,
    (j->>'target_id')::uuid,
    j->>'target_type',
    j->>'value',
    COALESCE(j->'tags', '{}')::jsonb,
    tensorzero.uuid_v7_to_timestamp((j->>'id')::uuid)
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (id) DO NOTHING;
"

# Demonstration Feedback
# Note: demonstration_feedback uses inference_id instead of target_id
# Note: value is JSONB (stores JsonInferenceOutput or Vec<ContentBlockChatOutput>)
# created_at is derived from the UUIDv7 id using tensorzero.uuid_v7_to_timestamp()
load_jsonl "small-fixtures/demonstration_feedback_examples.jsonl" "tensorzero.demonstration_feedback" "
INSERT INTO tensorzero.demonstration_feedback (
    id, inference_id, value, tags, created_at
)
SELECT
    (j->>'id')::uuid,
    (j->>'inference_id')::uuid,
    j->'value',
    COALESCE(j->'tags', '{}')::jsonb,
    tensorzero.uuid_v7_to_timestamp((j->>'id')::uuid)
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (id) DO NOTHING;
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
UNION ALL
SELECT 'boolean_metric_feedback', count(*) FROM tensorzero.boolean_metric_feedback
UNION ALL
SELECT 'float_metric_feedback', count(*) FROM tensorzero.float_metric_feedback
UNION ALL
SELECT 'comment_feedback', count(*) FROM tensorzero.comment_feedback
UNION ALL
SELECT 'demonstration_feedback', count(*) FROM tensorzero.demonstration_feedback
ORDER BY table_name;
EOF
