#!/bin/bash
set -euo pipefail

# Load small fixtures into Postgres tables.
#
# Table loading status:
# - "chat_inference_examples.jsonl" -> chat_inferences + chat_inference_data (Done)
# - "json_inference_examples.jsonl" -> json_inferences + json_inference_data (Done)
# - "metadata_only_inference_examples.jsonl" -> chat_inferences + json_inferences (Done, metadata only — no data rows)
# - "model_inference_examples.jsonl" -> model_inferences + model_inference_data (Done)
# - "boolean_metric_feedback_examples.jsonl" (Done)
# - "float_metric_feedback_examples.jsonl" (Done)
# - "demonstration_feedback_examples.jsonl" (Done)
# - "json_inference_datapoint_examples.jsonl" (Done)
# - "chat_inference_datapoint_examples.jsonl" (Done)
# - "comment_feedback_examples.jsonl" (Done)
# - "model_inference_cache_e2e.jsonl"
# - "jaro_winkler_similarity_feedback.jsonl"
# - "dynamic_evaluation_run_examples.jsonl" (Done)
# - "dynamic_evaluation_run_episode_examples.jsonl" (Done)
#
# Usage:
#   ./ui/fixtures/load_fixtures_postgres.sh
#
# Environment variables:
#   TENSORZERO_POSTGRES_URL - Postgres connection URL (default: postgres://postgres:postgres@localhost:5432/tensorzero_ui_fixtures)
#   TENSORZERO_SKIP_TRUNCATE - Set to 1 to skip truncating tables before loading
#   TENSORZERO_FIXTURES_DIR - Path to fixtures directory as seen by postgres server (default: /fixtures for docker, or $SCRIPT_DIR for local)
#   TENSORZERO_DROP_PARTITIONED_PKS - Set to 1 to drop/recreate PKs on partitioned bulk tables during large fixture load
#   TENSORZERO_SKIP_INFERENCE_TAG_GIN_INDEXES - Set to 1 to skip recreating idx_chat_inferences_tags and idx_json_inferences_tags
#   TENSORZERO_INDEX_BUILD_MAINTENANCE_WORK_MEM - Optional maintenance_work_mem value for index rebuild statements (example: 4GB)

POSTGRES_URL="${TENSORZERO_POSTGRES_URL:-postgres://postgres:postgres@localhost:5432/tensorzero_ui_fixtures}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

# Determine fixtures directory path as seen by the postgres server.
# Defaults to server-side COPY (faster, requires volume mount .:/fixtures:ro).
# Set TENSORZERO_USE_SERVER_COPY=0 for client-side \copy (no volume mount needed).
USE_SERVER_COPY="${TENSORZERO_USE_SERVER_COPY:-1}"
SERVER_FIXTURES_DIR="${TENSORZERO_FIXTURES_DIR:-/fixtures}"

# Helper function to load JSONL into a table via temp TEXT table
load_jsonl() {
    local file="$1"
    local table="$2"
    local insert_sql="$3"

    if [ ! -f "$file" ]; then
        echo "Warning: $file not found, skipping"
        return
    fi

    local start_time=$SECONDS
    echo "Loading $file into $table..."

    if [ "$USE_SERVER_COPY" = "1" ]; then
        # Server-side COPY (file must be accessible to postgres server)
        local server_file="${SERVER_FIXTURES_DIR}/${file}"
        psql -q "$POSTGRES_URL" <<EOF
-- Create temp table for raw text (each line is a JSON string)
CREATE TEMP TABLE tmp_jsonl (data TEXT);

-- Load JSONL data using server-side COPY
COPY tmp_jsonl (data) FROM '${server_file}' WITH (FORMAT csv, QUOTE E'\x01', DELIMITER E'\x02');

-- Insert into target table (cast text to jsonb for parsing)
$insert_sql

DROP TABLE tmp_jsonl;
EOF
    else
        # Client-side \copy (default for local development)
        psql -q "$POSTGRES_URL" <<EOF
-- Create temp table for raw text (each line is a JSON string)
CREATE TEMP TABLE tmp_jsonl (data TEXT);

-- Load JSONL data as text using CSV format to preserve backslash escapes
\copy tmp_jsonl (data) FROM '$file' WITH (FORMAT csv, QUOTE E'\x01', DELIMITER E'\x02')

-- Insert into target table (cast text to jsonb for parsing)
$insert_sql

DROP TABLE tmp_jsonl;
EOF
    fi

    echo "  Done ($(( SECONDS - start_time ))s)"
}

TOTAL_START=$SECONDS
echo "Loading fixtures into Postgres..."
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
TRUNCATE TABLE tensorzero.model_inferences CASCADE;
TRUNCATE TABLE tensorzero.chat_inference_data CASCADE;
TRUNCATE TABLE tensorzero.json_inference_data CASCADE;
TRUNCATE TABLE tensorzero.model_inference_data CASCADE;
-- Feedback tables (Step 2)
TRUNCATE TABLE tensorzero.boolean_metric_feedback CASCADE;
TRUNCATE TABLE tensorzero.float_metric_feedback CASCADE;
TRUNCATE TABLE tensorzero.comment_feedback CASCADE;
TRUNCATE TABLE tensorzero.demonstration_feedback CASCADE;
-- Datapoint tables
TRUNCATE TABLE tensorzero.chat_datapoints CASCADE;
TRUNCATE TABLE tensorzero.json_datapoints CASCADE;
-- Workflow evaluation tables (Step 5)
TRUNCATE TABLE tensorzero.workflow_evaluation_run_episodes CASCADE;
TRUNCATE TABLE tensorzero.workflow_evaluation_runs CASCADE;
EOF
    echo "  Done"
fi

# Download JSONL fixtures from R2 if not present
if [ ! -f "small-fixtures/chat_inference_examples.jsonl" ] || [ ! -f "small-fixtures/json_inference_examples.jsonl" ] || \
   [ ! -f "small-fixtures/boolean_metric_feedback_examples.jsonl" ] || [ ! -f "small-fixtures/float_metric_feedback_examples.jsonl" ] || \
   [ ! -f "small-fixtures/comment_feedback_examples.jsonl" ] || [ ! -f "small-fixtures/demonstration_feedback_examples.jsonl" ] || \
   [ ! -f "small-fixtures/chat_inference_datapoint_examples.jsonl" ] || [ ! -f "small-fixtures/json_inference_datapoint_examples.jsonl" ] || \
   [ ! -f "small-fixtures/model_inference_examples.jsonl" ] || \
   [ ! -f "small-fixtures/dynamic_evaluation_run_examples.jsonl" ] || \
   [ ! -f "small-fixtures/dynamic_evaluation_run_episode_examples.jsonl" ] || \
   [ ! -f "small-fixtures/metadata_only_inference_examples.jsonl" ]; then
    echo "Downloading small fixtures..."
    if [ "${TENSORZERO_DOWNLOAD_FIXTURES_WITHOUT_CREDENTIALS:-}" = "1" ]; then
        uv run ./download-small-fixtures-http.py
    else
        uv run ./download-small-fixtures.py
    fi
fi

# Chat Inferences (metadata)
# created_at is derived from the UUIDv7 id using tensorzero.uuid_v7_to_timestamp()
load_jsonl "small-fixtures/chat_inference_examples.jsonl" "tensorzero.chat_inferences" "
INSERT INTO tensorzero.chat_inferences (
    id, function_name, variant_name, episode_id,
    processing_time_ms, ttft_ms, tags,
    created_at
)
SELECT
    (j->>'id')::uuid,
    j->>'function_name',
    j->>'variant_name',
    (j->>'episode_id')::uuid,
    (j->>'processing_time_ms')::integer,
    (j->>'ttft_ms')::integer,
    COALESCE(NULLIF(j->>'tags', '')::jsonb, '{}'),
    tensorzero.uuid_v7_to_timestamp((j->>'id')::uuid)
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (id, created_at) DO NOTHING;
"

# Chat Inference IO
# Note: input, output, inference_params are JSONB in our schema
# ClickHouse stores these as String (JSON-encoded), so we use ->> to extract text then cast to jsonb
# NULLIF handles empty strings that would fail jsonb cast
load_jsonl "small-fixtures/chat_inference_examples.jsonl" "tensorzero.chat_inference_data" "
INSERT INTO tensorzero.chat_inference_data (
    id, input, output, inference_params,
    extra_body, dynamic_tools, dynamic_provider_tools, allowed_tools,
    tool_choice, parallel_tool_calls, created_at
)
SELECT
    (j->>'id')::uuid,
    COALESCE(NULLIF(j->>'input', '')::jsonb, '{}'),
    COALESCE(NULLIF(j->>'output', '')::jsonb, '[]'),
    COALESCE(NULLIF(j->>'inference_params', '')::jsonb, '{}'),
    COALESCE(NULLIF(j->>'extra_body', '')::jsonb, '[]'),
    COALESCE(NULLIF(j->>'dynamic_tools', '')::jsonb, '[]'),
    COALESCE(NULLIF(j->>'dynamic_provider_tools', '')::jsonb, '[]'),
    NULLIF(j->>'allowed_tools', '')::jsonb,
    j->'tool_choice',
    (j->>'parallel_tool_calls')::boolean,
    tensorzero.uuid_v7_to_timestamp((j->>'id')::uuid)
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (id, created_at) DO NOTHING;
"

# JSON Inferences (metadata)
# created_at is derived from the UUIDv7 id using tensorzero.uuid_v7_to_timestamp()
load_jsonl "small-fixtures/json_inference_examples.jsonl" "tensorzero.json_inferences" "
INSERT INTO tensorzero.json_inferences (
    id, function_name, variant_name, episode_id,
    processing_time_ms, ttft_ms, tags, created_at
)
SELECT
    (j->>'id')::uuid,
    j->>'function_name',
    j->>'variant_name',
    (j->>'episode_id')::uuid,
    (j->>'processing_time_ms')::integer,
    (j->>'ttft_ms')::integer,
    COALESCE(NULLIF(j->>'tags', '')::jsonb, '{}'),
    tensorzero.uuid_v7_to_timestamp((j->>'id')::uuid)
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (id, created_at) DO NOTHING;
"

# JSON Inference IO
# Note: input, output, output_schema, inference_params, auxiliary_content are JSONB in our schema
# ClickHouse stores these as String (JSON-encoded), so we use ->> to extract text then cast to jsonb
# NULLIF handles empty strings that would fail jsonb cast
load_jsonl "small-fixtures/json_inference_examples.jsonl" "tensorzero.json_inference_data" "
INSERT INTO tensorzero.json_inference_data (
    id, input, output, output_schema, inference_params,
    extra_body, auxiliary_content, created_at
)
SELECT
    (j->>'id')::uuid,
    COALESCE(NULLIF(j->>'input', '')::jsonb, '{}'),
    COALESCE(NULLIF(j->>'output', '')::jsonb, '{}'),
    COALESCE(NULLIF(j->>'output_schema', '')::jsonb, '{}'),
    COALESCE(NULLIF(j->>'inference_params', '')::jsonb, '{}'),
    COALESCE(NULLIF(j->>'extra_body', '')::jsonb, '[]'),
    COALESCE(NULLIF(j->>'auxiliary_content', '')::jsonb, '{}'),
    tensorzero.uuid_v7_to_timestamp((j->>'id')::uuid)
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (id, created_at) DO NOTHING;
"

# Metadata-only chat inferences (simulating data retention expiry — no data table rows)
load_jsonl "small-fixtures/metadata_only_inference_examples.jsonl" "tensorzero.chat_inferences" "
INSERT INTO tensorzero.chat_inferences (
    id, function_name, variant_name, episode_id,
    processing_time_ms, ttft_ms, tags, created_at
)
SELECT
    (j->>'id')::uuid,
    j->>'function_name',
    j->>'variant_name',
    (j->>'episode_id')::uuid,
    (j->>'processing_time_ms')::integer,
    (j->>'ttft_ms')::integer,
    COALESCE(NULLIF(j->>'tags', '')::jsonb, '{}'),
    tensorzero.uuid_v7_to_timestamp((j->>'id')::uuid)
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
WHERE j->>'inference_type' = 'chat'
ON CONFLICT (id, created_at) DO NOTHING;
"

# Metadata-only JSON inferences (simulating data retention expiry — no data table rows)
load_jsonl "small-fixtures/metadata_only_inference_examples.jsonl" "tensorzero.json_inferences" "
INSERT INTO tensorzero.json_inferences (
    id, function_name, variant_name, episode_id,
    processing_time_ms, ttft_ms, tags, created_at
)
SELECT
    (j->>'id')::uuid,
    j->>'function_name',
    j->>'variant_name',
    (j->>'episode_id')::uuid,
    (j->>'processing_time_ms')::integer,
    (j->>'ttft_ms')::integer,
    COALESCE(NULLIF(j->>'tags', '')::jsonb, '{}'),
    tensorzero.uuid_v7_to_timestamp((j->>'id')::uuid)
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
WHERE j->>'inference_type' = 'json'
ON CONFLICT (id, created_at) DO NOTHING;
"

# Model Inferences (metadata)
# created_at is derived from the UUIDv7 id using tensorzero.uuid_v7_to_timestamp()
load_jsonl "small-fixtures/model_inference_examples.jsonl" "tensorzero.model_inferences" "
INSERT INTO tensorzero.model_inferences (
    id, inference_id, input_tokens, output_tokens,
    response_time_ms, model_name, model_provider_name,
    ttft_ms, cached, finish_reason, created_at
)
SELECT
    (j->>'id')::uuid,
    (j->>'inference_id')::uuid,
    (j->>'input_tokens')::integer,
    (j->>'output_tokens')::integer,
    (j->>'response_time_ms')::integer,
    j->>'model_name',
    j->>'model_provider_name',
    (j->>'ttft_ms')::integer,
    COALESCE((j->>'cached')::boolean, false),
    j->>'finish_reason',
    tensorzero.uuid_v7_to_timestamp((j->>'id')::uuid)
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (id, created_at) DO NOTHING;
"

# Model Inference IO
# Note: input_messages, output are JSONB in our schema
# ClickHouse stores these as String (JSON-encoded), so we use ->> to extract text then cast to jsonb
# NULLIF handles empty strings that would fail jsonb cast
load_jsonl "small-fixtures/model_inference_examples.jsonl" "tensorzero.model_inference_data" "
INSERT INTO tensorzero.model_inference_data (
    id, raw_request, raw_response, system,
    input_messages, output, created_at
)
SELECT
    (j->>'id')::uuid,
    j->>'raw_request',
    j->>'raw_response',
    j->>'system',
    COALESCE(NULLIF(j->>'input_messages', '')::jsonb, '[]'),
    COALESCE(NULLIF(j->>'output', '')::jsonb, '[]'),
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
# ClickHouse stores value as String (JSON-encoded), so we use ->> to extract text then cast to jsonb
# NULLIF handles empty strings that would fail jsonb cast
# created_at is derived from the UUIDv7 id using tensorzero.uuid_v7_to_timestamp()
load_jsonl "small-fixtures/demonstration_feedback_examples.jsonl" "tensorzero.demonstration_feedback" "
INSERT INTO tensorzero.demonstration_feedback (
    id, inference_id, value, tags, created_at
)
SELECT
    (j->>'id')::uuid,
    (j->>'inference_id')::uuid,
    NULLIF(j->>'value', '')::jsonb,
    COALESCE(NULLIF(j->>'tags', '')::jsonb, '{}'),
    tensorzero.uuid_v7_to_timestamp((j->>'id')::uuid)
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (id) DO NOTHING;
"

# =====================================================================
# Datapoint Tables
# =====================================================================

# Chat Datapoints
# Note: input, output are stored as JSON-encoded strings in ClickHouse
# We use ->> to extract text then cast to jsonb
# NULLIF handles empty strings that would fail jsonb cast
# regexp_replace re-escapes newlines that were unescaped during JSON string extraction
# created_at is derived from the UUIDv7 id using tensorzero.uuid_v7_to_timestamp()
# updated_at is parsed from the fixture's updated_at field
load_jsonl "small-fixtures/chat_inference_datapoint_examples.jsonl" "tensorzero.chat_datapoints" "
INSERT INTO tensorzero.chat_datapoints (
    id, dataset_name, function_name, episode_id,
    input, output,
    dynamic_tools, dynamic_provider_tools, allowed_tools, tool_choice, parallel_tool_calls,
    tags, is_custom, source_inference_id, staled_at, created_at, updated_at
)
SELECT
    (j->>'id')::uuid,
    j->>'dataset_name',
    j->>'function_name',
    (j->>'episode_id')::uuid,
    COALESCE(NULLIF(j->>'input', '')::jsonb, '{}'),
    NULLIF(j->>'output', '')::jsonb,
    COALESCE(NULLIF(j->>'dynamic_tools', '')::jsonb, '[]'),
    COALESCE(NULLIF(j->>'dynamic_provider_tools', '')::jsonb, '[]'),
    NULLIF(j->>'allowed_tools', '')::jsonb,
    j->'tool_choice',
    (j->>'parallel_tool_calls')::boolean,
    COALESCE(j->'tags', '{}')::jsonb,
    COALESCE((j->>'is_custom')::boolean, false),
    (j->>'source_inference_id')::uuid,
    NULLIF(j->>'staled_at', '')::timestamptz,
    tensorzero.uuid_v7_to_timestamp((j->>'id')::uuid),
    COALESCE(NULLIF(j->>'updated_at', '')::timestamptz, NOW())
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (id) DO NOTHING;
"

# JSON Datapoints
# Note: input, output, output_schema are stored as JSON-encoded strings in ClickHouse
# We use ->> to extract text then cast to jsonb
# NULLIF handles empty strings that would fail jsonb cast
# regexp_replace re-escapes newlines that were unescaped during JSON string extraction
# created_at is derived from the UUIDv7 id using tensorzero.uuid_v7_to_timestamp()
# updated_at is parsed from the fixture's updated_at field
load_jsonl "small-fixtures/json_inference_datapoint_examples.jsonl" "tensorzero.json_datapoints" "
INSERT INTO tensorzero.json_datapoints (
    id, dataset_name, function_name, episode_id,
    input, output, output_schema, tags,
    is_custom, source_inference_id, staled_at, created_at, updated_at
)
SELECT
    (j->>'id')::uuid,
    j->>'dataset_name',
    j->>'function_name',
    (j->>'episode_id')::uuid,
    COALESCE(NULLIF(regexp_replace(j->>'input', E'\\n', E'\\\\n', 'g'), '')::jsonb, '{}'),
    NULLIF(regexp_replace(j->>'output', E'\\n', E'\\\\n', 'g'), '')::jsonb,
    COALESCE(NULLIF(regexp_replace(j->>'output_schema', E'\\n', E'\\\\n', 'g'), '')::jsonb, '{}'),
    COALESCE(j->'tags', '{}')::jsonb,
    COALESCE((j->>'is_custom')::boolean, false),
    (j->>'source_inference_id')::uuid,
    NULLIF(j->>'staled_at', '')::timestamptz,
    tensorzero.uuid_v7_to_timestamp((j->>'id')::uuid),
    COALESCE(NULLIF(j->>'updated_at', '')::timestamptz, NOW())
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (id) DO NOTHING;
"

# =====================================================================
# Workflow Evaluation Tables (Step 5)
# =====================================================================

# Workflow Evaluation Runs
# Note: run_id_uint is a decimal string representation of UInt128 which needs to be converted to UUID
# We use tensorzero.uint128_to_uuid() to convert the UInt128 to UUID format
# is_deleted maps to staled_at (if true, use updated_at as staled_at)
# created_at is derived from the run_id UUID using tensorzero.uuid_v7_to_timestamp()
load_jsonl "small-fixtures/dynamic_evaluation_run_examples.jsonl" "tensorzero.workflow_evaluation_runs" "
INSERT INTO tensorzero.workflow_evaluation_runs (
    run_id, project_name, run_display_name, variant_pins, tags,
    staled_at, created_at, updated_at
)
SELECT
    tensorzero.uint128_to_uuid((j->>'run_id_uint')::NUMERIC),
    j->>'project_name',
    j->>'run_display_name',
    COALESCE(j->'variant_pins', '{}')::jsonb,
    COALESCE(j->'tags', '{}')::jsonb,
    CASE WHEN (j->>'is_deleted')::boolean THEN (j->>'updated_at')::timestamptz ELSE NULL END,
    tensorzero.uuid_v7_to_timestamp(tensorzero.uint128_to_uuid((j->>'run_id_uint')::NUMERIC)),
    (j->>'updated_at')::timestamptz
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (run_id) DO NOTHING;
"

# Workflow Evaluation Run Episodes
# Note: episode_id_uint is a decimal string representation of UInt128 which needs to be converted to UUID
# run_id is already a UUID string in the fixture
# is_deleted maps to staled_at (if true, use updated_at as staled_at)
# created_at is derived from the episode_id UUID using tensorzero.uuid_v7_to_timestamp()
load_jsonl "small-fixtures/dynamic_evaluation_run_episode_examples.jsonl" "tensorzero.workflow_evaluation_run_episodes" "
INSERT INTO tensorzero.workflow_evaluation_run_episodes (
    episode_id, run_id, variant_pins, task_name, tags,
    staled_at, created_at, updated_at
)
SELECT
    tensorzero.uint128_to_uuid((j->>'episode_id_uint')::NUMERIC),
    (j->>'run_id')::uuid,
    COALESCE(j->'variant_pins', '{}')::jsonb,
    j->>'datapoint_name',
    COALESCE(j->'tags', '{}')::jsonb,
    CASE WHEN (j->>'is_deleted')::boolean THEN (j->>'updated_at')::timestamptz ELSE NULL END,
    tensorzero.uuid_v7_to_timestamp(tensorzero.uint128_to_uuid((j->>'episode_id_uint')::NUMERIC)),
    (j->>'updated_at')::timestamptz
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (episode_id) DO NOTHING;
"

# =====================================================================
# Large Fixtures (optional)
# =====================================================================

# If TENSORZERO_SKIP_LARGE_FIXTURES equals 1, skip large fixtures
if [ "${TENSORZERO_SKIP_LARGE_FIXTURES:-}" = "1" ]; then
    echo ""
    echo "TENSORZERO_SKIP_LARGE_FIXTURES is set to 1 - skipping large fixtures"
else
    echo ""
    echo "Loading large fixtures..."

    # Download large fixtures if not present
    if [ ! -d "large-fixtures" ] || [ -z "$(ls -A large-fixtures/*.parquet 2>/dev/null)" ]; then
        echo "Downloading large fixtures..."
        uv run ./download-large-fixtures.py
    fi

    # Convert source parquet to Postgres-compatible parquet
    CONVERT_START=$SECONDS
    echo "Converting parquet to Postgres-compatible format..."
    uv run ./load_large_fixtures_postgres.py
    echo "  Conversion done ($(( SECONDS - CONVERT_START ))s)"

    # Enable pg_parquet extension for server-side COPY FROM parquet
    echo "Creating pg_parquet extension..."
    psql -q "$POSTGRES_URL" -c "CREATE EXTENSION IF NOT EXISTS pg_parquet"

    SERVER_PARQUET_DIR="${SERVER_FIXTURES_DIR}/large-fixtures/postgres-parquet"

    # Tables we're bulk loading into
    BULK_TABLES="'boolean_metric_feedback','float_metric_feedback','comment_feedback','demonstration_feedback','chat_inferences','chat_inference_data','json_inferences','json_inference_data','model_inferences','model_inference_data'"
    DROP_PARTITIONED_PKS="${TENSORZERO_DROP_PARTITIONED_PKS:-0}"

    # Optional: drop/recreate PKs on partitioned bulk tables for faster COPY.
    # This is disabled by default because PK removal also removes uniqueness checks
    # during load.
    PARTITIONED_PK_DEFS=""
    PARTITIONED_PK_COUNT=0
    if [ "$DROP_PARTITIONED_PKS" = "1" ]; then
        echo "Capturing PK definitions for partitioned bulk tables..."
        PARTITIONED_PK_DEFS=$(psql -qtAX "$POSTGRES_URL" -c "
            SELECT format(
                'ALTER TABLE tensorzero.%I ADD CONSTRAINT %I %s;',
                c.relname,
                con.conname,
                pg_get_constraintdef(con.oid)
            )
            FROM pg_constraint con
            JOIN pg_class c ON c.oid = con.conrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            JOIN pg_partitioned_table pt ON pt.partrelid = c.oid
            WHERE n.nspname = 'tensorzero'
              AND c.relname IN ($BULK_TABLES)
              AND con.contype = 'p'
            ORDER BY c.relname;
        ")
        PARTITIONED_PK_COUNT=$(echo "$PARTITIONED_PK_DEFS" | grep -c 'ADD CONSTRAINT' || true)
        if [ "$PARTITIONED_PK_COUNT" -gt 0 ]; then
            PK_DROP_START=$SECONDS
            echo "Dropping $PARTITIONED_PK_COUNT PK constraints on partitioned bulk tables..."
            psql -qtAX "$POSTGRES_URL" -c "
                SELECT format(
                    'ALTER TABLE tensorzero.%I DROP CONSTRAINT %I;',
                    c.relname,
                    con.conname
                )
                FROM pg_constraint con
                JOIN pg_class c ON c.oid = con.conrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                JOIN pg_partitioned_table pt ON pt.partrelid = c.oid
                WHERE n.nspname = 'tensorzero'
                  AND c.relname IN ($BULK_TABLES)
                  AND con.contype = 'p'
                ORDER BY c.relname;
            " | psql -q -v ON_ERROR_STOP=1 "$POSTGRES_URL"
            echo "  Dropped partitioned PK constraints ($(( SECONDS - PK_DROP_START ))s)"
        else
            echo "  No partitioned PK constraints found"
        fi
    fi

    # Drop non-PK indexes on bulk-loaded tables (index maintenance is the main bottleneck).
    # Save index definitions so we can recreate them after loading.
    #
    # For partitioned tables, pg_indexes returns "CREATE INDEX ... ON ONLY <table>" which
    # only creates parent index metadata without per-partition indexes. We rewrite
    # "ON ONLY" to "ON" so per-partition indexes are auto-created during rebuild.
    # (Targeting only _default partition indexes isn't feasible because PostgreSQL doesn't
    # allow dropping child indexes that are attached to a parent partitioned index.)
    echo "Dropping non-PK indexes for bulk loading..."
    INDEX_DROP_START=$SECONDS
    INDEX_DEFS=$(psql -qtAX "$POSTGRES_URL" -c "
        SELECT regexp_replace(indexdef, '[[:space:]]+ON[[:space:]]+ONLY[[:space:]]+', ' ON ', 'gi') || ';'
        FROM pg_indexes i
        WHERE i.schemaname = 'tensorzero'
          AND i.tablename IN ($BULK_TABLES)
          AND NOT EXISTS (
              SELECT 1 FROM pg_index pi
              JOIN pg_class c ON c.oid = pi.indexrelid
              WHERE c.relname = i.indexname AND pi.indisprimary
          )
        ORDER BY i.indexname;
    ")
    # Safety check: partitioned-table index DDL must not contain ON ONLY.
    if echo "$INDEX_DEFS" | grep -E -q 'ON ONLY tensorzero\\.(chat_inferences|json_inferences|model_inferences|chat_inference_data|json_inference_data|model_inference_data)([[:space:]]|$)'; then
        echo "Error: index recreation SQL still contains ON ONLY for partitioned bulk tables"
        exit 1
    fi
    # Optional: skip heavyweight GIN indexes on inference metadata tag columns.
    # These can take a long time to build on large fixture datasets.
    if [ "${TENSORZERO_SKIP_INFERENCE_TAG_GIN_INDEXES:-0}" = "1" ]; then
        echo "Skipping idx_chat_inferences_tags and idx_json_inferences_tags rebuild"
        INDEX_DEFS=$(echo "$INDEX_DEFS" | grep -vE '^CREATE INDEX( IF NOT EXISTS)? (idx_chat_inferences_tags|idx_json_inferences_tags) ' || true)
    fi
    INDEX_COUNT=$(echo "$INDEX_DEFS" | grep -c 'CREATE INDEX' || true)
    if [ "$INDEX_COUNT" -gt 0 ]; then
        # Generate DROP statements with proper SQL identifier quoting via format(%I)
        # and execute in a single psql call.
        psql -qtAX "$POSTGRES_URL" -c "
            SELECT format('DROP INDEX IF EXISTS tensorzero.%I;', i.indexname)
            FROM pg_indexes i
            WHERE i.schemaname = 'tensorzero'
              AND i.tablename IN ($BULK_TABLES)
              AND NOT EXISTS (
                  SELECT 1 FROM pg_index pi
                  JOIN pg_class c ON c.oid = pi.indexrelid
                  WHERE c.relname = i.indexname AND pi.indisprimary
              )
            ORDER BY i.indexname;
        " | psql -q -v ON_ERROR_STOP=1 "$POSTGRES_URL"
        echo "  Dropped $INDEX_COUNT indexes ($(( SECONDS - INDEX_DROP_START ))s)"
    else
        echo "  No non-PK indexes found"
    fi

    # Bulk load large parquet files in parallel batches of 2.
    # Each table gets its own psql backend with synchronous_commit=off.
    # Tables are independent (no FK constraints) and indexes are already dropped.
    # Parallelism is limited to avoid OOM in the Postgres container.
    BULK_LOAD_START=$SECONDS
    echo "Bulk loading all large fixtures (2-way parallel)..."

    copy_parquet() {
        local table="$1"
        local col_names="$2"
        psql -q "$POSTGRES_URL" -c "SET synchronous_commit = off; COPY tensorzero.${table} (${col_names}) FROM '${SERVER_PARQUET_DIR}/${table}.parquet' WITH (format 'parquet')"
        echo "  Loaded ${table}"
    }

    # Batch 1: feedback tables
    copy_parquet "boolean_metric_feedback" "id, target_id, metric_name, value, tags, created_at" &
    copy_parquet "float_metric_feedback" "id, target_id, metric_name, value, tags, created_at" &
    wait

    # Batch 2: feedback tables
    copy_parquet "comment_feedback" "id, target_id, target_type, value, tags, created_at" &
    copy_parquet "demonstration_feedback" "id, inference_id, value, tags, created_at" &
    wait

    # Batch 3: inference metadata
    copy_parquet "chat_inferences" "id, function_name, variant_name, episode_id, processing_time_ms, ttft_ms, tags, created_at" &
    copy_parquet "json_inferences" "id, function_name, variant_name, episode_id, processing_time_ms, ttft_ms, tags, created_at" &
    wait

    # Batch 4: inference metadata + data
    copy_parquet "model_inferences" "id, inference_id, input_tokens, output_tokens, response_time_ms, model_name, model_provider_name, ttft_ms, cached, finish_reason, created_at" &
    copy_parquet "chat_inference_data" "id, input, output, inference_params, extra_body, dynamic_tools, dynamic_provider_tools, allowed_tools, tool_choice, parallel_tool_calls, created_at" &
    wait

    # Batch 5: inference data (heavy JSONB columns)
    copy_parquet "json_inference_data" "id, input, output, output_schema, inference_params, extra_body, auxiliary_content, created_at" &
    copy_parquet "model_inference_data" "id, raw_request, raw_response, system, input_messages, output, created_at" &
    wait
    echo "  Bulk load done ($(( SECONDS - BULK_LOAD_START ))s)"

    # Recreate partitioned-table PKs if dropped
    if [ "$DROP_PARTITIONED_PKS" = "1" ] && [ "$PARTITIONED_PK_COUNT" -gt 0 ]; then
        PK_REBUILD_START=$SECONDS
        echo "Recreating $PARTITIONED_PK_COUNT partitioned PK constraints..."
        echo "$PARTITIONED_PK_DEFS" | psql -q -v ON_ERROR_STOP=1 "$POSTGRES_URL"
        echo "  Done ($(( SECONDS - PK_REBUILD_START ))s)"
    fi

    # Recreate indexes and print per-index timing for diagnostics.
    if [ "$INDEX_COUNT" -gt 0 ]; then
        INDEX_REBUILD_START=$SECONDS
        INDEX_MAINTENANCE_WORK_MEM="${TENSORZERO_INDEX_BUILD_MAINTENANCE_WORK_MEM:-}"
        echo "Recreating $INDEX_COUNT indexes..."
        if [ -n "$INDEX_MAINTENANCE_WORK_MEM" ]; then
            echo "  Using maintenance_work_mem=${INDEX_MAINTENANCE_WORK_MEM} for each index build"
        fi
        while IFS= read -r idx_sql; do
            if [ -z "$idx_sql" ]; then
                continue
            fi
            INDEX_ONE_START=$SECONDS
            echo "  Running ${idx_sql}..."
            if [ -n "$INDEX_MAINTENANCE_WORK_MEM" ]; then
                psql -q -v ON_ERROR_STOP=1 "$POSTGRES_URL" -c "SET maintenance_work_mem = '${INDEX_MAINTENANCE_WORK_MEM}'; ${idx_sql}"
            else
                psql -q -v ON_ERROR_STOP=1 "$POSTGRES_URL" -c "$idx_sql"
            fi
            echo "    Done ($(( SECONDS - INDEX_ONE_START ))s)"
        done <<< "$INDEX_DEFS"
        echo "  Done ($(( SECONDS - INDEX_REBUILD_START ))s)"
    fi
fi

# =====================================================================
# Refresh incremental rollups
# =====================================================================
#

BACKFILL_START=$SECONDS
echo "Backfilling model provider statistics and latency histograms..."
psql -q "$POSTGRES_URL" <<EOF
SELECT tensorzero.refresh_model_provider_statistics_incremental(
    full_refresh => TRUE
);
SELECT tensorzero.refresh_model_latency_histogram_minute_incremental(
    full_refresh => TRUE
);
SELECT tensorzero.refresh_model_latency_histogram_hour_incremental(
    full_refresh => TRUE
);
EOF
echo "  Done ($(( SECONDS - BACKFILL_START ))s)"

echo ""
echo "All fixtures loaded successfully! ($(( SECONDS - TOTAL_START ))s total)"
