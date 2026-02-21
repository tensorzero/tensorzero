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
SELECT tensorzero.refresh_inference_by_function_statistics_incremental(
    full_refresh => TRUE
);
EOF


echo ""
echo "All fixtures loaded successfully!"
