#!/bin/bash
set -euo pipefail

POSTGRES_URL="${TENSORZERO_POSTGRES_URL:-postgres://postgres:postgres@localhost:5432/tensorzero-e2e-tests}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

# Helper function to load JSONL into a table via temp TEXT table
load_jsonl() {
    local file="$1"
    local table="$2"
    local insert_sql="$3"

    if [ ! -f "$file" ]; then
        return
    fi

    if should_skip_file "$file"; then
        echo "Skipping $file (waiting for $TENSORZERO_START_FROM)"
        return
    fi

    echo "Loading $file into $table..."

    psql -q "$POSTGRES_URL" <<EOF
-- Create temp table for raw text (each line is a JSON string)
CREATE TEMP TABLE tmp_jsonl (data TEXT);

-- Load JSONL data as text using CSV format to preserve backslash escapes
\copy tmp_jsonl (data) FROM '$file' WITH (FORMAT csv, QUOTE E'\x01', DELIMITER E'\x02')

-- Insert into target table (cast text to jsonb)
$insert_sql

DROP TABLE tmp_jsonl;
EOF

    echo "  Done"
}

# Helper function to load Parquet files using Python
load_parquet() {
    local file="$1"
    local table="$2"

    if [ ! -f "$file" ]; then
        return
    fi

    if should_skip_file "$file"; then
        echo "Skipping $file (waiting for $TENSORZERO_START_FROM)"
        return
    fi

    echo "Loading $file into $table..."

    uv run "$SCRIPT_DIR/load_parquet.py" "$file" "$table" "$POSTGRES_URL"

    echo "  Done"
}

echo "Loading fixtures into PostgreSQL..."

# Truncate all tables first (unless skipped)
if [ "${TENSORZERO_SKIP_TRUNCATE:-}" = "1" ]; then
    echo "TENSORZERO_SKIP_TRUNCATE is set - skipping truncate"
else
    echo "Truncating tables..."
    psql -q "$POSTGRES_URL" <<EOF
TRUNCATE TABLE chat_inference CASCADE;
TRUNCATE TABLE json_inference CASCADE;
TRUNCATE TABLE model_inference CASCADE;
TRUNCATE TABLE boolean_metric_feedback CASCADE;
TRUNCATE TABLE float_metric_feedback CASCADE;
TRUNCATE TABLE comment_feedback CASCADE;
TRUNCATE TABLE demonstration_feedback CASCADE;
TRUNCATE TABLE chat_inference_datapoint CASCADE;
TRUNCATE TABLE json_inference_datapoint CASCADE;
EOF
    echo "  Done"
fi

# Track whether we should skip files (for TENSORZERO_START_FROM)
SKIP_UNTIL_FILE="${TENSORZERO_START_FROM:-}"
should_skip_file() {
    local file="$1"
    if [ -z "$SKIP_UNTIL_FILE" ]; then
        return 1  # Don't skip
    fi
    if [ "$file" = "$SKIP_UNTIL_FILE" ] || [[ "$file" == *"$SKIP_UNTIL_FILE"* ]]; then
        SKIP_UNTIL_FILE=""  # Found it, stop skipping
        return 1  # Don't skip this file
    fi
    return 0  # Skip this file
}

# Load small fixtures unless skipped
if [ "${TENSORZERO_SKIP_SMALL_FIXTURES:-}" = "1" ]; then
    echo "TENSORZERO_SKIP_SMALL_FIXTURES is set - skipping small fixtures"
else

# Chat Inference
load_jsonl "chat_inference_examples.jsonl" "chat_inference" "
INSERT INTO chat_inference (id, function_name, variant_name, episode_id, input, output, tool_params, inference_params, processing_time_ms, tags, extra_body, ttft_ms)
SELECT
    (j->>'id')::uuid,
    j->>'function_name',
    j->>'variant_name',
    (j->>'episode_id')::uuid,
    (j->>'input')::jsonb,
    (j->>'output')::jsonb,
    j->>'tool_params',
    (j->>'inference_params')::jsonb,
    (j->>'processing_time_ms')::integer,
    COALESCE(j->'tags', '{}')::jsonb,
    (j->>'extra_body')::jsonb,
    (j->>'ttft_ms')::integer
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (id) DO NOTHING;
"

# JSON Inference
load_jsonl "json_inference_examples.jsonl" "json_inference" "
INSERT INTO json_inference (id, function_name, variant_name, episode_id, input, output, output_schema, inference_params, processing_time_ms, tags, extra_body, ttft_ms, auxiliary_content)
SELECT
    (j->>'id')::uuid,
    j->>'function_name',
    j->>'variant_name',
    (j->>'episode_id')::uuid,
    (j->>'input')::jsonb,
    (j->>'output')::jsonb,
    j->>'output_schema',
    (j->>'inference_params')::jsonb,
    (j->>'processing_time_ms')::integer,
    COALESCE(j->'tags', '{}')::jsonb,
    (j->>'extra_body')::jsonb,
    (j->>'ttft_ms')::integer,
    j->>'auxiliary_content'
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (id) DO NOTHING;
"

# Model Inference
load_jsonl "model_inference_examples.jsonl" "model_inference" "
INSERT INTO model_inference (id, inference_id, raw_request, raw_response, model_name, model_provider_name, input_tokens, output_tokens, response_time_ms, ttft_ms, system, input_messages, output, cached, finish_reason)
SELECT
    (j->>'id')::uuid,
    (j->>'inference_id')::uuid,
    j->>'raw_request',
    j->>'raw_response',
    j->>'model_name',
    j->>'model_provider_name',
    (j->>'input_tokens')::integer,
    (j->>'output_tokens')::integer,
    (j->>'response_time_ms')::integer,
    (j->>'ttft_ms')::integer,
    j->>'system',
    (j->>'input_messages')::jsonb,
    (j->>'output')::jsonb,
    COALESCE((j->>'cached')::boolean, false),
    CASE
        WHEN j->>'finish_reason' IN ('stop', 'length', 'tool_calls', 'content_filter', 'other')
        THEN (j->>'finish_reason')::finish_reason_enum
        WHEN j->>'finish_reason' IS NOT NULL THEN 'other'::finish_reason_enum
        ELSE NULL
    END
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (id) DO NOTHING;
"

# Boolean Metric Feedback
load_jsonl "boolean_metric_feedback_examples.jsonl" "boolean_metric_feedback" "
INSERT INTO boolean_metric_feedback (id, target_id, metric_name, value, tags)
SELECT
    (j->>'id')::uuid,
    (j->>'target_id')::uuid,
    j->>'metric_name',
    (j->>'value')::boolean,
    COALESCE(j->'tags', '{}')::jsonb
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (id) DO NOTHING;
"

# Float Metric Feedback
load_jsonl "float_metric_feedback_examples.jsonl" "float_metric_feedback" "
INSERT INTO float_metric_feedback (id, target_id, metric_name, value, tags)
SELECT
    (j->>'id')::uuid,
    (j->>'target_id')::uuid,
    j->>'metric_name',
    (j->>'value')::real,
    COALESCE(j->'tags', '{}')::jsonb
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (id) DO NOTHING;
"

# Jaro Winkler Similarity Feedback (also float feedback)
load_jsonl "jaro_winkler_similarity_feedback.jsonl" "float_metric_feedback" "
INSERT INTO float_metric_feedback (id, target_id, metric_name, value, tags)
SELECT
    (j->>'id')::uuid,
    (j->>'target_id')::uuid,
    j->>'metric_name',
    (j->>'value')::real,
    COALESCE(j->'tags', '{}')::jsonb
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (id) DO NOTHING;
"

# Comment Feedback
load_jsonl "comment_feedback_examples.jsonl" "comment_feedback" "
INSERT INTO comment_feedback (id, target_id, target_type, value, tags)
SELECT
    (j->>'id')::uuid,
    (j->>'target_id')::uuid,
    COALESCE(j->>'target_type', 'inference')::target_type_enum,
    j->>'value',
    COALESCE(j->'tags', '{}')::jsonb
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (id) DO NOTHING;
"

# Demonstration Feedback
load_jsonl "demonstration_feedback_examples.jsonl" "demonstration_feedback" "
INSERT INTO demonstration_feedback (id, inference_id, value, tags)
SELECT
    (j->>'id')::uuid,
    (j->>'inference_id')::uuid,
    j->>'value',
    COALESCE(j->'tags', '{}')::jsonb
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (id) DO NOTHING;
"

# Chat Inference Datapoint
load_jsonl "chat_inference_datapoint_examples.jsonl" "chat_inference_datapoint" "
INSERT INTO chat_inference_datapoint (id, dataset_name, function_name, episode_id, input, output, tool_params, tags, source_inference_id, is_deleted, is_custom, staled_at)
SELECT
    (j->>'id')::uuid,
    j->>'dataset_name',
    j->>'function_name',
    (j->>'episode_id')::uuid,
    (j->>'input')::jsonb,
    (j->>'output')::jsonb,
    j->>'tool_params',
    COALESCE(j->'tags', '{}')::jsonb,
    (j->>'source_inference_id')::uuid,
    COALESCE((j->>'is_deleted')::boolean, false),
    COALESCE((j->>'is_custom')::boolean, false),
    (j->>'staled_at')::timestamptz
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (dataset_name, id) DO NOTHING;
"

# JSON Inference Datapoint
load_jsonl "json_inference_datapoint_examples.jsonl" "json_inference_datapoint" "
INSERT INTO json_inference_datapoint (id, dataset_name, function_name, episode_id, input, output, output_schema, tags, auxiliary_content, source_inference_id, is_deleted, is_custom, staled_at)
SELECT
    (j->>'id')::uuid,
    j->>'dataset_name',
    j->>'function_name',
    (j->>'episode_id')::uuid,
    (j->>'input')::jsonb,
    (j->>'output')::jsonb,
    j->>'output_schema',
    COALESCE(j->'tags', '{}')::jsonb,
    j->>'auxiliary',
    (j->>'source_inference_id')::uuid,
    COALESCE((j->>'is_deleted')::boolean, false),
    COALESCE((j->>'is_custom')::boolean, false),
    (j->>'staled_at')::timestamptz
FROM tmp_jsonl, LATERAL (SELECT data::jsonb AS j) AS parsed
ON CONFLICT (dataset_name, id) DO NOTHING;
"

    echo ""
    echo "Small fixtures loaded."
fi  # end of small fixtures

# Load large fixtures if available and not skipped
if [ "${TENSORZERO_SKIP_LARGE_FIXTURES:-}" = "1" ]; then
    echo "TENSORZERO_SKIP_LARGE_FIXTURES is set - skipping large fixtures"
else
    # Download large fixtures if not present
    if [ ! -d "./s3-fixtures" ] || [ -z "$(ls -A ./s3-fixtures 2>/dev/null)" ]; then
        echo "Downloading large fixtures..."
        uv run ./download-large-fixtures.py
    fi

    echo ""
    echo "Loading large fixtures..."

    load_parquet "./s3-fixtures/large_chat_inference_v2.parquet" "chat_inference"
    load_parquet "./s3-fixtures/large_json_inference_v2.parquet" "json_inference"
    load_parquet "./s3-fixtures/large_chat_model_inference_v2.parquet" "model_inference"
    load_parquet "./s3-fixtures/large_json_model_inference_v2.parquet" "model_inference"
    load_parquet "./s3-fixtures/large_chat_boolean_feedback.parquet" "boolean_metric_feedback"
    load_parquet "./s3-fixtures/large_json_boolean_feedback.parquet" "boolean_metric_feedback"
    load_parquet "./s3-fixtures/large_chat_float_feedback.parquet" "float_metric_feedback"
    load_parquet "./s3-fixtures/large_json_float_feedback.parquet" "float_metric_feedback"
    load_parquet "./s3-fixtures/large_chat_comment_feedback.parquet" "comment_feedback"
    load_parquet "./s3-fixtures/large_json_comment_feedback.parquet" "comment_feedback"
    load_parquet "./s3-fixtures/large_chat_demonstration_feedback.parquet" "demonstration_feedback"
    load_parquet "./s3-fixtures/large_json_demonstration_feedback.parquet" "demonstration_feedback"

    echo "Large fixtures loaded."
fi

echo ""
echo "All fixtures loaded successfully!"

# Print row counts
psql -q "$POSTGRES_URL" <<EOF
SELECT 'chat_inference' as table_name, count(*) FROM chat_inference
UNION ALL SELECT 'json_inference', count(*) FROM json_inference
UNION ALL SELECT 'model_inference', count(*) FROM model_inference
UNION ALL SELECT 'boolean_metric_feedback', count(*) FROM boolean_metric_feedback
UNION ALL SELECT 'float_metric_feedback', count(*) FROM float_metric_feedback
UNION ALL SELECT 'comment_feedback', count(*) FROM comment_feedback
UNION ALL SELECT 'demonstration_feedback', count(*) FROM demonstration_feedback
UNION ALL SELECT 'chat_inference_datapoint', count(*) FROM chat_inference_datapoint
UNION ALL SELECT 'json_inference_datapoint', count(*) FROM json_inference_datapoint;
EOF
