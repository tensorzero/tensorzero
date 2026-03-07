#!/bin/bash

set -euo pipefail

DATABASE_NAME="${1:-tensorzero_ui_fixtures}"
CLICKHOUSE_HOST_VAR="${CLICKHOUSE_HOST}"
CLICKHOUSE_USER_VAR="${CLICKHOUSE_USER:-chuser}"
CLICKHOUSE_PASSWORD_VAR="${CLICKHOUSE_PASSWORD:-chpassword}"
CLICKHOUSE_SECURE_FLAG=""
if [ "${CLICKHOUSE_SECURE:-0}" = "1" ]; then
  CLICKHOUSE_SECURE_FLAG="--secure"
fi

echo "Verifying fixture counts for tables..."
echo "==============================================="

# Hardcoded row counts for large Native+LZ4 fixture files.
# To update these counts after regenerating fixtures, run:
#   for f in large-fixtures/*.parquet; do
#     echo "$f: $(clickhouse-local --query "SELECT count() FROM file('$f', 'Parquet')")"
#   done
declare -A large_fixture_counts
large_fixture_counts["large_chat_inference_v2.native.lz4"]=10000000
large_fixture_counts["large_chat_model_inference_v2.native.lz4"]=10000000
large_fixture_counts["large_json_inference_v2.native.lz4"]=10000000
large_fixture_counts["large_json_model_inference_v2.native.lz4"]=10000000
large_fixture_counts["large_chat_boolean_feedback.native.lz4"]=2500000
large_fixture_counts["large_chat_float_feedback.native.lz4"]=2500000
large_fixture_counts["large_chat_comment_feedback.native.lz4"]=2500000
large_fixture_counts["large_chat_demonstration_feedback.native.lz4"]=2500000
large_fixture_counts["large_json_boolean_feedback.native.lz4"]=2500000
large_fixture_counts["large_json_float_feedback.native.lz4"]=2500000
large_fixture_counts["large_json_comment_feedback.native.lz4"]=2500000
large_fixture_counts["large_json_demonstration_feedback.native.lz4"]=2500000

# Define tables and their corresponding JSONL files + large fixture file basenames
declare -A table_jsonl_files
table_jsonl_files["JsonInference"]="./small-fixtures/json_inference_examples.jsonl"
table_jsonl_files["BooleanMetricFeedback"]="./small-fixtures/boolean_metric_feedback_examples.jsonl"
table_jsonl_files["BooleanMetricFeedbackByTargetId"]="./small-fixtures/boolean_metric_feedback_examples.jsonl"
table_jsonl_files["FloatMetricFeedback"]="./small-fixtures/float_metric_feedback_examples.jsonl ./small-fixtures/jaro_winkler_similarity_feedback.jsonl"
table_jsonl_files["FloatMetricFeedbackByTargetId"]="./small-fixtures/float_metric_feedback_examples.jsonl ./small-fixtures/jaro_winkler_similarity_feedback.jsonl"
table_jsonl_files["CommentFeedback"]="./small-fixtures/comment_feedback_examples.jsonl"
table_jsonl_files["DemonstrationFeedback"]="./small-fixtures/demonstration_feedback_examples.jsonl"
table_jsonl_files["ChatInference"]="./small-fixtures/chat_inference_examples.jsonl"
table_jsonl_files["ModelInference"]="./small-fixtures/model_inference_examples.jsonl"
table_jsonl_files["ChatInferenceDatapoint FINAL"]="./small-fixtures/chat_inference_datapoint_examples.jsonl"
table_jsonl_files["JsonInferenceDatapoint FINAL"]="./small-fixtures/json_inference_datapoint_examples.jsonl"
table_jsonl_files["DynamicEvaluationRun"]="./small-fixtures/dynamic_evaluation_run_examples.jsonl"
table_jsonl_files["DynamicEvaluationRunEpisode"]="./small-fixtures/dynamic_evaluation_run_episode_examples.jsonl"

# Map tables to their large fixture file basenames
declare -A table_large_fixtures
table_large_fixtures["JsonInference"]="large_json_inference_v2.native.lz4"
table_large_fixtures["BooleanMetricFeedback"]="large_chat_boolean_feedback.native.lz4 large_json_boolean_feedback.native.lz4"
table_large_fixtures["BooleanMetricFeedbackByTargetId"]="large_chat_boolean_feedback.native.lz4 large_json_boolean_feedback.native.lz4"
table_large_fixtures["FloatMetricFeedback"]="large_chat_float_feedback.native.lz4 large_json_float_feedback.native.lz4"
table_large_fixtures["FloatMetricFeedbackByTargetId"]="large_chat_float_feedback.native.lz4 large_json_float_feedback.native.lz4"
table_large_fixtures["CommentFeedback"]="large_chat_comment_feedback.native.lz4 large_json_comment_feedback.native.lz4"
table_large_fixtures["DemonstrationFeedback"]="large_chat_demonstration_feedback.native.lz4 large_json_demonstration_feedback.native.lz4"
table_large_fixtures["ChatInference"]="large_chat_inference_v2.native.lz4"
table_large_fixtures["ModelInference"]="large_chat_model_inference_v2.native.lz4 large_json_model_inference_v2.native.lz4"

# Track if there's any mismatch
mismatch=0

# Check each table
for table in "${!table_jsonl_files[@]}"; do
    # Get total count from database
    db_count=$(clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG \
              --database "$DATABASE_NAME" --query "SELECT count() FROM $table" 2>/dev/null || echo "ERROR")

    if [[ "$db_count" == "ERROR" ]]; then
        echo "  $table: ERROR accessing ClickHouse"
        mismatch=1
        continue
    fi

    # Count JSONL rows
    read -r -a jsonl_files <<< "${table_jsonl_files[$table]}"
    total_file_count=0
    for file in "${jsonl_files[@]}"; do
        if [ -f "$file" ]; then
            file_count=$(grep -v '^[[:space:]]*$' "$file" | wc -l)
            echo "  - $file: $file_count rows"
            total_file_count=$(($total_file_count + $file_count))
        else
            echo "  - WARNING: File $file not found"
            mismatch=1
        fi
    done

    # Add hardcoded counts for large fixtures
    if [ -n "${table_large_fixtures[$table]+x}" ]; then
        read -r -a native_files <<< "${table_large_fixtures[$table]}"
        for native_file in "${native_files[@]}"; do
            count=${large_fixture_counts[$native_file]}
            echo "  - $native_file: $count rows (hardcoded)"
            total_file_count=$(($total_file_count + $count))
        done
    fi

    # Compare counts
    if [ "$db_count" -eq "$total_file_count" ]; then
        echo "  $table: OK (expected: $total_file_count, actual: $db_count rows)"
    else
        echo "  $table: MISMATCH (expected: $total_file_count, actual: $db_count rows)"
        mismatch=1
    fi
    echo
done



echo "==============================================="
if [ $mismatch -eq 0 ]; then
    echo "All fixture table counts match!"

    duplicate_found=0
    tables_to_check_duplicates=("FloatMetricFeedbackByTargetId" "BooleanMetricFeedbackByTargetId")

    for table in "${tables_to_check_duplicates[@]}"; do
        echo "Checking for duplicate ids in $table..."
        duplicates=$(clickhouse-client --host $CLICKHOUSE_HOST_VAR --user $CLICKHOUSE_USER_VAR --password $CLICKHOUSE_PASSWORD_VAR $CLICKHOUSE_SECURE_FLAG \
                      --database "$DATABASE_NAME" --query "SELECT
        id,
        count() AS duplicate_count
    FROM $table
    GROUP BY id
    HAVING duplicate_count > 1;" 2>/dev/null || echo "ERROR")

        if [[ "$duplicates" == "ERROR" ]]; then
            echo "  ERROR accessing ClickHouse for $table"
            duplicate_found=1 # Treat error as a potential problem
        elif [ -z "$duplicates" ]; then
            echo "  OK: No duplicate ids found in $table."
        else
            echo "  WARNING: Duplicate ids found in $table:"
            echo "$duplicates"
            duplicate_found=1
        fi
        echo
    done

    if [ $duplicate_found -eq 0 ]; then
        echo "No duplicate ids found in checked tables."
        exit 0
    else
        echo "Duplicate ids found or error occurred during check. Please review output."
        exit 1
    fi
else
    echo "Some fixture table counts don't match. Please check the output above."
    exit 1
fi
