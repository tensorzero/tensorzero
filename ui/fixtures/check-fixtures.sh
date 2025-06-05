#!/bin/bash

set -euo pipefail

DATABASE_NAME="${1:-tensorzero_ui_fixtures}"

echo "Verifying fixture counts for tables..."
echo "==============================================="

# Define tables and their corresponding files
declare -A all_tables
all_tables["JsonInference"]="json_inference_examples.jsonl ./s3-fixtures/large_json_inference_v2.parquet"
all_tables["BooleanMetricFeedback"]="boolean_metric_feedback_examples.jsonl ./s3-fixtures/large_chat_boolean_feedback.parquet ./s3-fixtures/large_json_boolean_feedback.parquet"
all_tables["BooleanMetricFeedbackByTargetId"]="boolean_metric_feedback_examples.jsonl ./s3-fixtures/large_chat_boolean_feedback.parquet ./s3-fixtures/large_json_boolean_feedback.parquet"
all_tables["FloatMetricFeedback"]="float_metric_feedback_examples.jsonl ./s3-fixtures/large_chat_float_feedback.parquet ./s3-fixtures/large_json_float_feedback.parquet"
all_tables["FloatMetricFeedbackByTargetId"]="float_metric_feedback_examples.jsonl ./s3-fixtures/large_chat_float_feedback.parquet ./s3-fixtures/large_json_float_feedback.parquet"
all_tables["CommentFeedback"]="comment_feedback_examples.jsonl ./s3-fixtures/large_chat_comment_feedback.parquet ./s3-fixtures/large_json_comment_feedback.parquet"
all_tables["DemonstrationFeedback"]="demonstration_feedback_examples.jsonl ./s3-fixtures/large_chat_demonstration_feedback.parquet ./s3-fixtures/large_json_demonstration_feedback.parquet"
all_tables["ChatInference"]="chat_inference_examples.jsonl ./s3-fixtures/large_chat_inference_v2.parquet"
all_tables["ModelInference"]="model_inference_examples.jsonl ./s3-fixtures/large_chat_model_inference_v2.parquet ./s3-fixtures/large_json_model_inference_v2.parquet"
all_tables["ChatInferenceDatapoint FINAL"]="chat_inference_datapoint_examples.jsonl"
all_tables["JsonInferenceDatapoint FINAL"]="json_inference_datapoint_examples.jsonl"
all_tables["ModelInferenceCache"]="model_inference_cache_examples.jsonl"
all_tables["DynamicEvaluationRun"]="dynamic_evaluation_run_examples.jsonl"
all_tables["DynamicEvaluationRunEpisode"]="dynamic_evaluation_run_episode_examples.jsonl"

# Track if there's any mismatch
mismatch=0

# Check each table
for table in "${!all_tables[@]}"; do
    # Get total count from database
    db_count=$(clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword \
              --database "$DATABASE_NAME" --query "SELECT count() FROM $table" 2>/dev/null || echo "ERROR")

    if [[ "$db_count" == "ERROR" ]]; then
        echo "  $table: ERROR accessing ClickHouse"
        mismatch=1
        continue
    fi

    # Get the list of files for this table
    read -r -a files <<< "${all_tables[$table]}"
    echo "Files: ${files[@]}"

    # Count total non-empty lines across all source files
    total_file_count=0
    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            if [[ "$file" == *.parquet ]]; then
                # For parquet files, use parquet-tools to count rows
                if command -v uv run parquet-tools &> /dev/null; then
                    file_count=$(uv run parquet-tools inspect "$file" | grep "num_rows:" | awk '{print $2}')
                    echo "  - $file: $file_count rows (parquet)"
                else
                    echo "  - WARNING: parquet-tools not installed, cannot count rows in $file"
                    mismatch=1
                fi
            else
                # For regular text files, count non-empty lines
                file_count=$(grep -v '^[[:space:]]*$' "$file" | wc -l)
                echo "  - $file: $file_count rows"
            fi
        else
            echo "  - WARNING: File $file not found in current directory or s3-fixtures/"
            mismatch=1
        fi
        total_file_count=$(($total_file_count + $file_count))
    done

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
        duplicates=$(clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword \
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
