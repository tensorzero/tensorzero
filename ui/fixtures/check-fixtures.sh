#!/bin/bash
# filepath: /home/aaron/repos/tensorzero/ui/fixtures/verify_feedback_fixtures.sh

set -euo pipefail

echo "Verifying fixture counts for feedback tables..."
echo "==============================================="

# Define feedback tables and their corresponding files
declare -A feedback_tables
feedback_tables["BooleanMetricFeedback"]="boolean_metric_feedback_examples.jsonl"
feedback_tables["FloatMetricFeedback"]="float_metric_feedback_examples.jsonl"
feedback_tables["CommentFeedback"]="comment_feedback_examples.jsonl"
feedback_tables["DemonstrationFeedback"]="demonstration_feedback_examples.jsonl"

# Track if there's any mismatch
mismatch=0

# Check each feedback table
for table in "${!feedback_tables[@]}"; do
    file="${feedback_tables[$table]}"
    
    # Count non-empty lines in the file
    file_count=$(grep -v '^[[:space:]]*$' "$file" | wc -l)
    
    # Get row count from database
    db_count=$(clickhouse-client --host $CLICKHOUSE_HOST --user chuser --password chpassword \
              --database tensorzero_ui_fixtures --query "SELECT count() FROM $table" 2>/dev/null || echo "ERROR")
    
    # Compare counts
    if [[ "$db_count" == "ERROR" ]]; then
        echo "  $file -> $table: ERROR accessing ClickHouse"
        mismatch=1
    elif [ "$db_count" -eq "$file_count" ]; then
        echo "  $file -> $table: OK ($file_count rows)"
    else
        echo "  $file -> $table: MISMATCH (file: $file_count, DB: $db_count)"
        mismatch=1
    fi
done

echo "==============================================="
if [ $mismatch -eq 0 ]; then
    echo "All feedback fixture counts match!"
    exit 0
else
    echo "Some feedback fixture counts don't match. Please check the output above."
    exit 1
fi