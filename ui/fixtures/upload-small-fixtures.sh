#!/bin/bash
set -euxo pipefail

cd "$(dirname "$0")"

# Generate timestamp for this upload
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
echo "Using version: $TIMESTAMP"

# List of JSONL files to upload
# These are uploaded with path-based versioning: {timestamp}/{filename}
JSONL_FILES=(
    "boolean_metric_feedback_examples.jsonl"
    "chat_inference_datapoint_examples.jsonl"
    "chat_inference_examples.jsonl"
    "comment_feedback_examples.jsonl"
    "demonstration_feedback_examples.jsonl"
    "dynamic_evaluation_run_episode_examples.jsonl"
    "dynamic_evaluation_run_examples.jsonl"
    "float_metric_feedback_examples.jsonl"
    "jaro_winkler_similarity_feedback.jsonl"
    "json_inference_datapoint_examples.jsonl"
    "json_inference_examples.jsonl"
    # "model_inference_cache_e2e.jsonl" -- handled by /regen-fixtures in GitHub
    "model_inference_examples.jsonl"
)

# Check that all files exist before uploading
MISSING_FILES=()
for file in "${JSONL_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    echo "Error: The following files are missing:"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "Run 'uv run download-small-fixtures.py' first to download existing fixtures."
    exit 1
fi

# Upload each file to versioned path
for file in "${JSONL_FILES[@]}"; do
    echo "Uploading $file to ${TIMESTAMP}/${file}..."
    aws s3 --endpoint-url https://19918a216783f0ac9e052233569aef60.r2.cloudflarestorage.com/ \
        cp "$file" "s3://tensorzero-fixtures/${TIMESTAMP}/${file}"
done

# Update the version manifest
echo "$TIMESTAMP" > fixtures-version.txt
echo "Updated fixtures-version.txt with version: $TIMESTAMP"

echo "Done uploading JSONL fixtures"
echo "Don't forget to commit fixtures-version.txt!"
