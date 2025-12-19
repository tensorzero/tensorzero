#!/bin/bash
set -euxo pipefail

cd "$(dirname "$0")"

# List of JSONL files to upload
# When updating a file, rename it with a version suffix (e.g., model_inference_examples_v2.jsonl)
# and update download-small-fixtures.py to use the new filename
JSONL_FILES=(
    "model_inference_examples.jsonl"
    "chat_inference_examples.jsonl"
    "json_inference_examples.jsonl"
    "boolean_metric_feedback_examples.jsonl"
    "float_metric_feedback_examples.jsonl"
    "demonstration_feedback_examples.jsonl"
    "model_inference_cache_e2e.jsonl"
    "json_inference_datapoint_examples.jsonl"
    "chat_inference_datapoint_examples.jsonl"
    "dynamic_evaluation_run_episode_examples.jsonl"
    "jaro_winkler_similarity_feedback.jsonl"
    "comment_feedback_examples.jsonl"
    "dynamic_evaluation_run_examples.jsonl"
)

# Upload each file
for file in "${JSONL_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "Uploading $file..."
        aws s3 --endpoint-url https://19918a216783f0ac9e052233569aef60.r2.cloudflarestorage.com/ \
            cp "$file" "s3://tensorzero-fixtures/${file}"
    else
        echo "Warning: $file not found, skipping"
    fi
done

echo "Done uploading JSONL fixtures"
