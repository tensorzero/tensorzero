#!/bin/bash
# Converts all large fixture Parquet files to ClickHouse Native+LZ4 format.
#
# Prerequisites:
#   - Docker must be running
#   - Parquet files must exist in ./large-fixtures/
#
# Usage:
#   ./convert-parquet-to-native.sh
#
# After conversion, upload both formats to R2:
#   ./upload-large-fixtures.sh

set -euxo pipefail

cd "$(dirname "$0")"

LARGE_FIXTURES_DIR="./large-fixtures"
ABSOLUTE_FIXTURES_DIR="$(cd "$LARGE_FIXTURES_DIR" && pwd)"

PARQUET_FILES=(
    "large_chat_inference_v2.parquet"
    "large_chat_model_inference_v2.parquet"
    "large_json_inference_v2.parquet"
    "large_json_model_inference_v2.parquet"
    "large_chat_boolean_feedback.parquet"
    "large_chat_float_feedback.parquet"
    "large_chat_comment_feedback.parquet"
    "large_chat_demonstration_feedback.parquet"
    "large_json_boolean_feedback.parquet"
    "large_json_float_feedback.parquet"
    "large_json_comment_feedback.parquet"
    "large_json_demonstration_feedback.parquet"
)

for parquet_file in "${PARQUET_FILES[@]}"; do
    input="${LARGE_FIXTURES_DIR}/${parquet_file}"
    native_name="${parquet_file%.parquet}.native.lz4"
    output="${LARGE_FIXTURES_DIR}/${native_name}"

    if [ ! -f "$input" ]; then
        echo "ERROR: Input file not found: $input"
        exit 1
    fi

    echo "Converting: $parquet_file -> $native_name"

    docker run --rm \
        -v "${ABSOLUTE_FIXTURES_DIR}:/fixtures" \
        clickhouse/clickhouse-server \
        clickhouse-local --query \
        "SELECT * FROM file('/fixtures/${parquet_file}', 'Parquet') INTO OUTFILE '/fixtures/${native_name}' TRUNCATE FORMAT Native SETTINGS input_format_parquet_use_native_reader_v3=0"

    input_size=$(stat -f%z "$input" 2>/dev/null || stat -c%s "$input" 2>/dev/null)
    output_size=$(stat -f%z "$output" 2>/dev/null || stat -c%s "$output" 2>/dev/null)
    echo "  Parquet: $((input_size / 1024 / 1024)) MB -> Native+LZ4: $((output_size / 1024 / 1024)) MB"
done

echo "Conversion complete."
