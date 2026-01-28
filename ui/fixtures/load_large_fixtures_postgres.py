# /// script
# dependencies = [
#   "pyarrow",
# ]
# ///

"""Convert large parquet fixtures to CSV files for Postgres COPY.

This script reads parquet files from large-fixtures/ and converts them to CSV
format suitable for Postgres COPY. The CSV files are written to
large-fixtures/postgres-csv/ and can be mounted into a Postgres container.

Usage:
    uv run ./load_large_fixtures_postgres.py

The shell script load_fixtures_postgres.sh handles mounting and COPY commands.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


def uuid_v7_to_timestamp(uuid_str: str) -> str:
    """Extract timestamp from UUIDv7 and return as RFC 3339 string."""
    # UUIDv7 has the timestamp in the first 48 bits (first 12 hex chars)
    hex_str = uuid_str.replace("-", "")
    timestamp_ms = int(hex_str[:12], 16)
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

LARGE_FIXTURES_DIR = Path("./large-fixtures")
OUTPUT_DIR = LARGE_FIXTURES_DIR / "postgres-csv"

# Mapping of output CSV files to their source parquet files
FEEDBACK_FILES = {
    "boolean_metric_feedback": [
        "large_chat_boolean_feedback.parquet",
        "large_json_boolean_feedback.parquet",
    ],
    "float_metric_feedback": [
        "large_chat_float_feedback.parquet",
        "large_json_float_feedback.parquet",
    ],
    "comment_feedback": [
        "large_chat_comment_feedback.parquet",
        "large_json_comment_feedback.parquet",
    ],
    "demonstration_feedback": [
        "large_chat_demonstration_feedback.parquet",
        "large_json_demonstration_feedback.parquet",
    ],
}


def map_to_json(tags_map) -> str:
    """Convert pyarrow map to JSON string."""
    if tags_map is None:
        return "{}"
    # tags_map is a list of (key, value) tuples
    result = {}
    for item in tags_map:
        if item is not None:
            key, value = item
            if key is not None:
                result[str(key)] = str(value) if value is not None else None
    return json.dumps(result)


def target_type_to_string(target_type: int) -> str:
    """Convert target_type int to string."""
    return "episode" if target_type == 1 else "inference"


def convert_boolean_feedback(table, out_file):
    """Convert boolean feedback parquet to CSV."""
    ids = table["id"].to_pylist()
    target_ids = table["target_id"].to_pylist()
    metric_names = table["metric_name"].to_pylist()
    values = table["value"].to_pylist()
    tags_list = table["tags"].to_pylist()

    for row_id, target_id, metric_name, value, tags in zip(ids, target_ids, metric_names, values, tags_list):
        value_str = "true" if value else "false"
        tags_json = map_to_json(tags)
        tags_escaped = tags_json.replace('"', '""')
        created_at = uuid_v7_to_timestamp(row_id)
        out_file.write(f'{row_id},{target_id},{metric_name},{value_str},"{tags_escaped}",{created_at}\n')


def convert_float_feedback(table, out_file):
    """Convert float feedback parquet to CSV."""
    ids = table["id"].to_pylist()
    target_ids = table["target_id"].to_pylist()
    metric_names = table["metric_name"].to_pylist()
    values = table["value"].to_pylist()
    tags_list = table["tags"].to_pylist()

    for row_id, target_id, metric_name, value, tags in zip(ids, target_ids, metric_names, values, tags_list):
        tags_json = map_to_json(tags)
        tags_escaped = tags_json.replace('"', '""')
        created_at = uuid_v7_to_timestamp(row_id)
        out_file.write(f'{row_id},{target_id},{metric_name},{value},"{tags_escaped}",{created_at}\n')


def convert_comment_feedback(table, out_file):
    """Convert comment feedback parquet to CSV."""
    ids = table["id"].to_pylist()
    target_ids = table["target_id"].to_pylist()
    target_types = table["target_type"].to_pylist()
    values = table["value"].to_pylist()
    tags_list = table["tags"].to_pylist()

    for row_id, target_id, target_type, value, tags in zip(ids, target_ids, target_types, values, tags_list):
        target_type_str = target_type_to_string(target_type)
        value_escaped = value.replace('"', '""') if value else ""
        tags_json = map_to_json(tags)
        tags_escaped = tags_json.replace('"', '""')
        created_at = uuid_v7_to_timestamp(row_id)
        out_file.write(f'{row_id},{target_id},{target_type_str},"{value_escaped}","{tags_escaped}",{created_at}\n')


def convert_demonstration_feedback(table, out_file):
    """Convert demonstration feedback parquet to CSV."""
    ids = table["id"].to_pylist()
    inference_ids = table["inference_id"].to_pylist()
    values = table["value"].to_pylist()
    tags_list = table["tags"].to_pylist()

    for row_id, inference_id, value, tags in zip(ids, inference_ids, values, tags_list):
        value_escaped = value.replace('"', '""') if value else ""
        tags_json = map_to_json(tags)
        tags_escaped = tags_json.replace('"', '""')
        created_at = uuid_v7_to_timestamp(row_id)
        out_file.write(f'{row_id},{inference_id},"{value_escaped}","{tags_escaped}",{created_at}\n')


CONVERTERS = {
    "boolean_metric_feedback": convert_boolean_feedback,
    "float_metric_feedback": convert_float_feedback,
    "comment_feedback": convert_comment_feedback,
    "demonstration_feedback": convert_demonstration_feedback,
}


def main():
    # Check if parquet files exist
    missing_files = []
    for files in FEEDBACK_FILES.values():
        for f in files:
            if not (LARGE_FIXTURES_DIR / f).exists():
                missing_files.append(f)

    if missing_files:
        print(
            "Large fixture files not found. Run download-large-fixtures.py first.",
            flush=True,
        )
        print(f"Missing: {', '.join(missing_files)}", flush=True)
        return 1

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    total_rows = 0
    skipped = 0
    for table_name, parquet_files in FEEDBACK_FILES.items():
        csv_path = OUTPUT_DIR / f"{table_name}.csv"

        # Skip if CSV already exists
        if csv_path.exists():
            size_mb = csv_path.stat().st_size / (1024 * 1024)
            print(f"\nSkipping {csv_path.name} (already exists, {size_mb:.1f} MB)", flush=True)
            skipped += 1
            continue

        print(f"\nConverting to {csv_path.name}:", flush=True)

        converter = CONVERTERS[table_name]

        with open(csv_path, "w") as out_file:
            for parquet_file in parquet_files:
                file_path = LARGE_FIXTURES_DIR / parquet_file
                print(f"  Reading {parquet_file}...", flush=True)
                table = pq.read_table(file_path)
                num_rows = table.num_rows
                print(f"    {num_rows:,} rows", flush=True)

                print(f"  Converting...", flush=True)
                converter(table, out_file)
                total_rows += num_rows

        # Print file size
        size_mb = csv_path.stat().st_size / (1024 * 1024)
        print(f"  Written: {size_mb:.1f} MB", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    if total_rows > 0:
        print(f"Total rows converted: {total_rows:,}", flush=True)
    if skipped > 0:
        print(f"Skipped {skipped} existing CSV file(s)", flush=True)
    print(f"CSV files location: {OUTPUT_DIR}", flush=True)
    print(f"{'=' * 60}", flush=True)
    return 0


if __name__ == "__main__":
    exit(main())
