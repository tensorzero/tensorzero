# /// script
# dependencies = [
#   "pyarrow",
# ]
# ///

"""Convert large parquet fixtures to Postgres-compatible parquet files for pg_parquet.

This script reads parquet files from large-fixtures/ and converts them to parquet
format with Postgres-compatible types (e.g. UUIDs as binary(16), timestamps as
microsecond-precision Arrow timestamps) suitable for pg_parquet's COPY FROM.

Output files are written to large-fixtures/postgres-parquet/.

Usage:
    uv run ./load_large_fixtures_postgres.py [--processes N] [--batch-size N]

The shell script load_fixtures_postgres.sh handles COPY commands via pg_parquet.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# =====================================================================
# Column-level conversion helpers (operate on whole columns at once)
# =====================================================================


def convert_uuids(uuid_strings: list[str]) -> list[bytes]:
    """Convert UUID strings to 16-byte binary for pg_parquet."""
    return [bytes.fromhex(u.replace("-", "")) for u in uuid_strings]


def convert_timestamps_from_uuidv7(uuid_strings: list[str]) -> list[datetime]:
    """Extract timestamps from UUIDv7 strings as UTC datetimes."""
    return [datetime.fromtimestamp(int(u.replace("-", "")[:12], 16) / 1000, tz=timezone.utc) for u in uuid_strings]


def convert_tags(tags_list: list) -> list[str]:
    """Convert pyarrow map values to JSON strings."""
    return [_map_to_json(t) for t in tags_list]


def convert_arrays_to_json(arr_list: list) -> list[str]:
    """Convert pyarrow list/array values to JSON strings."""
    return [json.dumps(list(x) if x else []) for x in arr_list]


def decode_finish_reason(column: pa.ChunkedArray) -> pa.ChunkedArray:
    """Decode a ClickHouse Enum8 finish_reason column from parquet.

    ClickHouse exports Enum8 as raw int8 in parquet. We map the integer values
    back to their string labels based on the ClickHouse schema:
    'stop' = 1, 'length' = 2, 'tool_call' = 3, 'content_filter' = 4,
    'unknown' = 5, 'stop_sequence' = 6
    """
    if pa.types.is_dictionary(column.type):
        return column.cast(pa.utf8())
    if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
        return column
    # ClickHouse Enum8 exported as int8
    mapping = {1: "stop", 2: "length", 3: "tool_call", 4: "content_filter", 5: "unknown", 6: "stop_sequence"}
    return pa.chunked_array(
        [pa.array([mapping.get(v) if v is not None else None for v in column.to_pylist()], type=pa.utf8())]
    )


def _map_to_json(tags_map) -> str:
    if tags_map is None:
        return "{}"
    result = {}
    for item in tags_map:
        if item is not None:
            key, value = item
            if key is not None:
                result[str(key)] = str(value) if value is not None else None
    return json.dumps(result)


def get_column(table, name, default=None):
    """Get a column's Python list, or a list of defaults if missing."""
    if name in table.column_names:
        return table[name].to_pylist()
    return [default] * table.num_rows


# =====================================================================
# Shared builders for common column patterns
# =====================================================================


def build_id_and_timestamp(table):
    """Build id (binary) and created_at (timestamp) columns from UUIDv7 ids."""
    ids_raw = table["id"].to_pylist()
    return (
        pa.array(convert_uuids(ids_raw), type=pa.binary(16)),
        pa.array(convert_timestamps_from_uuidv7(ids_raw), type=pa.timestamp("us", tz="UTC")),
    )


# =====================================================================
# Feedback converters
# =====================================================================

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


def convert_boolean_feedback(tables):
    parts = []
    for table in tables:
        ids, created_at = build_id_and_timestamp(table)
        parts.append(
            pa.table(
                {
                    "id": ids,
                    "target_id": pa.array(convert_uuids(table["target_id"].to_pylist()), type=pa.binary(16)),
                    "metric_name": table["metric_name"].cast(pa.utf8()),
                    "value": table["value"].cast(pa.bool_()),
                    "tags": pa.array(convert_tags(table["tags"].to_pylist()), type=pa.utf8()),
                    "created_at": created_at,
                }
            )
        )
    return {"boolean_metric_feedback": pa.concat_tables(parts)}


def convert_float_feedback(tables):
    parts = []
    for table in tables:
        ids, created_at = build_id_and_timestamp(table)
        parts.append(
            pa.table(
                {
                    "id": ids,
                    "target_id": pa.array(convert_uuids(table["target_id"].to_pylist()), type=pa.binary(16)),
                    "metric_name": table["metric_name"].cast(pa.utf8()),
                    "value": table["value"].cast(pa.float64()),
                    "tags": pa.array(convert_tags(table["tags"].to_pylist()), type=pa.utf8()),
                    "created_at": created_at,
                }
            )
        )
    return {"float_metric_feedback": pa.concat_tables(parts)}


def convert_comment_feedback(tables):
    parts = []
    for table in tables:
        ids, created_at = build_id_and_timestamp(table)
        target_types = table["target_type"].to_pylist()
        parts.append(
            pa.table(
                {
                    "id": ids,
                    "target_id": pa.array(convert_uuids(table["target_id"].to_pylist()), type=pa.binary(16)),
                    "target_type": pa.array(
                        ["inference" if t == 1 else "episode" for t in target_types],
                        type=pa.utf8(),
                    ),
                    "value": pa.array(
                        [v if v else "" for v in table["value"].to_pylist()],
                        type=pa.utf8(),
                    ),
                    "tags": pa.array(convert_tags(table["tags"].to_pylist()), type=pa.utf8()),
                    "created_at": created_at,
                }
            )
        )
    return {"comment_feedback": pa.concat_tables(parts)}


def convert_demonstration_feedback(tables):
    parts = []
    for table in tables:
        ids, created_at = build_id_and_timestamp(table)
        parts.append(
            pa.table(
                {
                    "id": ids,
                    "inference_id": pa.array(convert_uuids(table["inference_id"].to_pylist()), type=pa.binary(16)),
                    "value": pa.array(
                        [v if v else None for v in table["value"].to_pylist()],
                        type=pa.utf8(),
                    ),
                    "tags": pa.array(convert_tags(table["tags"].to_pylist()), type=pa.utf8()),
                    "created_at": created_at,
                }
            )
        )
    return {"demonstration_feedback": pa.concat_tables(parts)}


# =====================================================================
# Inference converters
# =====================================================================

INFERENCE_FILES = {
    "chat_inference": ["large_chat_inference_v2.parquet"],
    "json_inference": ["large_json_inference_v2.parquet"],
    "model_inference": [
        "large_chat_model_inference_v2.parquet",
        "large_json_model_inference_v2.parquet",
    ],
}


def _coalesce_str(values, default="{}"):
    """Replace None/empty with default for NOT NULL JSONB columns."""
    return [v if v else default for v in values]


def convert_chat_inferences(tables):
    meta_parts = []
    data_parts = []
    for table in tables:
        ids, created_at = build_id_and_timestamp(table)
        episode_ids = pa.array(convert_uuids(table["episode_id"].to_pylist()), type=pa.binary(16))

        meta_parts.append(
            pa.table(
                {
                    "id": ids,
                    "function_name": table["function_name"].cast(pa.utf8()),
                    "variant_name": table["variant_name"].cast(pa.utf8()),
                    "episode_id": episode_ids,
                    "processing_time_ms": table["processing_time_ms"].cast(pa.int32()),
                    "ttft_ms": table["ttft_ms"].cast(pa.int32())
                    if "ttft_ms" in table.column_names
                    else pa.nulls(table.num_rows, type=pa.int32()),
                    "tags": pa.array(convert_tags(table["tags"].to_pylist()), type=pa.utf8()),
                    "created_at": created_at,
                }
            )
        )

        data_parts.append(
            pa.table(
                {
                    "id": ids,
                    "input": pa.array(_coalesce_str(table["input"].to_pylist(), "{}"), type=pa.utf8()),
                    "output": pa.array(_coalesce_str(table["output"].to_pylist(), "[]"), type=pa.utf8()),
                    "inference_params": pa.array(
                        _coalesce_str(table["inference_params"].to_pylist(), "{}"), type=pa.utf8()
                    ),
                    "extra_body": pa.array(_coalesce_str(get_column(table, "extra_body"), "[]"), type=pa.utf8()),
                    "dynamic_tools": pa.array(
                        convert_arrays_to_json(get_column(table, "dynamic_tools", [])), type=pa.utf8()
                    ),
                    "dynamic_provider_tools": pa.array(
                        convert_arrays_to_json(get_column(table, "dynamic_provider_tools", [])), type=pa.utf8()
                    ),
                    "allowed_tools": pa.array(get_column(table, "allowed_tools"), type=pa.utf8()),
                    "tool_choice": pa.array(get_column(table, "tool_choice"), type=pa.utf8()),
                    "parallel_tool_calls": pa.array(get_column(table, "parallel_tool_calls"), type=pa.bool_()),
                    "created_at": created_at,
                }
            )
        )

    return {
        "chat_inferences": pa.concat_tables(meta_parts),
        "chat_inference_data": pa.concat_tables(data_parts),
    }


def convert_json_inferences(tables):
    meta_parts = []
    data_parts = []
    for table in tables:
        ids, created_at = build_id_and_timestamp(table)
        episode_ids = pa.array(convert_uuids(table["episode_id"].to_pylist()), type=pa.binary(16))

        meta_parts.append(
            pa.table(
                {
                    "id": ids,
                    "function_name": table["function_name"].cast(pa.utf8()),
                    "variant_name": table["variant_name"].cast(pa.utf8()),
                    "episode_id": episode_ids,
                    "processing_time_ms": table["processing_time_ms"].cast(pa.int32()),
                    "ttft_ms": table["ttft_ms"].cast(pa.int32())
                    if "ttft_ms" in table.column_names
                    else pa.nulls(table.num_rows, type=pa.int32()),
                    "tags": pa.array(convert_tags(table["tags"].to_pylist()), type=pa.utf8()),
                    "created_at": created_at,
                }
            )
        )

        data_parts.append(
            pa.table(
                {
                    "id": ids,
                    "input": pa.array(_coalesce_str(table["input"].to_pylist(), "{}"), type=pa.utf8()),
                    "output": pa.array(_coalesce_str(table["output"].to_pylist(), "{}"), type=pa.utf8()),
                    "output_schema": pa.array(_coalesce_str(get_column(table, "output_schema"), "{}"), type=pa.utf8()),
                    "inference_params": pa.array(
                        _coalesce_str(table["inference_params"].to_pylist(), "{}"), type=pa.utf8()
                    ),
                    "extra_body": pa.array(_coalesce_str(get_column(table, "extra_body"), "[]"), type=pa.utf8()),
                    "auxiliary_content": pa.array(
                        _coalesce_str(get_column(table, "auxiliary_content"), "{}"), type=pa.utf8()
                    ),
                    "created_at": created_at,
                }
            )
        )

    return {
        "json_inferences": pa.concat_tables(meta_parts),
        "json_inference_data": pa.concat_tables(data_parts),
    }


def convert_model_inferences(tables):
    meta_parts = []
    data_parts = []
    for table in tables:
        ids, created_at = build_id_and_timestamp(table)
        inference_ids = pa.array(convert_uuids(table["inference_id"].to_pylist()), type=pa.binary(16))

        meta_parts.append(
            pa.table(
                {
                    "id": ids,
                    "inference_id": inference_ids,
                    "input_tokens": table["input_tokens"].cast(pa.int32()),
                    "output_tokens": table["output_tokens"].cast(pa.int32()),
                    "response_time_ms": table["response_time_ms"].cast(pa.int32()),
                    "model_name": table["model_name"].cast(pa.utf8()),
                    "model_provider_name": table["model_provider_name"].cast(pa.utf8()),
                    "ttft_ms": table["ttft_ms"].cast(pa.int32())
                    if "ttft_ms" in table.column_names
                    else pa.nulls(table.num_rows, type=pa.int32()),
                    "cached": table["cached"].cast(pa.bool_()),
                    "finish_reason": decode_finish_reason(table["finish_reason"]),
                    "created_at": created_at,
                }
            )
        )

        data_parts.append(
            pa.table(
                {
                    "id": ids,
                    "raw_request": table["raw_request"].cast(pa.utf8()),
                    "raw_response": table["raw_response"].cast(pa.utf8()),
                    "system": table["system"].cast(pa.utf8())
                    if "system" in table.column_names
                    else pa.nulls(table.num_rows, type=pa.utf8()),
                    "input_messages": pa.array(
                        _coalesce_str(get_column(table, "input_messages"), "[]"), type=pa.utf8()
                    ),
                    "output": pa.array(_coalesce_str(get_column(table, "output"), "[]"), type=pa.utf8()),
                    "created_at": created_at,
                }
            )
        )

    return {
        "model_inferences": pa.concat_tables(meta_parts),
        "model_inference_data": pa.concat_tables(data_parts),
    }


# =====================================================================
# Converter registry
# =====================================================================

# Maps a logical group name to (source_files, output_table_names, converter_function)
# Converters return {output_table_name: pa.Table}
ALL_FILE_GROUPS = {
    "boolean_metric_feedback": (
        FEEDBACK_FILES["boolean_metric_feedback"],
        ["boolean_metric_feedback"],
        convert_boolean_feedback,
    ),
    "float_metric_feedback": (
        FEEDBACK_FILES["float_metric_feedback"],
        ["float_metric_feedback"],
        convert_float_feedback,
    ),
    "comment_feedback": (FEEDBACK_FILES["comment_feedback"], ["comment_feedback"], convert_comment_feedback),
    "demonstration_feedback": (
        FEEDBACK_FILES["demonstration_feedback"],
        ["demonstration_feedback"],
        convert_demonstration_feedback,
    ),
    "chat_inference": (
        INFERENCE_FILES["chat_inference"],
        ["chat_inferences", "chat_inference_data"],
        convert_chat_inferences,
    ),
    "json_inference": (
        INFERENCE_FILES["json_inference"],
        ["json_inferences", "json_inference_data"],
        convert_json_inferences,
    ),
    "model_inference": (
        INFERENCE_FILES["model_inference"],
        ["model_inferences", "model_inference_data"],
        convert_model_inferences,
    ),
}

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

LARGE_FIXTURES_DIR = Path("./large-fixtures")
OUTPUT_DIR = LARGE_FIXTURES_DIR / "postgres-parquet"


def process_group(group_name, batch_size):
    """Process a single file group. Returns (group_name, rows_converted, was_skipped)."""
    parquet_files, output_tables, converter = ALL_FILE_GROUPS[group_name]

    all_exist = all((OUTPUT_DIR / f"{name}.parquet").exists() for name in output_tables)
    if all_exist:
        print(f"\nSkipping {group_name} (output files already exist)", flush=True)
        return (group_name, 0, True)

    group_rows = 0
    writers = {}
    rows_written = {table_name: 0 for table_name in output_tables}

    try:
        for parquet_file in parquet_files:
            file_path = LARGE_FIXTURES_DIR / parquet_file
            parquet_source = pq.ParquetFile(file_path)
            source_rows = parquet_source.metadata.num_rows
            group_rows += source_rows
            print(
                f"  [{group_name}] Reading {parquet_file} ({source_rows:,} rows) in batches of {batch_size:,}...",
                flush=True,
            )

            for batch in parquet_source.iter_batches(batch_size=batch_size):
                source_table = pa.Table.from_batches([batch])
                result_tables = converter([source_table])

                for table_name, arrow_table in result_tables.items():
                    if table_name not in writers:
                        out_path = OUTPUT_DIR / f"{table_name}.parquet"
                        writers[table_name] = pq.ParquetWriter(out_path, arrow_table.schema)
                    writers[table_name].write_table(arrow_table)
                    rows_written[table_name] += arrow_table.num_rows
    finally:
        for writer in writers.values():
            writer.close()

    for table_name in output_tables:
        out_path = OUTPUT_DIR / f"{table_name}.parquet"
        if not out_path.exists():
            print(
                f"  [{group_name}] Wrote {table_name}.parquet: {rows_written[table_name]:,} rows",
                flush=True,
            )
            continue
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(
            f"  [{group_name}] Wrote {out_path.name}: {rows_written[table_name]:,} rows, {size_mb:.1f} MB",
            flush=True,
        )

    return (group_name, group_rows, False)


def main():
    parser = argparse.ArgumentParser(description="Convert large parquet fixtures to Postgres-compatible format.")
    parser.add_argument("--processes", type=int, default=4, help="Number of parallel worker processes (default: 4)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Rows per chunk when reading source parquet files (default: 50000)",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        print("--batch-size must be greater than 0", flush=True)
        return 1

    # Collect all source files
    all_source_files = []
    for files, _, _ in ALL_FILE_GROUPS.values():
        all_source_files.extend(files)

    missing_files = [f for f in all_source_files if not (LARGE_FIXTURES_DIR / f).exists()]
    if missing_files:
        print(
            "Large fixture files not found. Run download-large-fixtures.py first.",
            flush=True,
        )
        print(f"Missing: {', '.join(missing_files)}", flush=True)
        return 1

    OUTPUT_DIR.mkdir(exist_ok=True)

    num_workers = min(args.processes, len(ALL_FILE_GROUPS))
    print(
        f"Processing {len(ALL_FILE_GROUPS)} file groups with {num_workers} workers (batch size: {args.batch_size:,})...",
        flush=True,
    )

    total_rows = 0
    skipped = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_group, group_name, args.batch_size): group_name for group_name in ALL_FILE_GROUPS
        }
        for future in as_completed(futures):
            group_name, rows, was_skipped = future.result()
            total_rows += rows
            if was_skipped:
                skipped += 1

    print(f"\n{'=' * 60}", flush=True)
    if total_rows > 0:
        print(f"Total rows converted: {total_rows:,}", flush=True)
    if skipped > 0:
        print(f"Skipped {skipped} group(s) with existing output", flush=True)
    print(f"Parquet files location: {OUTPUT_DIR}", flush=True)
    print(f"{'=' * 60}", flush=True)
    return 0


if __name__ == "__main__":
    exit(main())
