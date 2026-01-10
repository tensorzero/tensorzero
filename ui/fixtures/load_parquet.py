#!/usr/bin/env python3
"""
Load a Parquet file into a PostgreSQL table using COPY for speed.

Usage: python load_parquet.py <parquet_file> <table_name> <postgres_url>
"""

import io
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import psycopg2


def fix_nested_json(value):
    """Fix and re-serialize nested JSON strings."""
    # Handle None, NaN, and empty strings - check scalar first to avoid array ambiguity
    if value is None:
        return None
    if np.isscalar(value) and pd.isna(value):
        return None
    if value == "":
        return None
    if not isinstance(value, str):
        return json.dumps(value) if isinstance(value, (dict, list)) else value

    # Escape literal control characters
    fixed = value.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

    try:
        parsed = json.loads(fixed)
        return json.dumps(parsed, ensure_ascii=False)
    except json.JSONDecodeError:
        return fixed


def serialize_json_value(value):
    """Serialize a value to JSON string."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        if value == "":
            return None
        # Already a string - escape control chars and validate
        fixed = value.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        try:
            parsed = json.loads(fixed)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            return fixed
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def serialize_jsonb_array(value):
    """Convert an array of JSON strings to a JSONB array string."""
    if value is None:
        return "[]"
    if isinstance(value, float) and pd.isna(value):
        return "[]"
    if isinstance(value, str):
        # Already a string, might be empty or serialized
        if value in ("", "[]", "{}"):
            return "[]"
        # Try to parse as JSON array
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return json.dumps(parsed, ensure_ascii=False)
            return "[]"
        except json.JSONDecodeError:
            return "[]"
    if isinstance(value, (list, np.ndarray)):
        if len(value) == 0:
            return "[]"
        # Each element should be a JSON string - parse and re-serialize as array
        result = []
        for elem in value:
            if elem is None or (isinstance(elem, float) and pd.isna(elem)):
                continue
            if isinstance(elem, str):
                try:
                    parsed = json.loads(elem)
                    result.append(parsed)
                except json.JSONDecodeError:
                    # Skip invalid JSON
                    pass
            elif isinstance(elem, (dict, list)):
                result.append(elem)
        return json.dumps(result, ensure_ascii=False)
    return "[]"


def format_value_for_copy(value, col_name: str, table_name: str, json_fields: set, jsonb_array_fields: set):
    """Format a value for direct COPY to PostgreSQL."""
    # Handle None/NaN
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None

    # JSONB array fields (arrays of JSON strings -> JSONB array)
    if col_name in jsonb_array_fields:
        return serialize_jsonb_array(value)

    # JSON fields
    if col_name in json_fields:
        return serialize_json_value(value)

    # finish_reason enum - map invalid values to 'other'
    if col_name == "finish_reason":
        val = str(value) if value else ""
        if val in ("", "nan", "None"):
            return None
        if val in ("stop", "length", "tool_calls", "content_filter", "other"):
            return val
        return "other"

    # target_type enum
    if col_name == "target_type":
        val = str(value) if value else ""
        if val in ("", "nan", "None"):
            return None
        return val

    # Boolean fields
    if col_name in ("cached", "value", "is_deleted", "is_custom"):
        if table_name == "boolean_metric_feedback" and col_name == "value":
            val = str(value).lower() if value else ""
            if val in ("", "nan", "none"):
                return None
            return "t" if val in ("true", "1", "t") else "f"
        if col_name == "cached":
            # cached defaults to false (NOT NULL)
            val = str(value).lower() if value else ""
            if val in ("", "nan", "none"):
                return "f"
            return "t" if val in ("true", "1", "t") else "f"
        if col_name in ("is_deleted", "is_custom"):
            # These default to false (NOT NULL)
            val = str(value).lower() if value else ""
            if val in ("", "nan", "none"):
                return "f"
            return "t" if val in ("true", "1", "t") else "f"

    # String value
    val = str(value) if value is not None else ""
    if val in ("", "nan", "None"):
        return None
    return val


# Column mappings for each table type
TABLE_CONFIGS = {
    "chat_inference": {
        "columns": [
            "id",
            "function_name",
            "variant_name",
            "episode_id",
            "input",
            "output",
            "tool_params",
            "inference_params",
            "processing_time_ms",
            "tags",
            "extra_body",
            "dynamic_tools",
            "dynamic_provider_tools",
        ],
        "json_fields": {"input", "output", "inference_params", "extra_body", "tags"},
        # Arrays of JSON strings that need to be converted to JSONB arrays
        "jsonb_array_fields": {"dynamic_tools", "dynamic_provider_tools"},
    },
    "json_inference": {
        "columns": [
            "id",
            "function_name",
            "variant_name",
            "episode_id",
            "input",
            "output",
            "output_schema",
            "inference_params",
            "processing_time_ms",
            "tags",
            "extra_body",
            "auxiliary_content",
        ],
        "json_fields": {"input", "output", "inference_params", "extra_body", "tags"},
        "jsonb_array_fields": set(),
    },
    "model_inference": {
        "columns": [
            "id",
            "inference_id",
            "raw_request",
            "raw_response",
            "model_name",
            "model_provider_name",
            "input_tokens",
            "output_tokens",
            "response_time_ms",
            "ttft_ms",
            "system",
            "input_messages",
            "output",
            "cached",
            "finish_reason",
        ],
        "json_fields": {"input_messages", "output"},
        "jsonb_array_fields": set(),
    },
    "boolean_metric_feedback": {
        "columns": ["id", "target_id", "metric_name", "value", "tags"],
        "json_fields": {"tags"},
        "jsonb_array_fields": set(),
    },
    "float_metric_feedback": {
        "columns": ["id", "target_id", "metric_name", "value", "tags"],
        "json_fields": {"tags"},
        "jsonb_array_fields": set(),
    },
    "comment_feedback": {
        "columns": ["id", "target_id", "target_type", "value", "tags"],
        "json_fields": {"tags"},
        "jsonb_array_fields": set(),
    },
    "demonstration_feedback": {
        "columns": ["id", "inference_id", "value", "tags"],
        "json_fields": {"tags"},
        "jsonb_array_fields": set(),
    },
}


# Configurable chunk size - COPY handles large batches well
CHUNK_SIZE = int(os.environ.get("TENSORZERO_CHUNK_SIZE", "500000"))


def load_parquet_to_postgres(parquet_path: str, table_name: str, postgres_url: str):
    """Load a parquet file into a PostgreSQL table using COPY directly."""
    config = TABLE_CONFIGS.get(table_name)
    if not config:
        raise ValueError(f"Unknown table: {table_name}")

    # Get parquet file metadata to know total rows
    parquet_file = pq.ParquetFile(parquet_path)
    total_rows = parquet_file.metadata.num_rows
    print(f"  Total rows: {total_rows}, chunk size: {CHUNK_SIZE}", flush=True)

    # Get columns that exist in both parquet and config
    all_columns = [c.name for c in parquet_file.schema_arrow]
    available_columns = [c for c in config["columns"] if c in all_columns]
    json_fields = config["json_fields"]
    jsonb_array_fields = config.get("jsonb_array_fields", set())

    # Connect to PostgreSQL
    conn = psycopg2.connect(postgres_url)
    total_inserted = 0

    try:
        # Process in chunks using row groups or batches
        chunk_num = 0
        for batch in parquet_file.iter_batches(
            batch_size=CHUNK_SIZE, columns=available_columns, use_threads=True
        ):
            chunk_num += 1
            print(f"  Chunk {chunk_num}: reading...", end="", flush=True)
            df = batch.to_pandas(self_destruct=True)
            chunk_rows = len(df)
            print(f" processing {chunk_rows} rows...", end="", flush=True)

            # Format each column for direct COPY
            for col in available_columns:
                df[col] = df[col].apply(
                    lambda v, c=col: format_value_for_copy(v, c, table_name, json_fields, jsonb_array_fields)
                )

            # Write to CSV buffer with proper NULL handling
            print(" csv...", end="", flush=True)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, header=False, na_rep="\\N")
            csv_buffer.seek(0)

            # COPY directly to target table
            print(" copying...", end="", flush=True)
            with conn.cursor() as cur:
                cur.copy_expert(
                    f"COPY {table_name} ({', '.join(available_columns)}) FROM STDIN WITH (FORMAT csv, NULL '\\N')",
                    csv_buffer
                )

            conn.commit()
            total_inserted += chunk_rows
            print(f" done. {total_inserted}/{total_rows} ({100*total_inserted//total_rows}%)", flush=True)

            # Free memory
            del df
            del csv_buffer

        print(f"  Inserted {total_inserted} rows")
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <parquet_file> <table_name> <postgres_url>")
        sys.exit(1)

    load_parquet_to_postgres(sys.argv[1], sys.argv[2], sys.argv[3])
