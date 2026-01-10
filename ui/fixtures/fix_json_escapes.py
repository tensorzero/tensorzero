#!/usr/bin/env python3
r"""
Fix invalid JSON escapes in JSONL files.

This script handles:
1. Invalid escape sequences like \! \d \_ etc. in the outer JSON
2. Nested JSON strings (like 'output' field) that contain invalid escapes
3. Literal newlines that should be escaped as \n
"""

import json
import re
import sys

# Placeholder using characters that won't appear in JSON
PLACEHOLDER = "\x00"


def fix_invalid_escapes(s: str) -> str:
    """Fix invalid JSON escape sequences by doubling the backslash."""
    # Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX
    # First protect valid escapes, then fix invalid ones, then restore

    # Protect valid escapes with placeholders
    protected = s
    protected = protected.replace("\\\\", f"{PLACEHOLDER}DOUBLE{PLACEHOLDER}")
    protected = protected.replace('\\"', f"{PLACEHOLDER}QUOTE{PLACEHOLDER}")
    protected = protected.replace("\\/", f"{PLACEHOLDER}SLASH{PLACEHOLDER}")
    protected = protected.replace("\\b", f"{PLACEHOLDER}B{PLACEHOLDER}")
    protected = protected.replace("\\f", f"{PLACEHOLDER}F{PLACEHOLDER}")
    protected = protected.replace("\\n", f"{PLACEHOLDER}N{PLACEHOLDER}")
    protected = protected.replace("\\r", f"{PLACEHOLDER}R{PLACEHOLDER}")
    protected = protected.replace("\\t", f"{PLACEHOLDER}T{PLACEHOLDER}")
    # Protect \uXXXX
    protected = re.sub(
        r"\\u([0-9a-fA-F]{4})", f"{PLACEHOLDER}U\\1{PLACEHOLDER}", protected
    )

    # Double any remaining backslashes (they're invalid escapes)
    protected = protected.replace("\\", "\\\\")

    # Restore valid escapes
    protected = protected.replace(f"{PLACEHOLDER}DOUBLE{PLACEHOLDER}", "\\\\")
    protected = protected.replace(f"{PLACEHOLDER}QUOTE{PLACEHOLDER}", '\\"')
    protected = protected.replace(f"{PLACEHOLDER}SLASH{PLACEHOLDER}", "\\/")
    protected = protected.replace(f"{PLACEHOLDER}B{PLACEHOLDER}", "\\b")
    protected = protected.replace(f"{PLACEHOLDER}F{PLACEHOLDER}", "\\f")
    protected = protected.replace(f"{PLACEHOLDER}N{PLACEHOLDER}", "\\n")
    protected = protected.replace(f"{PLACEHOLDER}R{PLACEHOLDER}", "\\r")
    protected = protected.replace(f"{PLACEHOLDER}T{PLACEHOLDER}", "\\t")
    protected = re.sub(
        f"{PLACEHOLDER}U([0-9a-fA-F]{{4}}){PLACEHOLDER}", r"\\u\1", protected
    )

    return protected


def fix_nested_json_string(value: str) -> str:
    """Fix a string that contains JSON and re-serialize it."""
    if not value or not value.strip():
        return value

    # First fix invalid escapes in the string
    fixed = fix_invalid_escapes(value)

    # Also escape literal control characters (newlines, tabs, etc.)
    # These need to be JSON escape sequences inside the string
    fixed = fixed.replace("\n", "\\n")
    fixed = fixed.replace("\r", "\\r")
    fixed = fixed.replace("\t", "\\t")

    try:
        # Try to parse as JSON
        parsed = json.loads(fixed)
        # Re-serialize to ensure proper escaping
        return json.dumps(parsed, ensure_ascii=False)
    except json.JSONDecodeError:
        # If it's not valid JSON after fixing, return the fixed string
        return fixed


def process_record(record: dict) -> dict:
    """Process a single record, fixing nested JSON fields."""
    result = {}

    # Fields that contain nested JSON strings
    nested_json_fields = {
        "input",
        "output",
        "inference_params",
        "extra_body",
        "input_messages",
        "raw_request",
        "raw_response",
        "tool_params",
        "output_schema",
        "auxiliary_content",
        "value",
    }

    for key, value in record.items():
        if key in nested_json_fields and isinstance(value, str) and value:
            result[key] = fix_nested_json_string(value)
        else:
            result[key] = value

    return result


def process_file(input_path: str):
    """Process a JSONL file and output fixed JSON to stdout."""
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # First fix escapes in the raw line
            fixed_line = fix_invalid_escapes(line)

            try:
                record = json.loads(fixed_line)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}", file=sys.stderr)
                print(f"Line: {fixed_line[:200]}...", file=sys.stderr)
                continue

            # Process nested JSON fields
            fixed_record = process_record(record)

            # Output as compact JSON
            print(json.dumps(fixed_record, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input.jsonl>", file=sys.stderr)
        sys.exit(1)

    process_file(sys.argv[1])
