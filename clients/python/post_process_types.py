#!/usr/bin/env python3
"""
Post-process generated types to handle Option<Option<T>> from Rust.

For fields that can distinguish between:
- Omitted (don't change): represented as UNSET sentinel
- Null (set to null): represented as None
- Value (set to value): represented as the actual value

Based on Rust fields with #[serde(deserialize_with = "deserialize_double_option")]
marked with x-double-option: true in the OpenAPI schema.
"""

import json
import re
import sys
from pathlib import Path


# Sentinel value definition to be added at the top
SENTINEL_DEFINITION = '''
class _UnsetType:
    """Sentinel value to distinguish between omitted fields and null values."""
    def __repr__(self):
        return "UNSET"

UNSET = _UnsetType()
"""
Sentinel value to distinguish between omitted and null in API requests.

Usage:
- UNSET: Field is omitted (don't change existing value)
- None: Field is explicitly set to null
- value: Field is set to the provided value
"""
'''


def extract_double_option_fields(openapi_path: Path) -> set[tuple[str, str]]:
    """Extract (class_name, field_name) tuples from OpenAPI schema with x-double-option: true."""
    spec = json.loads(openapi_path.read_text())
    double_option_fields = set()

    schemas = spec.get('components', {}).get('schemas', {})

    for class_name, schema in schemas.items():
        properties = schema.get('properties', {})
        for field_name, field_schema in properties.items():
            if field_schema.get('x-double-option') is True:
                double_option_fields.add((class_name, field_name))

    return double_option_fields


def post_process_file(file_path: Path, double_option_fields: set[tuple[str, str]]) -> None:
    """Post-process the generated types file."""
    content = file_path.read_text()

    # Add UNSET sentinel after imports
    imports_end = content.find('\n\n@dataclass')
    if imports_end == -1:
        imports_end = content.find('\n\nclass ')

    if imports_end != -1:
        content = content[:imports_end] + '\n' + SENTINEL_DEFINITION + content[imports_end:]

    # Process each double-option field
    current_class = None
    lines = content.split('\n')
    result_lines = []
    modified_count = 0

    for i, line in enumerate(lines):
        # Track which class we're in
        class_match = re.match(r'^class (\w+)', line)
        if class_match:
            current_class = class_match.group(1)

        # Check if this is a double-option field
        if current_class:
            field_match = re.match(r'    (\w+): (.+?) = None$', line)
            if field_match:
                field_name = field_match.group(1)
                field_type = field_match.group(2)

                if (current_class, field_name) in double_option_fields:
                    # Replace the type to include UNSET
                    # Change: field_name: T | None = None
                    # To: field_name: T | None | _UnsetType = UNSET
                    new_type = field_type.replace(' | None', ' | None | _UnsetType')
                    new_line = f'    {field_name}: {new_type} = UNSET'
                    result_lines.append(new_line)
                    modified_count += 1
                    print(f"✓ Modified {current_class}.{field_name} to use UNSET")
                    continue

        result_lines.append(line)

    # Write back
    file_path.write_text('\n'.join(result_lines))
    print(f"✓ Added UNSET sentinel and updated {modified_count} fields")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <openapi_schema.json> <types_file.py>")
        sys.exit(1)

    openapi_path = Path(sys.argv[1])
    types_path = Path(sys.argv[2])

    if not openapi_path.exists():
        print(f"Error: OpenAPI schema not found: {openapi_path}")
        sys.exit(1)

    if not types_path.exists():
        print(f"Error: Types file not found: {types_path}")
        sys.exit(1)

    # Extract fields marked with x-double-option from OpenAPI schema
    print(f"Reading OpenAPI schema from {openapi_path}...")
    double_option_fields = extract_double_option_fields(openapi_path)
    print(f"Found {len(double_option_fields)} fields marked with x-double-option")

    # Post-process the generated types file
    print(f"Post-processing {types_path}...")
    post_process_file(types_path, double_option_fields)
