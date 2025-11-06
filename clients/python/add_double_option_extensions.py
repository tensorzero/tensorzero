#!/usr/bin/env python3
"""
Add x-double-option extension to OpenAPI schema for Option<Option<T>> fields.

This script post-processes the generated OpenAPI schema to mark fields that
correspond to Rust's Option<Option<T>> pattern with a custom x-double-option
extension. This allows datamodel-code-generator to generate appropriate UNSET
sentinel values for these fields.
"""

import json
import sys
from pathlib import Path


# Map of (class_name, field_name) tuples that correspond to Rust's Option<Option<T>>
# These fields use #[serde(deserialize_with = "deserialize_double_option")]
DOUBLE_OPTION_FIELDS = {
    ('UpdateChatDatapointRequest', 'tool_params'),
    ('UpdateJsonDatapointRequest', 'output'),
    ('DatapointMetadataUpdate', 'name'),
}


def add_double_option_extensions(openapi_path: Path) -> None:
    """Add x-double-option: true to fields that use deserialize_double_option."""
    spec = json.loads(openapi_path.read_text())

    schemas = spec.get('components', {}).get('schemas', {})
    modified_count = 0

    for class_name, field_name in DOUBLE_OPTION_FIELDS:
        if class_name in schemas:
            properties = schemas[class_name].get('properties', {})
            if field_name in properties:
                properties[field_name]['x-double-option'] = True
                print(f"✓ Added x-double-option to {class_name}.{field_name}")
                modified_count += 1
            else:
                print(f"⚠ Warning: Field {class_name}.{field_name} not found in schema")
        else:
            print(f"⚠ Warning: Schema {class_name} not found")

    # Write back the modified schema
    openapi_path.write_text(json.dumps(spec, indent=2) + '\n')
    print(f"✓ Modified {modified_count} fields with x-double-option extension")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <openapi_schema.json>")
        sys.exit(1)

    openapi_path = Path(sys.argv[1])
    if not openapi_path.exists():
        print(f"Error: File not found: {openapi_path}")
        sys.exit(1)

    add_double_option_extensions(openapi_path)
