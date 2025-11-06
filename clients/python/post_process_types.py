#!/usr/bin/env python3
"""
Post-process generated types to handle Option<Option<T>> from Rust.

For fields that can distinguish between:
- Omitted (don't change): represented as UNSET sentinel
- Null (set to null): represented as None
- Value (set to value): represented as the actual value

Based on Rust fields with #[serde(deserialize_with = "deserialize_double_option")]
"""

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


# Map of (class_name, field_name) tuples that should use UNSET
# These correspond to Rust's Option<Option<T>> fields
DOUBLE_OPTION_FIELDS = {
    ('UpdateChatDatapointRequest', 'tool_params'),
    ('UpdateJsonDatapointRequest', 'output'),
    ('DatapointMetadataUpdate', 'name'),
}


def post_process_file(file_path: Path) -> None:
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

                if (current_class, field_name) in DOUBLE_OPTION_FIELDS:
                    # Replace the type to include UNSET
                    # Change: field_name: T | None = None
                    # To: field_name: T | None | _UnsetType = UNSET
                    new_type = field_type.replace(' | None', ' | None | _UnsetType')
                    new_line = f'    {field_name}: {new_type} = UNSET'
                    result_lines.append(new_line)
                    continue

        result_lines.append(line)

    # Write back
    file_path.write_text('\n'.join(result_lines))
    print(f"✓ Added UNSET sentinel and updated {len(DOUBLE_OPTION_FIELDS)} fields")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <types_file.py>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    post_process_file(file_path)
