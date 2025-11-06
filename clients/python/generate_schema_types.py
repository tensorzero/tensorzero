#!/usr/bin/env python3
"""
Generate Python dataclasses from JSON Schema files.

This script reads JSON schema files generated from Rust types and creates
Python dataclasses that can be used with the TensorZero Python client.

Run this script from the clients/python directory:
    python generate_schema_types.py
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def fix_schema_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix invalid JSON schemas that mix $ref with other properties.

    Schemars generates schemas for tagged enums that have both $ref and properties,
    which is invalid in JSON Schema. This function resolves the $ref and merges
    the properties.
    """
    def resolve_ref(ref_path: str, schema_root: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a $ref path like #/$defs/TypeName"""
        parts = ref_path.split("/")[1:]  # Skip the # part
        current = schema_root
        for part in parts:
            current = current[part]
        return current.copy()

    def fix_one_of(one_of_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fix oneOf items that have invalid $ref + properties"""
        fixed_items = []
        for item in one_of_items:
            if "$ref" in item and len(item) > 1:
                # Invalid: has $ref plus other properties
                # Resolve the $ref and merge with other properties
                ref_schema = resolve_ref(item["$ref"], schema)
                merged = {**ref_schema}
                # Merge properties from the discriminator
                if "properties" in item:
                    if "properties" not in merged:
                        merged["properties"] = {}
                    merged["properties"].update(item["properties"])
                # Merge required fields
                if "required" in item:
                    if "required" not in merged:
                        merged["required"] = []
                    for req in item["required"]:
                        if req not in merged["required"]:
                            merged["required"].append(req)
                # Keep other top-level properties from the original item
                for key in item:
                    if key not in ["$ref", "properties", "required"]:
                        merged[key] = item[key]
                fixed_items.append(merged)
            else:
                fixed_items.append(item)
        return fixed_items

    # Fix the top-level oneOf if it exists
    if "oneOf" in schema:
        schema["oneOf"] = fix_one_of(schema["oneOf"])

    # Recursively fix any oneOf in definitions
    if "$defs" in schema:
        for def_name, def_schema in schema["$defs"].items():
            if isinstance(def_schema, dict) and "oneOf" in def_schema:
                schema["$defs"][def_name]["oneOf"] = fix_one_of(def_schema["oneOf"])

    return schema


def find_schema_files(schema_dir: Path) -> List[Path]:
    """Find all JSON schema files in the schema directory."""
    if not schema_dir.exists():
        print(f"Error: Schema directory not found: {schema_dir}", file=sys.stderr)
        sys.exit(1)

    schema_files = sorted(schema_dir.glob("*.json"))
    if not schema_files:
        print(f"Warning: No JSON schema files found in {schema_dir}", file=sys.stderr)

    return schema_files


def merge_schemas(schema_files: List[Path], output_file: Path) -> None:
    """
    Merge all JSON schemas into a single schema file with shared $defs.

    This allows datamodel-code-generator to generate deduplicated types.

    Args:
        schema_files: List of paths to JSON schema files
        output_file: Path to write the merged schema

    Returns:
        None
    """
    print(f"Merging {len(schema_files)} schemas...")

    # Collect all schemas and their definitions
    all_defs = {}
    root_schemas = {}

    for schema_file in schema_files:
        with open(schema_file, 'r') as f:
            schema = json.load(f)

        # Fix the schema
        schema = fix_schema_refs(schema)

        # Collect $defs
        if "$defs" in schema:
            for def_name, def_schema in schema["$defs"].items():
                # Only add if not already present (first one wins)
                if def_name not in all_defs:
                    all_defs[def_name] = def_schema

        # Store the root schema (without $defs) with the file name as key
        schema_name = schema_file.stem
        root_schema = {k: v for k, v in schema.items() if k != "$defs"}
        root_schemas[schema_name] = root_schema

    # Create merged schema with all definitions in $defs
    # and all root schemas in a oneOf with discriminator
    merged = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$defs": {
            # Add all collected definitions
            **all_defs,
            # Add each root schema as a definition
            **{name: schema for name, schema in root_schemas.items()}
        }
    }

    # Write merged schema
    with open(output_file, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"✓ Merged schema written to {output_file}")

def generate_dataclasses_from_schema(
    schema_file: Path,
    output_file: Path,
) -> None:
    """
    Generate Python dataclasses from a merged JSON schema file.

    Args:
        schema_file: Path to the merged JSON schema file
        output_file: Path to write the generated Python code
        double_optional_fields: Dict mapping type names to lists of double-optional field names
    """
    print(f"Generating dataclasses...")

    # Determine paths for custom templates
    script_dir = Path(__file__).parent

    try:
        # Use datamodel-code-generator to generate dataclasses
        cmd = [
            sys.executable,
            "-m",
            "datamodel_code_generator",
            "--input", str(schema_file),
            "--input-file-type", "jsonschema",
            "--output", str(output_file),
            "--output-model-type", "dataclasses.dataclass",
            "--use-standard-collections",
            "--use-union-operator",
            "--target-python-version", "3.12",
            "--use-schema-description",
            # "--enum-field-as-literal", "one",
            # "--use-title-as-name",
            "--disable-timestamp",
            "--use-annotated",
            "--field-constraints",
            # "--collapse-root-models",
            "--use-one-literal-as-default",
            "--field-extra-keys", "x-double-optional",
            "--custom-template-dir", str(script_dir / "templates"),
            "--custom-file-header-path", str(script_dir / "file_header.py"),
            "--keyword-only",
            "--use-field-description",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        print(f"✓ Generated {output_file}")

    except subprocess.CalledProcessError as e:
        print(f"Error generating dataclasses:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: datamodel-code-generator not found.", file=sys.stderr)
        print("Install it with: pip install 'datamodel-code-generator[http]'", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main entry point for the script."""
    # Determine paths
    script_dir = Path(__file__).parent
    schema_dir = script_dir / ".." / "schemas"
    schema_dir = schema_dir.resolve()
    output_dir = script_dir / "tensorzero"
    output_file = output_dir / "generated_types.py"
    temp_dir = script_dir / ".temp_schemas"

    print("=" * 70)
    print("Generating Python dataclasses from JSON schemas")
    print("=" * 70)
    print(f"Schema directory: {schema_dir}")
    print(f"Output file: {output_file}")
    print()

    # Find all schema files
    schema_files = find_schema_files(schema_dir)
    print(f"Found {len(schema_files)} schema files")
    print()

    try:
        # Create temp directory
        temp_dir.mkdir(exist_ok=True)

        # Merge all schemas into one and track double-optional fields
        merged_schema_file = temp_dir / "merged_schema.json"
        merge_schemas(schema_files, merged_schema_file)

        # Generate dataclasses from merged schema
        generate_dataclasses_from_schema(merged_schema_file, output_file)

        # Post-process: remove all "from __future__ import annotations" lines
        # (we include it in file_header.py, so datamodel-code-generator's version is redundant)
        print("Post-processing: removing redundant future imports...")
        with open(output_file, 'r') as f:
            lines = f.readlines()

        filtered_lines = [
            line for line in lines
            if line.strip() != "from __future__ import annotations"
        ]

        with open(output_file, 'w') as f:
            f.writelines(filtered_lines)

        print("✓ Removed duplicate future imports")

    finally:
        # Clean up temp files
        if temp_dir.exists():
            for temp_file in temp_dir.glob("*"):
                if temp_file.is_file():
                    temp_file.unlink()
            try:
                temp_dir.rmdir()
            except OSError:
                pass

    print()
    print("=" * 70)
    print("✓ Generation complete!")
    print("=" * 70)
    print()
    print("Generated types can be imported with:")
    print("    from tensorzero.generated_types import Input, DynamicToolParams, ...")


if __name__ == "__main__":
    main()
