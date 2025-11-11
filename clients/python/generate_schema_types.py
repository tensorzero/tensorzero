#!/usr/bin/env python3
"""
Generate Python dataclasses from JSON schemas derived from Rust types.


For pure value types (mostly used in APIs), we use `schemars` to generate the JSON schemas from Rust types,
and `datamodel-code-generator` to generate the Python dataclasses from the JSON schemas.

Rust Types (with annotations)
    ↓ (cargo test export_schema)
JSON Schemas (in clients/schemas/)
    ↓ (python generate_schema_types.py)
Python Dataclasses (in clients/python/tensorzero/generated_types)


This script runs both steps of the generation pipeline:

1. Runs cargo tests to generate JSON schemas from Rust types (using #[export_schema])
    - See `tensorzero-derive` crate for more details on the macro, and https://graham.cool/schemars/ for schemars docs.
2. Generates Python dataclasses from those JSON schemas
    - For detailed configuration options, see https://koxudaxi.github.io/datamodel-code-generator/.

It also sets up customizations to make them work well together:

1.  `schemars` generates one JSON Schema file for each Rust type, bringing in all its dependencies. Naively
    generating Python dataclasses from these JSON Schemas would cause dependencies to be repeatedly generated, so
    we merge them into a single schema file before generating Python dataclasses.
2.  When merging the schema files, some references are recursive
    (e.g. `InferenceFilter`) that point to the root schema of these
    types, so we need to rewrite the `$ref`s to point to the schema definition itself instead of the root schema (which becomes
    `Any` after python dataclass generation).
3.  To generate the __init__.py file so clients can simply write `from tensorzero.generated_types import X`, we
    collect all the type names from the schema and export them.
4.  We generate a custom header file for the generated types to include the `UNSET` sentinel value.



To run this script: run `pnpm generate-python-schemas` from the root of the repository.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def find_schema_files(schema_dir: Path) -> List[Path]:
    """Find all JSON schema files in the schema directory."""
    if not schema_dir.exists():
        print(f"Schema directory not found: {schema_dir}", file=sys.stderr)
        return []

    schema_files = sorted(schema_dir.glob("*.json"))
    if not schema_files:
        print(f"No JSON schema files found in {schema_dir}", file=sys.stderr)

    return schema_files


def rewrite_refs_recursive(obj: Any, schema_name: str) -> Any:
    """
    Recursively rewrite $ref pointers in a schema object.

    Rewrites "$ref": "#" to "$ref": "#/$defs/{schema_name}" for recursive references.

    Args:
        obj: The schema object to process (dict, list, or primitive)
        schema_name: The name of the schema (used for rewriting root references)

    Returns:
        The schema object with rewritten references
    """
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():  # pyright: ignore[reportUnknownVariableType]
            if key == "$ref" and value == "#":
                # Rewrite root reference to point to the schema definition
                result[key] = f"#/$defs/{schema_name}"
            else:
                # Recursively process nested objects
                result[key] = rewrite_refs_recursive(value, schema_name)
        return result  # pyright: ignore[reportUnknownVariableType]
    elif isinstance(obj, list):
        return [rewrite_refs_recursive(item, schema_name) for item in obj]  # pyright: ignore[reportUnknownVariableType]
    else:
        return obj


def preprocess_and_merge_schemas(schema_files: List[Path], output_file: Path) -> None:
    """
    Preprocess and merge all JSON schemas into a single schema file.

    Args:
        schema_files: List of paths to JSON schema files
        output_file: Path to write the merged schema

    Returns:
        None
    """
    print(f"Merging {len(schema_files)} schemas...")

    # Collect all schemas and their definitions
    all_defs: Dict[str, Any] = {}
    root_schemas: Dict[str, Any] = {}

    for schema_file in schema_files:
        with open(schema_file, "r") as f:
            schema = json.load(f)

        schema_name = schema_file.stem

        # Collect $defs (with refs rewritten)
        if "$defs" in schema:
            for def_name, def_schema in schema["$defs"].items():
                # Only add if not already present (first one wins)
                if def_name not in all_defs:
                    # Rewrite any "#" refs in the def to point to the root schema
                    all_defs[def_name] = rewrite_refs_recursive(def_schema, schema_name)

        # Store the root schema (without $defs) with the file name as key
        # and rewrite any "#" refs to point to the schema definition
        root_schema: Dict[str, Any] = {k: v for k, v in schema.items() if k != "$defs"}
        root_schemas[schema_name] = rewrite_refs_recursive(root_schema, schema_name)

    # Create merged schema with all definitions in $defs
    # and all root schemas in a oneOf with discriminator
    merged: Dict[str, Any] = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$defs": {
            # Add all collected definitions
            **all_defs,
            # Add each root schema as a definition
            **{name: schema for name, schema in root_schemas.items()},
        },
    }

    # Write merged schema
    with open(output_file, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"✓ Merged schema written to {output_file}")


def extract_names_from_schema_recursive(schema: Any, exported_names: set[str]) -> None:  # pyright: ignore[reportUnknownParameterType]
    """
    Recursively extract all type names from a schema definition.

    This processes:
    - title fields (become class names)
    - oneOf variants (discriminated unions with inline schemas)
    - allOf compositions (inheritance)
    - nested schemas

    Note: We skip titles in oneOf/anyOf variants that are just $ref pointers,
    since those don't create new types - they just reference existing ones.

    Args:
        schema: A JSON schema definition dictionary
                Takes Any to handle non-dict inputs gracefully
        exported_names: Set to accumulate exported type names into
    """
    if not isinstance(schema, dict):
        return

    # Extract title if present, but only if this isn't just a $ref wrapper
    # (titles on $ref-only schemas are just documentation)
    if "title" in schema and "$ref" not in schema:
        exported_names.add(schema["title"])  # pyright: ignore[reportUnknownArgumentType]

    # Recursively process oneOf (discriminated unions)
    # Only extract names from inline schemas, not from $ref pointers
    # Note: pyright can't infer types through dict iteration when input is Any
    if "oneOf" in schema:
        for variant in schema["oneOf"]:  # pyright: ignore[reportUnknownVariableType]
            # Skip variants that are just $ref pointers - they don't create new types
            if isinstance(variant, dict) and "$ref" not in variant:
                extract_names_from_schema_recursive(variant, exported_names)

    # Recursively process allOf (composition/inheritance)
    # For allOf, we do want to process $ref items to find inline properties
    if "allOf" in schema:
        for item in schema["allOf"]:  # pyright: ignore[reportUnknownVariableType]
            # Only recurse into inline schemas, not $ref pointers
            if isinstance(item, dict) and "$ref" not in item:
                extract_names_from_schema_recursive(item, exported_names)

    # Recursively process anyOf
    if "anyOf" in schema:
        for item in schema["anyOf"]:  # pyright: ignore[reportUnknownVariableType]
            # Skip variants that are just $ref pointers
            if isinstance(item, dict) and "$ref" not in item:
                extract_names_from_schema_recursive(item, exported_names)

    # Recursively process properties
    if "properties" in schema:
        for prop_schema in schema["properties"].values():  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            extract_names_from_schema_recursive(prop_schema, exported_names)

    # Recursively process items (for arrays)
    if "items" in schema:
        extract_names_from_schema_recursive(schema["items"], exported_names)


def extract_exported_names_from_schema(merged_schema: Dict[str, Any]) -> List[str]:
    """
    Extract all type names from the merged JSON schema.

    Args:
        merged_schema: A merged JSON schema dictionary with $defs

    Returns:
        A sorted list of names that should be exported
    """
    exported_names: set[str] = set()

    # Extract all definitions from $defs
    if "$defs" not in merged_schema:
        return []

    for def_name, def_schema in merged_schema["$defs"].items():
        # Always add the definition name itself (this becomes a class or type alias)
        exported_names.add(def_name)

        # Recursively extract names from this definition
        extract_names_from_schema_recursive(def_schema, exported_names)

    # Sort the names for consistent output
    return sorted(exported_names)


def generate_init_exports(exported_names: List[str], output_init_file: Path) -> None:
    """
    Generate a Python snippet for __init__.py that exports all generated types.

    The snippet includes both the import statement and the __all__ additions.
    """
    # Always include UNSET and UnsetType since they're always in generated_types.py
    # but not in the schema definitions
    special_exports = ["UNSET", "UnsetType"]
    all_exports = sorted(set(special_exports + exported_names))

    # Generate the __all__ list entries
    all_list_entries = "\n".join(f'    "{name}",' for name in all_exports)

    init_template = f"""# Auto-generated exports from generated_types.py
# To regenerate, run: `pnpm generate-python-schemas`

from .generated_types import *

__all__ = [
{all_list_entries}
]
"""

    # Write to a separate file that can be copied into __init__.py
    with open(output_init_file, "w") as f:
        f.write(init_template)

    print(f"✓ Generated exports written to {output_init_file}")
    print(f"  Found {len(all_exports)} exported types")
    print()
    print("To use these exports, copy the imports and __all__ entries from generated_exports.py")
    print("into your __init__.py file.")


def generate_dataclasses_from_schema(schema_file: Path, templates_dir: Path, output_file: Path) -> None:
    """
    Generate Python dataclasses from a JSON schema file using datamodel-code-generator.

    Args:
        schema_file: Path to the JSON schema file
        templates_dir: Path to the templates directory
        output_file: Path to write the generated Python code
    """
    print(f"Generating dataclasses from {schema_file.name}...")

    try:
        # Use datamodel-code-generator to generate dataclasses
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "datamodel_code_generator",
                "--input",
                str(schema_file),
                "--input-file-type",
                "jsonschema",
                "--output",
                str(output_file),
                "--output-model-type",
                "dataclasses.dataclass",
                "--target-python-version",
                "3.10",
                # Use list, dict instead of List, Dict
                "--use-standard-collections",
                # Use keyword-only arguments for dataclasses; otherwise we are sensitive to
                # field ordering.
                "--keyword-only",
                # For enums / unions, customize the title with schemars to control their names
                "--use-title-as-name",
                # Don't add generation timestamp
                "--disable-timestamp",
                # Generate Literal["a", "b", "c"] for unit enum values
                "--enum-field-as-literal",
                "all",
                # Do not generate __future__ imports; they are useless
                # and conflict with custom file headers
                "--disable-future-imports",
                # Generate union types as `A | B` instead of `Union[A, B]`
                "--use-union-operator",
                # Use field descriptions for docstrings
                "--use-field-description",
                # Explicitly pass extra keys to handle double-optional generation
                "--field-extra-keys",
                "x-double-option",
                # Use custom dataclass template for double-optionals
                "--custom-template-dir",
                str(templates_dir),
                # Use a custom file header to include the Unset sentinel value
                "--custom-file-header-path",
                str(templates_dir / "generated_types_header.py"),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

    except subprocess.CalledProcessError as e:
        print(f"Error generating dataclasses from {schema_file.name}:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: datamodel-code-generator not found.", file=sys.stderr)
        print("Install it with: pip install 'datamodel-code-generator[http]'", file=sys.stderr)
        sys.exit(1)


def generate_rust_schemas(repo_root: Path, schema_dir: Path) -> None:
    """
    Run cargo tests to generate JSON schemas from Rust types.

    Args:
        repo_root: Path to the repository root
        schema_dir: Path where schemas should be generated
    """
    print("=" * 70)
    print("Step 1: Generating JSON schemas from Rust types")
    print("=" * 70)
    print()

    try:
        result = subprocess.run(
            ["cargo", "test", "-p", "tensorzero-core", "export_schema"],
            cwd=repo_root,
            env={**os.environ, "SCHEMA_RS_EXPORT_DIR": str(schema_dir)},
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout:
            # Only print relevant output, not the full cargo output
            for line in result.stdout.splitlines():
                if "test result:" in line or "running" in line:
                    print(line)
    except subprocess.CalledProcessError as e:
        print(f"Error running cargo test: {e}", file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        sys.exit(1)

    print()


def main() -> None:
    """Main entry point for the script."""
    # Determine paths
    script_dir = Path(__file__).parent
    repo_root = (script_dir / "../..").resolve()
    schema_dir = repo_root / "clients" / "schemas"
    output_dir = script_dir / "tensorzero" / "generated_types"
    output_file = output_dir / "generated_types.py"
    output_init_file = output_dir / "__init__.py"
    temp_dir = script_dir / ".temp_schemas"
    templates_dir = script_dir / "templates"
    custom_header_file = templates_dir / "generated_types_header.py"

    print("=" * 70)
    print("TensorZero Python Schema Generation")
    print("=" * 70)
    print()

    # Step 1: Generate Rust schemas
    generate_rust_schemas(repo_root, schema_dir)

    # Step 2: Generate Python dataclasses
    print("=" * 70)
    print("Step 2: Generating Python dataclasses from JSON schemas")
    print("=" * 70)
    print(f"Schema directory: {schema_dir}")
    print(f"Output file: {output_file}")
    print()

    # Find all schema files
    schema_files = find_schema_files(schema_dir)
    print(f"Found {len(schema_files)} schema files")
    print()

    # If there are no schema files, just output the header file
    if len(schema_files) == 0:
        print("No schema files found, generating header file only...")
        # Copy custom_header_file to output_init_file
        shutil.copy(custom_header_file, output_init_file)
        return

    # Create temp directory for individual generated files
    temp_dir.mkdir(exist_ok=True)

    try:
        # Merge all schemas into one and track double-optional fields
        merged_schema_file = temp_dir / "merged_schema.json"
        preprocess_and_merge_schemas(schema_files, merged_schema_file)

        # Generate dataclasses from merged schema
        generate_dataclasses_from_schema(merged_schema_file, templates_dir, output_file)

        # Generate exports for __init__.py (before deleting temp files)
        print()
        print("Generating exports for __init__.py...")
        with open(merged_schema_file, "r") as f:
            merged_schema = json.load(f)
        exported_names = extract_exported_names_from_schema(merged_schema)
        generate_init_exports(exported_names, output_init_file)

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

    # Delete the schemas directory after successful generation
    print()
    print("Cleaning up schemas directory...")
    try:
        if schema_dir.exists():
            shutil.rmtree(schema_dir)
            print(f"✓ Deleted {schema_dir}")
    except Exception as e:
        print(f"⚠ Warning: Could not delete schemas directory: {e}")

    print()
    print("=" * 70)
    print("✓ Generation complete!")
    print("=" * 70)
    print()
    print("Generated types can be imported with:")
    print("    from tensorzero.generated_types import Input, DynamicToolParams, ...")
    print()
    print("See tensorzero/generated_types/__init__.py for the complete list of exports.")


if __name__ == "__main__":
    main()
