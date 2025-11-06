#!/bin/bash
# Generate all schemas: Rust JSON schemas and Python dataclasses
#
# This script:
# 1. Runs tests to generate JSON schemas from Rust types (using #[export_schema])
# 2. Generates Python dataclasses from the JSON schemas
#
# Run from the repository root:
#   ./generate_all_schemas.sh

set -e  # Exit on error

echo "========================================================================"
echo "TensorZero Schema Generation (ts-rs style)"
echo "========================================================================"
echo ""

# Step 1: Run tests to generate JSON schemas
echo "Step 1: Running tests to generate JSON schemas from Rust types..."
echo "------------------------------------------------------------------------"
cd tensorzero-core
cargo test export_schema
cd ..
echo ""

# Step 2: Generate Python dataclasses from JSON schemas
echo "Step 2: Generating Python dataclasses from JSON schemas..."
echo "------------------------------------------------------------------------"
cd clients/python
uv run generate_schema_types.py
cd ../..
echo ""

echo "========================================================================"
echo "✓ All schemas generated successfully!"
echo "========================================================================"
echo ""
echo "Generated files:"
echo "  - JSON schemas: schemas/*.json (14 files)"
echo "  - Python types: clients/python/tensorzero/generated_types.py"
echo ""
echo "To add a new type to schema generation:"
echo "  1. Add #[derive(JsonSchema)] to your Rust type"
echo "  2. Add #[cfg_attr(test, tensorzero_schema_generation::export_schema)]"
echo "  3. Run this script again!"
echo ""
