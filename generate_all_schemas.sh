#!/bin/bash
# Generate all schemas: Rust JSON schemas and Python dataclasses
#
# This script:
# 1. Generates JSON schemas from Rust types using schemars
# 2. Generates Python dataclasses from the JSON schemas
#
# Run from the repository root:
#   ./generate_all_schemas.sh

set -e  # Exit on error

echo "========================================================================"
echo "TensorZero Schema Generation"
echo "========================================================================"
echo ""

# Step 1: Generate JSON schemas from Rust types
echo "Step 1: Generating JSON schemas from Rust types..."
echo "------------------------------------------------------------------------"
cd tensorzero-core
cargo run --example generate_schemas
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
