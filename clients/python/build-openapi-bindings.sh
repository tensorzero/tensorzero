#!/bin/bash

# Build Python bindings from OpenAPI schema for TensorZero Python client
set -e

echo "Building Python bindings from OpenAPI schema..."

# Get the script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Step 1: Generate OpenAPI schema from Rust code
echo "Generating OpenAPI schema from Rust..."
cd "$PROJECT_ROOT"
cargo test --lib -p tensorzero-core export_openapi_schema -- --nocapture

# Step 2: Generate Python dataclasses from OpenAPI schema
echo "Generating Python dataclasses from OpenAPI schema..."
cd "$SCRIPT_DIR"

# Run datamodel-code-generator with Python dataclasses
uvx --from="datamodel-code-generator[http]" datamodel-codegen \
  --input "$PROJECT_ROOT/tensorzero-core/openapi/datasets_v1.json" \
  --input-file-type openapi \
  --output tensorzero/types_generated.py \
  --output-model-type dataclasses.dataclass \
  --use-standard-collections \
  --use-union-operator \
  --target-python-version 3.9 \
  --use-schema-description \
  --enum-field-as-literal one \
  --use-title-as-name \
  --use-annotated \
  --collapse-root-models \
  --use-one-literal-as-default

# Step 3: Format the generated file
echo "Formatting generated Python file..."
uvx black tensorzero/types_generated.py

echo "Build OpenAPI bindings completed successfully!"
echo "Generated types are in: tensorzero/types_generated.py"
