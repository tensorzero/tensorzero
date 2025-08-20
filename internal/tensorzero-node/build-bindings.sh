#!/bin/bash

# Build TypeScript bindings for TensorZero Node.js client
set -e

echo "Building TypeScript bindings..."

# Remove entire bindings directory and recreate it
echo "Cleaning up bindings directory..."
rm -rf lib/bindings
mkdir -p lib/bindings

# Generate TypeScript bindings from Rust code
echo "Generating TypeScript bindings from Rust..."
cd ../..
TS_RS_EXPORT_DIR="$(pwd)/internal/tensorzero-node/lib/bindings" cargo tsbuild
cd internal/tensorzero-node

# Generate index file
echo "Generating index file..."
node -e 'import("./lib/generate-index.js").then(m => m.generateIndex())'

# Format generated TypeScript files
echo "Formatting generated files..."
pnpm format --write "lib/bindings/**/*.ts"

echo "Build bindings completed successfully!"
