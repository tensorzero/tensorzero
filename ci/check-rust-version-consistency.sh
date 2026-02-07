#!/bin/bash

set -e

# Extract rust-version from Cargo.toml
RUST_VERSION=$(grep '^rust-version = ' Cargo.toml | sed 's/rust-version = "\(.*\)"/\1/')

if [ -z "$RUST_VERSION" ]; then
  echo "❌ Could not extract rust-version from Cargo.toml"
  exit 1
fi

echo "Expected Rust version from Cargo.toml: $RUST_VERSION"

# Check rust-toolchain.toml
TOOLCHAIN_FILE="rust-toolchain.toml"
if [ -f "$TOOLCHAIN_FILE" ]; then
  TOOLCHAIN_VERSION=$(grep 'channel = ' "$TOOLCHAIN_FILE" | sed 's/.*channel = "\(.*\)"/\1/')
  if [ "$TOOLCHAIN_VERSION" != "$RUST_VERSION" ]; then
    echo "❌ channel in $TOOLCHAIN_FILE ($TOOLCHAIN_VERSION) does not match Cargo.toml ($RUST_VERSION)"
    exit 1
  fi
  echo "✅ rust-toolchain.toml channel matches: $TOOLCHAIN_VERSION"
fi

# Extract major.minor for cargo-chef images (they use rust-X.YY format)
RUST_MAJOR_MINOR=$(echo "$RUST_VERSION" | sed 's/\([0-9]*\.[0-9]*\).*/\1/')

ERRORS=0

check_dockerfile() {
  local file="$1"
  local pattern="$2"
  local expected="$3"
  local description="$4"

  if [ ! -f "$file" ]; then
    echo "⚠️  File not found: $file"
    return
  fi

  if ! grep -q "$pattern" "$file"; then
    echo "❌ $file: Could not find pattern '$pattern'"
    ERRORS=$((ERRORS + 1))
    return
  fi

  local actual
  actual=$(grep "$pattern" "$file" | head -1)

  if ! echo "$actual" | grep -q "$expected"; then
    echo "❌ $file: Expected '$expected' but found:"
    echo "   $actual"
    ERRORS=$((ERRORS + 1))
  else
    echo "✅ $file: $description"
  fi
}

echo ""
echo "Checking Dockerfiles for Rust version consistency..."
echo ""

# Gateway Dockerfile uses cargo-chef with rust-X.YY format
check_dockerfile \
  "gateway/Dockerfile" \
  "FROM lukemathwalker/cargo-chef:" \
  "rust-$RUST_MAJOR_MINOR" \
  "cargo-chef uses rust-$RUST_MAJOR_MINOR"

# UI Dockerfile uses rust:X.YY.Z format
check_dockerfile \
  "ui/Dockerfile" \
  "FROM rust:.*AS tensorzero-node-build-env" \
  "rust:$RUST_VERSION" \
  "uses rust:$RUST_VERSION"

# Other Dockerfiles using rust:X.YY.Z
DOCKERFILES=(
  "provider-proxy/Dockerfile"
  "evaluations/Dockerfile"
  "tensorzero-core/tests/e2e/Dockerfile.gateway.e2e"
  "tensorzero-core/tests/e2e/Dockerfile.clickhouse"
  "tensorzero-core/tests/e2e/Dockerfile.live"
  "tensorzero-core/tests/mock-provider-api/Dockerfile"
  "ui/fixtures/Dockerfile.unit"
)

for dockerfile in "${DOCKERFILES[@]}"; do
  if [ -f "$dockerfile" ]; then
    check_dockerfile \
      "$dockerfile" \
      "FROM rust:" \
      "rust:$RUST_VERSION" \
      "uses rust:$RUST_VERSION"
  fi
done

echo ""

if [ $ERRORS -gt 0 ]; then
  echo "❌ Found $ERRORS Rust version inconsistencies"
  echo "Please update all Dockerfiles to use Rust $RUST_VERSION"
  exit 1
fi

echo "✅ All Dockerfiles use consistent Rust version: $RUST_VERSION"
