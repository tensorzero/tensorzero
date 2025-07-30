#!/bin/bash

# Exit on any error
set -e

# Extract version from Cargo.toml
CARGO_VERSION=$(grep '^version = ' Cargo.toml | sed 's/version = "\(.*\)"/\1/')

# Extract version from ui/package.json
UI_VERSION=$(grep '"version":' ui/package.json | sed 's/.*"version": "\(.*\)".*/\1/')

echo "Cargo.toml version: $CARGO_VERSION"
echo "ui/package.json version: $UI_VERSION"

# Check if versions match
if [ "$CARGO_VERSION" != "$UI_VERSION" ]; then
  echo "❌ Version mismatch detected!"
  echo "Cargo.toml version: $CARGO_VERSION"
  echo "ui/package.json version: $UI_VERSION"
  echo "Please ensure both files have the same version."
  exit 1
fi

echo "✅ Version consistency check passed: $CARGO_VERSION"

# Verify Cargo.lock is up to date using cargo tree --locked
if ! cargo tree --locked >/dev/null 2>&1; then
  echo "❌ Cargo.lock is not up to date with Cargo.toml"
  echo "Please run 'cargo update' to update Cargo.lock"
  exit 1
fi

echo "✅ Cargo.lock is up to date"
