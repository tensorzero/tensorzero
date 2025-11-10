#!/bin/bash
set -euxo pipefail

# ==============================================================================
# Node.js & pnpm Setup Utility
# ==============================================================================
# This script sets up Node.js, pnpm, and related tooling for CI environments
# Can be sourced from other scripts to avoid duplication
#
# Usage:
#   source utils/setup-node.sh
#   Or call directly: ./utils/setup-node.sh
# ==============================================================================

# ------------------------------------------------------------------------------
# Configurable versions (override via env if needed)
# ------------------------------------------------------------------------------
: "${NODE_VERSION:=24.11.0}"      # Use "24.11.0" for latest stable Node 24.x
: "${PNPM_VERSION:=9}"       # Use "9" (latest 9.x) or pin "9.12.3", etc.

echo "==============================================================================="
echo "Setting up Node.js ${NODE_VERSION} and pnpm ${PNPM_VERSION}"
echo "==============================================================================="

# ------------------------------------------------------------------------------
# Install Node & pnpm via Volta (cross-platform, CI-friendly)
# ------------------------------------------------------------------------------
export VOLTA_HOME="${HOME}/.volta"
export PATH="${VOLTA_HOME}/bin:${PATH}"

if ! command -v volta >/dev/null 2>&1; then
  echo "Installing Volta..."
  curl -fsSL https://get.volta.sh | bash -s -- --skip-setup
  # ensure just-installed volta is on PATH for this shell
  export PATH="${HOME}/.volta/bin:${PATH}"
fi

# Pin toolchain (respects versions above; you can also pin in package.json)
echo "Installing Node.js and pnpm..."
volta install "node@${NODE_VERSION}"
volta install "pnpm@${PNPM_VERSION}"

# Verify toolchain
echo "Verifying Node.js toolchain..."
node -v
npm -v
pnpm -v

# ------------------------------------------------------------------------------
# pnpm cache optimization
# ------------------------------------------------------------------------------
echo "Optimizing pnpm cache..."
# Use a stable, user-scoped pnpm store path (can be mounted/cached by your CI)
pnpm config set store-dir "${HOME}/.pnpm-store"
# Pre-resolve dependencies from lockfile (no node_modules yet; speeds up CI)
pnpm fetch
# Install exactly from lockfile
pnpm install --frozen-lockfile

# ------------------------------------------------------------------------------
# Build your workspace package(s)
# ------------------------------------------------------------------------------
echo "Building workspace packages..."
pnpm -r build  # builds `tensorzero-node` if defined in the workspace

echo "Node.js setup completed successfully!"
