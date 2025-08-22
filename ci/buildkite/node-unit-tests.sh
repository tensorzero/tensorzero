#!/bin/bash
set -euxo pipefail

# ------------------------------------------------------------------------------
# Configurable versions (override via env if needed)
# ------------------------------------------------------------------------------
: "${NODE_VERSION:=22}"      # Use "22" for latest 22.x, or pin e.g. "22.9.0" for determinism
: "${PNPM_VERSION:=9}"       # Use "9" (latest 9.x) or pin "9.12.3", etc.

# ------------------------------------------------------------------------------
# Set the short commit hash
# ------------------------------------------------------------------------------
SHORT_HASH=${BUILDKITE_COMMIT:0:7}

# ------------------------------------------------------------------------------
# Networking hardening used by your tests
# ------------------------------------------------------------------------------
echo "127.0.0.1 howdy.tensorzero.com" | sudo tee -a /etc/hosts

# ------------------------------------------------------------------------------
# Setup Rust (unchanged)
# ------------------------------------------------------------------------------
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustc --version

# ------------------------------------------------------------------------------
# Install Node & pnpm via Volta (cross-platform, CI-friendly)
# ------------------------------------------------------------------------------
export VOLTA_HOME="${HOME}/.volta"
export PATH="${VOLTA_HOME}/bin:${PATH}"

if ! command -v volta >/dev/null 2>&1; then
  curl -fsSL https://get.volta.sh | bash -s -- --skip-setup
  # ensure just-installed volta is on PATH for this shell
  export PATH="${HOME}/.volta/bin:${PATH}"
fi

# Pin toolchain (respects versions above; you can also pin in package.json)
volta install "node@${NODE_VERSION}"
volta install "pnpm@${PNPM_VERSION}"

# Verify toolchain
node -v
npm -v
pnpm -v

# ------------------------------------------------------------------------------
# pnpm cache optimization
# ------------------------------------------------------------------------------
# Use a stable, user-scoped pnpm store path (can be mounted/cached by your CI)
pnpm config set store-dir "${HOME}/.pnpm-store"
# Pre-resolve dependencies from lockfile (no node_modules yet; speeds up CI)
pnpm fetch
# Install exactly from lockfile
pnpm install --frozen-lockfile

# ------------------------------------------------------------------------------
# Build your workspace package(s)
# ------------------------------------------------------------------------------
pnpm -r build  # builds `tensorzero-node` if defined in the workspace

# ------------------------------------------------------------------------------
# Docker: download & load gateway container (unchanged)
# ------------------------------------------------------------------------------
buildkite-agent artifact download gateway-container.tar .
docker load < gateway-container.tar


# ------------------------------------------------------------------------------
# Fixture env for containers
# ------------------------------------------------------------------------------
{
  echo "FIREWORKS_ACCOUNT_ID=not_used"
  echo "TENSORZERO_CLICKHOUSE_URL=http://chuser:chpassword@localhost:8123/tensorzero_ui_fixtures"
  echo "TENSORZERO_GATEWAY_TAG=sha-$SHORT_HASH"
  echo "TENSORZERO_UI_TAG=sha-$SHORT_HASH"
} >> ui/fixtures/.env

{
  echo "FIREWORKS_API_KEY=not_used"
  echo "OPENAI_API_KEY=not_used"
} >> ui/fixtures/.env-gateway

# ------------------------------------------------------------------------------
# Start fixtures and wait
# ------------------------------------------------------------------------------
TENSORZERO_GATEWAY_TAG="sha-$SHORT_HASH" docker compose -f ui/fixtures/docker-compose.yml up -d
docker compose -f ui/fixtures/docker-compose.yml wait fixtures

# ------------------------------------------------------------------------------
# Test setup & execution
# ------------------------------------------------------------------------------
export OPENAI_API_KEY="notused"
export FIREWORKS_API_KEY="notused"
export TENSORZERO_CLICKHOUSE_URL="http://chuser:chpassword@localhost:8123/tensorzero_ui_fixtures"
export TENSORZERO_GATEWAY_URL="http://localhost:3000"
export TENSORZERO_UI_CONFIG_PATH="ui/fixtures/config/tensorzero.toml"
# Set up Buildkite test analytics collection
export BUILDKITE_ANALYTICS_TOKEN=$(buildkite-agent secret get NODE_UNIT_ANALYTICS_ACCESS_TOKEN)

pnpm add -D -w buildkite-test-collector
pnpm ui:test
