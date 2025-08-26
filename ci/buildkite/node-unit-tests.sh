#!/bin/bash
set -euo pipefail

# Set up Buildkite test analytics collection
export BUILDKITE_ANALYTICS_TOKEN=$(buildkite-agent secret get NODE_UNIT_ANALYTICS_ACCESS_TOKEN)
if [ -z "$BUILDKITE_ANALYTICS_TOKEN" ]; then
    echo "Error: BUILDKITE_ANALYTICS_TOKEN is not set"
    exit 1
fi

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
# Cleanup function (always runs on exit)
# ------------------------------------------------------------------------------
cleanup() {
  echo "==============================================================================="
  echo "Running cleanup and debug steps..."
  echo "==============================================================================="

  echo "Printing Docker Compose logs..."
  docker compose -f ui/fixtures/docker-compose.yml logs -t || true

  echo "Checking for commit hash in gateway logs..."
  if docker compose -f ui/fixtures/docker-compose.yml logs gateway | grep "(commit: ${SHORT_HASH})"; then
    echo "SUCCESS: Commit hash ${SHORT_HASH} found in gateway logs"
  else
    echo "ERROR: Commit hash ${SHORT_HASH} not found in gateway logs"
  fi

  echo "Printing ClickHouse error logs..."
  docker exec fixtures-clickhouse-1 cat /var/log/clickhouse-server/clickhouse-server.err.log || echo "Warning: Could not print ClickHouse error logs"

  echo "Printing ClickHouse trace logs..."
  docker exec fixtures-clickhouse-1 cat /var/log/clickhouse-server/clickhouse-server.log || echo "Warning: Could not print ClickHouse trace logs"
}

# Set trap to run cleanup on any exit (success or failure)
trap cleanup EXIT

# ------------------------------------------------------------------------------
# Turn off howdy
# ------------------------------------------------------------------------------
echo "127.0.0.1 howdy.tensorzero.com" | sudo tee -a /etc/hosts

# ------------------------------------------------------------------------------
# Setup Rust
# ------------------------------------------------------------------------------
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustc --version

# ------------------------------------------------------------------------------
# Setup Node.js and pnpm using shared utility
# ------------------------------------------------------------------------------
source "$(dirname "$0")/utils/setup-node.sh"

# ------------------------------------------------------------------------------
# Docker: download & load gateway container
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
export TENSORZERO_UI_CONFIG_PATH="fixtures/config/tensorzero.toml"

pnpm add -D -w buildkite-test-collector
pnpm ui:test
