#!/bin/bash
set -euxo pipefail

# Set the short commit hash
SHORT_HASH=${BUILDKITE_COMMIT:0:7}
# Set DNS to not call howdy
echo "127.0.0.1 howdy.tensorzero.com" | sudo tee -a /etc/hosts

# Uncomment if needed: Cleanup disk space
# ./ci/free-disk-space.sh

# Setup Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Verify installation
rustc --version

# Install Node
# Download and install nvm:
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
# in lieu of restarting the shell
\. "$HOME/.nvm/nvm.sh"
# Download and install Node.js:
nvm install 22
# Verify the Node.js version:
node -v # Should print "v22.18.0".
nvm current # Should print "v22.18.0".
# Verify npm version:
npm -v # Should print "10.9.3".

# Install pnpm
curl -fsSL https://get.pnpm.io/install.sh | sh -

# Use pnpm to install Node deps
pnpm install --frozen-lockfile

# Build `tensorzero-node`
pnpm build

# Download the gateway container from BuildKite artifacts
buildkite-agent artifact download gateway-container.tar gateway-container.tar

# Load it into Docker
docker load < gateway-container.tar

# Set up fake credentials needed for the gateway to start up
echo "FIREWORKS_ACCOUNT_ID=not_used" >> fixtures/.env
echo "TENSORZERO_CLICKHOUSE_URL=http://chuser:chpassword@localhost:8123/tensorzero_ui_fixtures" >> fixtures/.env
echo "TENSORZERO_GATEWAY_TAG=sha-$SHORT_HASH" >> fixtures/.env
echo "TENSORZERO_UI_TAG=sha-$SHORT_HASH" >> fixtures/.env

# Environment variables only used by the gateway container
# We deliberately leave these unset when starting the UI container, to ensure
# that it doesn't depend on them being set
echo "FIREWORKS_API_KEY=not_used" >> fixtures/.env-gateway
echo "OPENAI_API_KEY=not_used" >> fixtures/.env-gateway

TENSORZERO_GATEWAY_TAG=sha-$SHORT_HASH docker compose -f fixtures/docker-compose.yml up -d
docker compose -f fixtures/docker-compose.yml wait fixtures

# Run the ui unit tests
pnpm ui:test
