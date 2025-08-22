# ------------------------------------------------------------------------------
# Set the short commit hash
# ------------------------------------------------------------------------------
SHORT_HASH=${BUILDKITE_COMMIT:0:7}

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
pnpm --filter=tensorzero-ui exec playwright install --with-deps chromium

# ------------------------------------------------------------------------------
# Docker: download & load gateway container
# ------------------------------------------------------------------------------
buildkite-agent artifact download gateway-container.tar .
docker load < gateway-container.tar

# ------------------------------------------------------------------------------
# Docker: download & load ui container
# ------------------------------------------------------------------------------
buildkite-agent artifact download ui-container.tar .
docker load < ui-container.tar

# ------------------------------------------------------------------------------
# Docker: download & load mock-inference-provider container
# ------------------------------------------------------------------------------
buildkite-agent artifact download mock-inference-provider-container.tar .
docker load < mock-inference-provider-container.tar

echo "Loaded all containers"

# Environment variables shared by the gateway and ui containers
echo "TENSORZERO_CLICKHOUSE_URL=http://chuser:chpassword@clickhouse:8123/tensorzero_ui_fixtures" >> fixtures/.env
echo "TENSORZERO_GATEWAY_URL=http://gateway:3000" >> fixtures/.env
echo "TENSORZERO_GATEWAY_TAG=sha-$SHORT_HASH" >> fixtures/.env
echo "TENSORZERO_UI_TAG=sha-$SHORT_HASH" >> fixtures/.env
echo "TENSORZERO_MOCK_INFERENCE_PROVIDER=sha-$SHORT_HASH" >> fixtures/.env
# We need these set in the ui container, so that we construct the correct optimizer config
# to pass to 'experimentalLaunchOptimizationWorkflow'
echo "FIREWORKS_BASE_URL=http://mock-inference-provider:3030/fireworks/" >> fixtures/.env
echo "OPENAI_BASE_URL=http://mock-inference-provider:3030/openai/" >> fixtures/.env
echo "FIREWORKS_ACCOUNT_ID=fake_fireworks_account" >> fixtures/.env
echo "VITE_TENSORZERO_FORCE_CACHE_ON=1" >> fixtures/.env

# Environment variables only used by the gateway container
# We deliberately leave these unset when starting the UI container, to ensure
# that it doesn't depend on them being set
# We set API credentials to dummy values
# but give real S3 credentials so the bucket can be read
echo "FIREWORKS_ACCOUNT_ID='not_used'" >> fixtures/.env-gateway
echo "FIREWORKS_API_KEY='not_used'" >> fixtures/.env-gateway
echo "OPENAI_API_KEY='not_used'" >> fixtures/.env-gateway
echo "ANTHROPIC_API_KEY='not_used'" >> fixtures/.env-gateway
# TODO
echo "S3_ACCESS_KEY_ID=$(buildkite-agent secret get S3_ACCESS_KEY_ID)" >> fixtures/.env-gateway
echo "S3_SECRET_ACCESS_KEY=$(buildkite-agent secret get S3_SECRET_ACCESS_KEY)" >> fixtures/.env-gateway

# Start the containers for playwright tests
docker compose -f fixtures/docker-compose.e2e.yml up --no-build -d
docker compose -f fixtures/docker-compose.e2e.yml wait fixtures
docker compose -f fixtures/docker-compose.ui.yml up --no-build -d --wait

pnpm ui:test:e2e --grep-invert "@credentials"
