#!/bin/bash
set -euo pipefail

# Clear disk space
./ci/free-disk-space.sh

# Get all env vars
source ci/buildkite/utils/live-tests-env.sh

# Install `gdb`
sudo apt-get update && sudo apt-get install -y gdb

# Warm up modal instances
curl -H "Modal-Key: $MODAL_KEY" -H "Modal-Secret: $MODAL_SECRET" https://tensorzero--vllm-inference-vllm-inference.modal.run/docs > vllm_modal_logs.txt &
curl -H "Modal-Key: $MODAL_KEY" -H "Modal-Secret: $MODAL_SECRET" https://tensorzero--sglang-inference-sglang-inference.modal.run/ > sglang_modal_logs.txt &

# ------------------------------------------------------------------------------
# Set the short commit hash
# ------------------------------------------------------------------------------
SHORT_HASH=${BUILDKITE_COMMIT:0:7}
export TENSORZERO_GATEWAY_TAG=sha-$SHORT_HASH
export TENSORZERO_MOCK_INFERENCE_PROVIDER_TAG=sha-$SHORT_HASH
# ------------------------------------------------------------------------------
# Setup Rust
# ------------------------------------------------------------------------------
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustc --version

# Install `uv`
curl -LsSf https://astral.sh/uv/0.6.17/install.sh | sh
source $HOME/.local/bin/env

# Get the fixtures
buildkite-agent artifact download fixtures.tar.gz ui/fixtures
tar -xzvf ui/fixtures/fixtures.tar.gz

# ------------------------------------------------------------------------------
# Docker: download & load gateway container
# ------------------------------------------------------------------------------
buildkite-agent artifact download gateway-container.tar .
docker load < gateway-container.tar

# ------------------------------------------------------------------------------
# Docker: download & load mock-inference-provider container
# ------------------------------------------------------------------------------
buildkite-agent artifact download mock-inference-provider-container.tar .
docker load < mock-inference-provider-container.tar

echo "Loaded all containers"

# ------------------------------------------------------------------------------
# Setup Node.js and pnpm using shared utility
# ------------------------------------------------------------------------------
# source "$(dirname "$0")/utils/setup-node.sh"
# pnpm install --frozen-lockfile

# Install cargo-binstall
curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash

# Install cargo-nextest
# TODO: install the pre-built package instead of building from source
cargo binstall -y cargo-nextest --secure

# Write GCP JWT key to file
buildkite-agent secret get GCP_JWT_KEY > ~/gcp_jwt_key.json

# Set up TENSORZERO_CLICKHOUSE_URL for E2E tests
export TENSORZERO_CLICKHOUSE_URL="http://chuser:chpassword@localhost:8123/tensorzero_e2e_tests"

# TODO: handle batch writes
# echo "TENSORZERO_CLICKHOUSE_BATCH_WRITES=${{ matrix.batch_writes }}" >> $GITHUB_ENV
# - name: Configure batch writes in tensorzero.toml
#   if: matrix.batch_writes == true
#   run: |
#     echo "[gateway.observability.batch_writes]" >> tensorzero-core/tests/e2e/tensorzero.toml
#     echo "enabled = true" >> tensorzero-core/tests/e2e/tensorzero.toml
#     echo "flush_interval_ms = 80" >> tensorzero-core/tests/e2e/tensorzero.toml
#     echo "__force_allow_embedded_batch_writes = true" >> tensorzero-core/tests/e2e/tensorzero.toml

# TODO: Download provider proxy cache
AWS_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY ./ci/download-provider-proxy-cache.sh

# Launch services for E2E tests
docker compose -f tensorzero-core/tests/e2e/docker-compose.yml up -d --wait

# Launch the provider proxy cache for E2E tests
./ci/run-provider-proxy.sh ci

# Build the gateway for E2E tests
cargo build-e2e

# Launch the gateway for E2E tests
TENSORZERO_E2E_PROXY="http://localhost:3003" cargo run-e2e > e2e_logs.txt 2>&1 &
while ! curl -s -f http://localhost:3000/health >/dev/null 2>&1; do
  echo "Waiting for gateway to be healthy..."
  sleep 1
done
export GATEWAY_PID=$!
# TODO: add all the othe auxiliary tests
TENSORZERO_E2E_PROXY="http://localhost:3003" cargo test-e2e --no-fail-fast

# Upload the test JUnit XML files
curl -X POST \
  -H "Authorization: Token token=$BUILDKITE_ANALYTICS_TOKEN" \
  -F "format=junit" \
  -F "data=@target/nextest/e2e/junit.xml" \
  -F "run_env[CI]=buildkite" \
  -F "run_env[key]=$BUILDKITE_BUILD_ID" \
  -F "run_env[number]=$BUILDKITE_BUILD_NUMBER" \
  -F "run_env[job_id]=$BUILDKITE_JOB_ID" \
  -F "run_env[branch]=$BUILDKITE_BRANCH" \
  -F "run_env[commit_sha]=$BUILDKITE_COMMIT" \
  -F "run_env[message]=$BUILDKITE_MESSAGE" \
  -F "run_env[url]=$BUILDKITE_BUILD_URL" \
  https://analytics-api.buildkite.com/v1/uploads
