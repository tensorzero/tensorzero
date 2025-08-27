#!/bin/bash
set -euo pipefail

# Clear disk space
./ci/free-disk-space.sh

# Get all env vars
source ci/buildkite/utils/live-tests-env.sh

# Install `gdb`
sudo apt-get update && sudo apt-get install -y gdb

# ------------------------------------------------------------------------------
# Set the short commit hash
# ------------------------------------------------------------------------------
SHORT_HASH=${BUILDKITE_COMMIT:0:7}
export TENSORZERO_GATEWAY_TAG=ci-sha-$SHORT_HASH
export TENSORZERO_MOCK_INFERENCE_PROVIDER_TAG=ci-sha-$SHORT_HASH
export TENSORZERO_PROVIDER_PROXY_CACHE_TAG=ci-sha-$SHORT_HASH


# Get the fixtures
buildkite-agent artifact download fixtures.tar.gz ui/fixtures
tar -xzvf ui/fixtures/fixtures.tar.gz


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
