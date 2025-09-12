#!/bin/bash
set -euo pipefail

# Common CI trap helper
source ci/buildkite/utils/trap-helpers.sh
# Print logs from the ClickHouse compose stack on exit
tz_setup_compose_logs_trap tensorzero-core/tests/e2e/docker-compose.clickhouse.yml

# Set BUILDKITE_ANALYTICS_TOKEN
export BUILDKITE_ANALYTICS_TOKEN=$(buildkite-agent secret get CLICKHOUSE_TESTS_ANALYTICS_ACCESS_TOKEN)

if [ -z "$BUILDKITE_ANALYTICS_TOKEN" ]; then
    echo "Error: BUILDKITE_ANALYTICS_TOKEN is not set"
    exit 1
fi


# ------------------------------------------------------------------------------
# Set the short commit hash
# ------------------------------------------------------------------------------
SHORT_HASH=${BUILDKITE_COMMIT:0:7}

# Get the fixtures
buildkite-agent artifact download fixtures.tar.gz ui/fixtures
test ui/fixtures/fixtures.tar.gz
tar -xzvf ui/fixtures/fixtures.tar.gz

# Write the GCP JWT key to a file
echo $(buildkite-agent secret get GCP_JWT_KEY) > gcp_jwt_key.json


# ------------------------------------------------------------------------------
# Docker Hub auth (for pulling built images)
# ------------------------------------------------------------------------------
source ci/buildkite/utils/docker-hub-credentials.sh
echo "$DOCKER_HUB_ACCESS_TOKEN" | docker login -u "$DOCKER_HUB_USERNAME" --password-stdin
echo "Logged in to Docker Hub"

# ------------------------------------------------------------------------------
# Environment for image tags and test config
# ------------------------------------------------------------------------------
export TENSORZERO_GATEWAY_TAG=ci-sha-$SHORT_HASH
export TENSORZERO_MOCK_INFERENCE_PROVIDER_TAG=ci-sha-$SHORT_HASH
export TENSORZERO_PYTHON_CLIENT_TESTS_TAG=ci-sha-$SHORT_HASH
export TENSORZERO_OPENAI_NODE_CLIENT_TESTS_TAG=ci-sha-$SHORT_HASH
export TENSORZERO_SKIP_LARGE_FIXTURES=1

# ------------------------------------------------------------------------------
# Pull images referenced by the compose file
# ------------------------------------------------------------------------------
docker compose -f clients/docker-compose.tests.yml pull

# ------------------------------------------------------------------------------
# Start shared infrastructure first
# ------------------------------------------------------------------------------
echo "Starting shared infrastructure..."
docker compose -f clients/docker-compose.tests.yml up -d \
  clickhouse mock-inference-provider minio provider-proxy gateway fixtures

# Wait for all dependencies to be healthy
echo "Waiting for infrastructure to be ready..."
docker compose -f clients/docker-compose.tests.yml wait \
  clickhouse mock-inference-provider minio provider-proxy gateway

# ------------------------------------------------------------------------------
# Run tests in parallel against shared infrastructure
# ------------------------------------------------------------------------------
set +e

# Run tests in parallel
docker compose -f clients/docker-compose.tests.yml run --rm \
  -e TENSORZERO_CI=1 \
  python-client-tests &
PYTHON_PID=$!

docker compose -f clients/docker-compose.tests.yml run --rm \
  -e TENSORZERO_CI=1 \
  openai-node-client-tests &
NODE_PID=$!

docker compose -f clients/docker-compose.tests.yml run --rm \
  -e TENSORZERO_CI=1 \
  openai-go-client-tests &
GO_PID=$!

# Wait for all tests
wait $PYTHON_PID
PYTHON_EXIT_CODE=$?
wait $NODE_PID
NODE_EXIT_CODE=$?
wait $GO_PID
GO_EXIT_CODE=$?

set -e


# ------------------------------------------------------------------------------
# Report results
# ------------------------------------------------------------------------------
echo "=============================="
echo "Test Results:"
echo "Python client tests: $([ $PYTHON_EXIT_CODE -eq 0 ] && echo "PASSED" || echo "FAILED") (exit code: $PYTHON_EXIT_CODE)"
echo "Node client tests: $([ $NODE_EXIT_CODE -eq 0 ] && echo "PASSED" || echo "FAILED") (exit code: $NODE_EXIT_CODE)"
echo "Go client tests: $([ $GO_EXIT_CODE -eq 0 ] && echo "PASSED" || echo "FAILED") (exit code: $GO_EXIT_CODE)"
echo "=============================="

# Exit with failure if any test failed
if [ $PYTHON_EXIT_CODE -ne 0 ] || [ $NODE_EXIT_CODE -ne 0 ] || [ $GO_EXIT_CODE -ne 0 ]; then
    echo "One or more test suites failed"
    exit 1
else
    echo "All test suites passed"
    exit 0
fi
