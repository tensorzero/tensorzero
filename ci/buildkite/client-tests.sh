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
# Run client tests container via Docker Compose and capture exit code
# ------------------------------------------------------------------------------
set +e

# Start both test containers in parallel
docker compose -f clients/docker-compose.tests.yml run --rm \
  -e TENSORZERO_CI=1 \
  python-client-tests &
PYTHON_PID=$!

docker compose -f clients/docker-compose.tests.yml run --rm \
  -e TENSORZERO_CI=1 \
  openai-node-client-tests &
NODE_PID=$!

# Wait for both and capture exit codes
wait $PYTHON_PID
PYTHON_EXIT_CODE=$?
wait $NODE_PID
NODE_EXIT_CODE=$?

set -e
# ------------------------------------------------------------------------------
# Report results
# ------------------------------------------------------------------------------
echo "=============================="
echo "Test Results:"
echo "Python client tests: $([ $PYTHON_EXIT_CODE -eq 0 ] && echo "PASSED" || echo "FAILED") (exit code: $PYTHON_EXIT_CODE)"
echo "Node client tests: $([ $NODE_EXIT_CODE -eq 0 ] && echo "PASSED" || echo "FAILED") (exit code: $NODE_EXIT_CODE)"
echo "=============================="

# Exit with failure if any test failed
if [ $PYTHON_EXIT_CODE -ne 0 ] || [ $NODE_EXIT_CODE -ne 0 ]; then
    echo "One or more test suites failed"
    exit 1
else
    echo "All test suites passed"
    exit 0
fi
