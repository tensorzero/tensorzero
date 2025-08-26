#!/bin/bash
set -euo pipefail

# Common CI trap helper
source ci/buildkite/utils/trap-helpers.sh
# Print logs from the ClickHouse compose stack on exit
tz_setup_compose_logs_trap tensorzero-core/tests/e2e/docker-compose.clickhouse.yml

# Ensure required ClickHouse version variable is set
if [ -z "${TENSORZERO_CLICKHOUSE_VERSION:-}" ]; then
  echo "Error: TENSORZERO_CLICKHOUSE_VERSION environment variable is required"
  exit 1
fi

# ------------------------------------------------------------------------------
# Set the short commit hash
# ------------------------------------------------------------------------------
SHORT_HASH=${BUILDKITE_COMMIT:0:7}

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
export TENSORZERO_CLICKHOUSE_TESTS_TAG=ci-sha-$SHORT_HASH
export TENSORZERO_SKIP_LARGE_FIXTURES=1

# ------------------------------------------------------------------------------
# Pull images referenced by the compose file
# ------------------------------------------------------------------------------
docker compose -f tensorzero-core/tests/e2e/docker-compose.clickhouse.yml pull

# ------------------------------------------------------------------------------
# Run ClickHouse tests container via Docker Compose
# ------------------------------------------------------------------------------
docker compose -f tensorzero-core/tests/e2e/docker-compose.clickhouse.yml run --rm \
  -e TENSORZERO_CI=1 \
  clickhouse-tests
