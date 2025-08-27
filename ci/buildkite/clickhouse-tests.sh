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

# Get the fixtures
buildkite-agent artifact download fixtures.tar.gz ui/fixtures
tar -xzvf ui/fixtures/fixtures.tar.gz

sleep infinity

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

# Upload the test JUnit XML files
curl -X POST \
  -H "Authorization: Token token=$BUILDKITE_ANALYTICS_TOKEN" \
  -F "format=junit" \
  -F "data=@target/nextest/clickhouse/junit.xml" \
  -F "run_env[CI]=buildkite" \
  -F "run_env[key]=$BUILDKITE_BUILD_ID" \
  -F "run_env[number]=$BUILDKITE_BUILD_NUMBER" \
  -F "run_env[job_id]=$BUILDKITE_JOB_ID" \
  -F "run_env[branch]=$BUILDKITE_BRANCH" \
  -F "run_env[commit_sha]=$BUILDKITE_COMMIT" \
  -F "run_env[message]=$BUILDKITE_MESSAGE" \
  -F "run_env[url]=$BUILDKITE_BUILD_URL" \
  https://analytics-api.buildkite.com/v1/uploads
