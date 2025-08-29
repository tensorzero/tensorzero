#!/bin/bash
set -euo pipefail

# Common CI trap helper
source ci/buildkite/utils/trap-helpers.sh
# Print logs from the ClickHouse compose stack on exit
tz_setup_compose_logs_trap tensorzero-core/tests/e2e/docker-compose.clickhouse.yml

# Get all env vars
source ci/buildkite/utils/live-tests-env.sh

# Set BUILDKITE_ANALYTICS_TOKEN
export BUILDKITE_ANALYTICS_TOKEN=$(buildkite-agent secret get LIVE_TESTS_ANALYTICS_ACCESS_TOKEN)

if [ -z "$BUILDKITE_ANALYTICS_TOKEN" ]; then
    echo "Error: BUILDKITE_ANALYTICS_TOKEN is not set"
    exit 1
fi


# ------------------------------------------------------------------------------
# Set the short commit hash
# ------------------------------------------------------------------------------
SHORT_HASH=${BUILDKITE_COMMIT:0:7}
export TENSORZERO_GATEWAY_TAG=ci-sha-$SHORT_HASH
export TENSORZERO_MOCK_INFERENCE_PROVIDER_TAG=ci-sha-$SHORT_HASH
export TENSORZERO_PROVIDER_PROXY_TAG=ci-sha-$SHORT_HASH
export TENSORZERO_LIVE_TESTS_TAG=ci-sha-$SHORT_HASH

# E2E tests don't need large fixtures
export TENSORZERO_SKIP_LARGE_FIXTURES=1


# Get the fixtures
buildkite-agent artifact download fixtures.tar.gz ui/fixtures
tar -xzvf ui/fixtures/fixtures.tar.gz

# Get the provider-proxy-cache
buildkite-agent artifact download provider-proxy-cache.tar.gz ci
tar -xzvf ci/provider-proxy-cache.tar.gz

# Write the GCP JWT key to a file
echo $(buildkite-agent secret get GCP_JWT_KEY) > gcp_jwt_key.json

# ------------------------------------------------------------------------------
# Docker Hub auth (for pulling built images)
# ------------------------------------------------------------------------------
source ci/buildkite/utils/docker-hub-credentials.sh
echo "$DOCKER_HUB_ACCESS_TOKEN" | docker login -u "$DOCKER_HUB_USERNAME" --password-stdin
echo "Logged in to Docker Hub"

# ------------------------------------------------------------------------------
# Pull images referenced by the compose file
# ------------------------------------------------------------------------------
docker compose -f tensorzero-core/tests/e2e/docker-compose.live.yml pull

# ------------------------------------------------------------------------------
# Run live tests container via Docker Compose and capture exit code
# ------------------------------------------------------------------------------
set +e
docker compose -f tensorzero-core/tests/e2e/docker-compose.live.yml run --rm \
  -e TENSORZERO_CI=1 \
  live-tests
TEST_EXIT_CODE=$?
set -e


# Upload the test JUnit XML files (regardless of test results)
if [ -f "target/nextest/e2e/junit.xml" ]; then
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
    echo "analytics uploaded"
else
    echo "Warning: JUnit XML file not found at target/nextest/e2e/junit.xml"
fi

# Exit with the original test exit code
exit $TEST_EXIT_CODE
