#!/bin/bash
set -euo pipefail

# Clear disk space
# ./ci/free-disk-space.sh

# Get all env vars
source ci/buildkite/utils/live-tests-env.sh


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

# Get the provider-proxy-cache
buildkite-agent artifact download provider-proxy-cache.tar.gz ci
tar -xzvf ci/provider-proxy-cache.tar.gz

# Write the GCP JWT key to a file
echo $(buildkite-agent secret get MISTRAL_API_KEY) > tensorzero-core/tests/e2e/gcp-jwt-key.json

# ------------------------------------------------------------------------------
# Run live tests container via Docker Compose and capture exit code
# ------------------------------------------------------------------------------
set +e
docker compose -f tensorzero-core/tests/e2e/docker-compose.live.yml run --rm \
  -e TENSORZERO_CI=1 \
  live-tests
TEST_EXIT_CODE=$?
set -e


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
