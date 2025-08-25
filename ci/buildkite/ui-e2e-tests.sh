#!/bin/bash
set -euo pipefail

# Cleanup function (always runs on exit)
cleanup() {
  echo "==============================================================================="
  echo "Running cleanup and debug steps..."
  echo "==============================================================================="

  echo "Printing Docker Compose logs..."
  docker compose -f ui/fixtures/docker-compose.e2e.ci.yml logs -t || true
}

# Set trap to run cleanup on any exit (success or failure)
trap cleanup EXIT

# Set up Buildkite test analytics collection
export BUILDKITE_ANALYTICS_TOKEN=$(buildkite-agent secret get UI_E2E_ANALYTICS_ACCESS_TOKEN)
if [ -z "$BUILDKITE_ANALYTICS_TOKEN" ]; then
    echo "Error: BUILDKITE_ANALYTICS_TOKEN is not set"
    exit 1
fi

# ------------------------------------------------------------------------------
# Set the short commit hash
# ------------------------------------------------------------------------------
SHORT_HASH=${BUILDKITE_COMMIT:0:7}

# ------------------------------------------------------------------------------
# Setup Docker Hub credentials
# ------------------------------------------------------------------------------
source ci/buildkite/utils/docker-hub-credentials.sh

# Login to Docker Hub
echo "$DOCKER_HUB_ACCESS_TOKEN" | docker login -u "$DOCKER_HUB_USERNAME" --password-stdin
echo "Logged in to Docker Hub"

# ------------------------------------------------------------------------------
# Setup environment variables
# ------------------------------------------------------------------------------
echo "BUILDKITE_ANALYTICS_TOKEN=$BUILDKITE_ANALYTICS_TOKEN" >> ui/fixtures/.env
{
  echo "TENSORZERO_CLICKHOUSE_URL=http://chuser:chpassword@clickhouse:8123/tensorzero_ui_fixtures"
  echo "TENSORZERO_GATEWAY_URL=http://gateway:3000"
  echo "TENSORZERO_GATEWAY_TAG=ci-sha-$SHORT_HASH"
  echo "TENSORZERO_UI_TAG=ci-sha-$SHORT_HASH"
  echo "TENSORZERO_E2E_TESTS_TAG=ci-sha-$SHORT_HASH"
  echo "TENSORZERO_MOCK_INFERENCE_PROVIDER_TAG=ci-sha-$SHORT_HASH"
  # UI container env vars for optimizer config
  echo "FIREWORKS_BASE_URL=http://mock-inference-provider:3030/fireworks/"
  echo "OPENAI_BASE_URL=http://mock-inference-provider:3030/openai/"
  echo "FIREWORKS_ACCOUNT_ID=fake_fireworks_account"
  echo "VITE_TENSORZERO_FORCE_CACHE_ON=1"
} >> ui/fixtures/.env

# Environment variables only used by the gateway container
{
  echo "FIREWORKS_ACCOUNT_ID=not_used"
  echo "FIREWORKS_API_KEY=not_used"
  echo "OPENAI_API_KEY=not_used"
  echo "ANTHROPIC_API_KEY=not_used"
  # Set real S3 credentials so images can load
  echo "S3_ACCESS_KEY_ID=$(buildkite-agent secret get S3_ACCESS_KEY_ID)"
  echo "S3_SECRET_ACCESS_KEY=$(buildkite-agent secret get S3_SECRET_ACCESS_KEY)"
} >> ui/fixtures/.env-gateway

echo "Set up environment variables"

# ------------------------------------------------------------------------------
# Pull Docker images
# ------------------------------------------------------------------------------
docker compose -f ui/fixtures/docker-compose.e2e.ci.yml pull

echo "Pulled Docker images"

# ------------------------------------------------------------------------------
# Run e2e tests
# ------------------------------------------------------------------------------
# The --grep-invert "@credentials" excludes tests that require real credentials
docker compose -f ui/fixtures/docker-compose.e2e.ci.yml run \
  --rm \
  -e TENSORZERO_CI=1 \
  e2e-tests \
  --grep-invert "@credentials"
