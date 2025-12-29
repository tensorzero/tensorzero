#!/bin/bash
set -euo pipefail

# Common CI trap helper
source ci/buildkite/utils/trap-helpers.sh
# Print logs from the E2E compose stack on exit
tz_setup_compose_logs_trap ui/fixtures/docker-compose.e2e.ci.yml

# Set up Buildkite test analytics collection
export BUILDKITE_ANALYTICS_TOKEN=$(buildkite-agent secret get UI_E2E_ANALYTICS_ACCESS_TOKEN)
if [ -z "$BUILDKITE_ANALYTICS_TOKEN" ]; then
    echo "Error: BUILDKITE_ANALYTICS_TOKEN is not set"
    exit 1
fi

# Install zip utility if not available
if ! command -v zip &> /dev/null; then
    apt-get update && apt-get install -y zip
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
  echo "TENSORZERO_COMMIT_TAG=ci-sha-$SHORT_HASH"
  echo "TENSORZERO_INTERNAL_MOCK_PROVIDER_API=http://mock-provider-api:3030"
  echo "VITE_TENSORZERO_FORCE_CACHE_ON=1"
} >> ui/fixtures/.env

# Environment variables only used by the gateway container
{
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
# Ensure artifact output directories exist on host (mounted into container)
mkdir -p ui/fixtures/test-results ui/fixtures/playwright-report

# The --grep-invert "@credentials" excludes tests that require real credentials
docker compose -f ui/fixtures/docker-compose.e2e.ci.yml run \
  --rm \
  -e TENSORZERO_CI=1 \
  ui-e2e-tests \
  --grep-invert "@credentials"

# Create a comprehensive Playwright artifacts zip
cd ui/fixtures
zip -r ui-e2e-artifacts.zip \
  test-results/ \
  playwright-report/ \
  -x "*.log" "*.tmp"
cd ../..

echo "E2E test artifacts packaged into ui/fixtures/ui-e2e-artifacts.zip"
echo "Contents include:"
echo " - test-results/ (screenshots, videos, traces)"
echo " - playwright-report/ (HTML report)"
