#!/bin/bash
set -euo pipefail

# Common CI trap helper
source ci/buildkite/utils/trap-helpers.sh
# Print logs from the unit test compose stack on exit
tz_setup_compose_logs_trap ui/fixtures/docker-compose.unit.yml

# Set up Buildkite test analytics collection
export BUILDKITE_ANALYTICS_TOKEN=$(buildkite-agent secret get NODE_UNIT_ANALYTICS_ACCESS_TOKEN)
if [ -z "$BUILDKITE_ANALYTICS_TOKEN" ]; then
    echo "Error: BUILDKITE_ANALYTICS_TOKEN is not set"
    exit 1
fi
SHORT_HASH=${BUILDKITE_COMMIT:0:7}


source ci/buildkite/utils/docker-hub-credentials.sh

# Login to Docker Hub (make sure DOCKER_HUB_USERNAME and DOCKER_HUB_ACCESS_TOKEN are set)
echo "$DOCKER_HUB_ACCESS_TOKEN" | docker login -u "$DOCKER_HUB_USERNAME" --password-stdin
echo "Logged in to Docker Hub"


echo $BUILDKITE_ANALYTICS_TOKEN >> ui/fixtures/.env
{
  echo "TENSORZERO_CLICKHOUSE_URL=http://chuser:chpassword@localhost:8123/tensorzero_ui_fixtures"
  echo "TENSORZERO_COMMIT_TAG=ci-sha-$SHORT_HASH"
  echo "TENSORZERO_INTERNAL_MOCK_PROVIDER_API=http://mock-provider-api:3030"
} >> ui/fixtures/.env
{
  echo "FIREWORKS_API_KEY=not_used"
  echo "OPENAI_API_KEY=not_used"
} >> ui/fixtures/.env-gateway
echo "Set up environment variables"

docker compose -f ui/fixtures/docker-compose.unit.yml pull

echo "Pulled Docker images"

docker compose -f ui/fixtures/docker-compose.unit.yml run --rm node-unit-tests
