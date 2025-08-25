#!/bin/bash
set -euo pipefail


# Set up Buildkite test analytics collection
export BUILDKITE_ANALYTICS_TOKEN=$(buildkite-agent secret get NODE_UNIT_ANALYTICS_ACCESS_TOKEN)
if [ -z "$BUILDKITE_ANALYTICS_TOKEN" ]; then
    echo "Error: BUILDKITE_ANALYTICS_TOKEN is not set"
    exit 1
fi


source ci/buildkite/utils/docker-hub-credentials.sh

# Login to Docker Hub (make sure DOCKER_HUB_USERNAME and DOCKER_HUB_ACCESS_TOKEN are set)
echo "$DOCKER_HUB_ACCESS_TOKEN" | docker login -u "$DOCKER_HUB_USERNAME" --password-stdin

docker compose -f ui/fixtures/docker-compose.unit.yml pull

docker compose -f ui/fixtures/docker-compose.unit.yml run --rm unit-tests
