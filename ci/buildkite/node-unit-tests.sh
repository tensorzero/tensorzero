#!/bin/bash
set -euo pipefail

source ci/buildkite/utils/docker-hub-credentials.sh

# Login to Docker Hub (make sure DOCKER_HUB_USERNAME and DOCKER_HUB_ACCESS_TOKEN are set)
echo "$DOCKER_HUB_ACCESS_TOKEN" | docker login -u "$DOCKER_HUB_USERNAME" --password-stdin

docker compose -f ui/fixtures/docker-compose.unit.yml pull

docker compose -f ui/fixtures/docker-compose.unit.yml run --rm unit-tests
