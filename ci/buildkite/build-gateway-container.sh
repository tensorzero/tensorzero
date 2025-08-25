#!/bin/bash
set -euo pipefail

# Get the short hash from the buildkite environment variable
SHORT_HASH=${BUILDKITE_COMMIT:0:7}

# Build the container
docker build --load --build-arg BUILDKIT_CONTEXT_KEEP_GIT_DIR=1 -f gateway/Dockerfile . -t tensorzero/gateway-dev:ci-sha-$SHORT_HASH

source ci/buildkite/utils/docker-hub-credentials.sh

# Login to Docker Hub (make sure DOCKER_HUB_USERNAME and DOCKER_HUB_ACCESS_TOKEN are set)
echo "$DOCKER_HUB_ACCESS_TOKEN" | docker login -u "$DOCKER_HUB_USERNAME" --password-stdin

# Push to Docker Hub
docker push tensorzero/gateway-dev:ci-sha-$SHORT_HASH
