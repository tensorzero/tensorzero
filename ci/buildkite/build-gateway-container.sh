#!/bin/bash
set -euo pipefail

# Get the short hash from the buildkite environment variable
SHORT_HASH=${BUILDKITE_COMMIT:0:7}

# Build the container
docker build --load --build-arg BUILDKIT_CONTEXT_KEEP_GIT_DIR=1 -f gateway/Dockerfile . -t tensorzero/gateway-dev:ci-sha-$SHORT_HASH

# Set up all other environment variables
export DOCKER_HUB_ACCESS_TOKEN=$(buildkite-agent secret get DOCKER_HUB_ACCESS_TOKEN)
if [ -z "$DOCKER_HUB_ACCESS_TOKEN" ]; then
    echo "Error: DOCKER_HUB_ACCESS_TOKEN is not set"
    exit 1
fi

export DOCKER_HUB_USERNAME=$(buildkite-agent secret get DOCKER_HUB_USERNAME)
if [ -z "$DOCKER_HUB_USERNAME" ]; then
    echo "Error: DOCKER_HUB_USERNAME is not set"
    exit 1
fi



# Login to Docker Hub (make sure DOCKER_HUB_USERNAME and DOCKER_HUB_ACCESS_TOKEN are set)
echo "$DOCKER_HUB_ACCESS_TOKEN" | docker login -u "$DOCKER_HUB_USERNAME" --password-stdin

# Push to Docker Hub
docker push tensorzero/gateway-dev:ci-sha-$SHORT_HASH
