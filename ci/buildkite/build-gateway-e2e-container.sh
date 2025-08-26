#!/bin/bash
set -euo pipefail

# Get the short hash from the buildkite environment variable
SHORT_HASH=${BUILDKITE_COMMIT:0:7}

# Build the container
TAG=tensorzero/gateway-e2e:ci-sha-$SHORT_HASH
docker build --load --build-arg BUILDKIT_CONTEXT_KEEP_GIT_DIR=1 -f tensorzero-core/tests/e2e/Dockerfile.gateway.e2e . -t $TAG

source ci/buildkite/utils/docker-hub-credentials.sh

# Login to Docker Hub (make sure DOCKER_HUB_USERNAME and DOCKER_HUB_ACCESS_TOKEN are set)
echo "$DOCKER_HUB_ACCESS_TOKEN" | docker login -u "$DOCKER_HUB_USERNAME" --password-stdin

# Push to Docker Hub
echo "Pushing $TAG"
docker push $TAG
