#!/bin/bash
set -euo pipefail

# Get the short hash from the buildkite environment variable
SHORT_HASH=${BUILDKITE_COMMIT:0:7}
TAG=tensorzero/gateway-e2e:ci-sha-$SHORT_HASH
LATEST_TAG=tensorzero/gateway-e2e:latest

source ci/buildkite/utils/docker-hub-credentials.sh

# Login to Docker Hub (make sure DOCKER_HUB_USERNAME and DOCKER_HUB_ACCESS_TOKEN are set)
echo "$DOCKER_HUB_ACCESS_TOKEN" | docker login -u "$DOCKER_HUB_USERNAME" --password-stdin

# Pull latest image for caching (ignore errors if image doesn't exist)
docker pull $LATEST_TAG || true

# Build the container with cache
docker build --load --build-arg BUILDKIT_CONTEXT_KEEP_GIT_DIR=1 \
  --build-arg PROFILE=release \ # Use 'release' instead of 'performance', which is faster to build
  --build-arg CARGO_BUILD_FLAGS="--features e2e_tests" \
  --cache-from $LATEST_TAG \
  -f gateway/Dockerfile . -t $TAG

# Tag with latest and push both tags
docker tag $TAG $LATEST_TAG
echo "Pushing $TAG"
docker push $TAG
echo "Pushing $LATEST_TAG"
docker push $LATEST_TAG
