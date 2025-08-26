#!/bin/bash
set -euo pipefail

# Get the short hash from the buildkite environment variable
SHORT_HASH=${BUILDKITE_COMMIT:0:7}
TAG=tensorzero/gateway-e2e:ci-sha-$SHORT_HASH

source ci/buildkite/utils/docker-hub-credentials.sh

# Login to Docker Hub (make sure DOCKER_HUB_USERNAME and DOCKER_HUB_ACCESS_TOKEN are set)
echo "$DOCKER_HUB_ACCESS_TOKEN" | docker login -u "$DOCKER_HUB_USERNAME" --password-stdin

# Pull latest image for caching (ignore errors if image doesn't exist)
# docker pull tensorzero/gateway-e2e:latest || true

# Build the container with cache
docker build --load --build-arg BUILDKIT_CONTEXT_KEEP_GIT_DIR=1 \
  --cache-from tensorzero/gateway-e2e:latest \
  -f tensorzero-core/tests/e2e/Dockerfile.gateway.e2e . -t $TAG

# Tag with latest and push both tags
docker tag $TAG tensorzero/gateway-e2e:latest
echo "Pushing $TAG"
docker push $TAG
echo "Pushing tensorzero/gateway-e2e:latest"
docker push tensorzero/gateway-e2e:latest
