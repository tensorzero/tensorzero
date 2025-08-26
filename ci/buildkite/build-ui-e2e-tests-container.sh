#!/bin/sh
set -euo pipefail

# Get the short hash from the buildkite environment variable
SHORT_HASH=${BUILDKITE_COMMIT:0:7}
TAG=tensorzero/ui-e2e-tests:ci-sha-$SHORT_HASH

# ------------------------------------------------------------------------------
# Setup Docker Hub credentials
# ------------------------------------------------------------------------------
source ci/buildkite/utils/docker-hub-credentials.sh

# Login to Docker Hub (make sure DOCKER_HUB_USERNAME and DOCKER_HUB_ACCESS_TOKEN are set)
echo "$DOCKER_HUB_ACCESS_TOKEN" | docker login -u "$DOCKER_HUB_USERNAME" --password-stdin

# Pull latest image for caching (ignore errors if image doesn't exist)
docker pull tensorzero/ui-e2e-tests:latest || true

# Build container with cache
docker build --load --build-arg BUILDKIT_CONTEXT_KEEP_GIT_DIR=1 \
  --cache-from tensorzero/ui-e2e-tests:latest \
  -f ui/fixtures/Dockerfile.e2e . -t $TAG

# Tag with latest and push both tags
docker tag $TAG tensorzero/ui-e2e-tests:latest
echo "Pushing $TAG"
docker push $TAG
echo "Pushing tensorzero/ui-e2e-tests:latest"
docker push tensorzero/ui-e2e-tests:latest
