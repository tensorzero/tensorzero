#!/bin/sh
set -euo pipefail

# Get the short hash from the buildkite environment variable
SHORT_HASH=${BUILDKITE_COMMIT:0:7}

# Build container
docker build --load --build-arg BUILDKIT_CONTEXT_KEEP_GIT_DIR=1 -f ui/fixtures/Dockerfile.e2e . -t tensorzero/e2e-tests:ci-sha-$SHORT_HASH

# ------------------------------------------------------------------------------
# Setup Docker Hub credentials
# ------------------------------------------------------------------------------
source ci/buildkite/utils/docker-hub-credentials.sh

# Login to Docker Hub (make sure DOCKER_HUB_USERNAME and DOCKER_HUB_ACCESS_TOKEN are set)
echo "$DOCKER_HUB_ACCESS_TOKEN" | docker login -u "$DOCKER_HUB_USERNAME" --password-stdin

# Push to Docker Hub
TAG=tensorzero/e2e-tests:ci-sha-$SHORT_HASH
echo "Pushing $TAG"
docker push $TAG
