#!/bin/sh
set -euo pipefail

# Get the short hash from the buildkite environment variable
SHORT_HASH=${BUILDKITE_COMMIT:0:7}
TAG=tensorzero/clickhouse-tests:ci-sha-$SHORT_HASH

# Load Docker Hub credentials and login
source ci/buildkite/utils/docker-hub-credentials.sh
echo "$DOCKER_HUB_ACCESS_TOKEN" | docker login -u "$DOCKER_HUB_USERNAME" --password-stdin

# Pull latest image for caching (ignore errors if image doesn't exist)
docker pull tensorzero/clickhouse-tests:latest || true

# Build container from the ClickHouse tests Dockerfile with cache
docker build --load --build-arg BUILDKIT_CONTEXT_KEEP_GIT_DIR=1 \
  --cache-from tensorzero/clickhouse-tests:latest \
  -f tensorzero-core/tests/e2e/Dockerfile.clickhouse \
  . -t $TAG

# Tag with latest and push both tags
docker tag $TAG tensorzero/clickhouse-tests:latest
echo "Pushing $TAG"
docker push $TAG
echo "Pushing tensorzero/clickhouse-tests:latest"
docker push tensorzero/clickhouse-tests:latest
