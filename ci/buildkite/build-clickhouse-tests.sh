#!/bin/sh
set -euo pipefail

# Get the short hash from the buildkite environment variable
SHORT_HASH=${BUILDKITE_COMMIT:0:7}

# Build container from the ClickHouse tests Dockerfile
docker build --load --build-arg BUILDKIT_CONTEXT_KEEP_GIT_DIR=1 \
  -f tensorzero-core/tests/e2e/Dockerfile.clickhouse \
  . -t tensorzero/clickhouse-tests:ci-sha-$SHORT_HASH

# Load Docker Hub credentials and push
source ci/buildkite/utils/docker-hub-credentials.sh
echo "$DOCKER_HUB_ACCESS_TOKEN" | docker login -u "$DOCKER_HUB_USERNAME" --password-stdin
docker push tensorzero/clickhouse-tests:ci-sha-$SHORT_HASH
