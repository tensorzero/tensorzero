#!/bin/sh

set -euxo pipefail

# Get the short hash from the buildkite environment variable
SHORT_HASH=${BUILDKITE_COMMIT:0:7}
# Build and save container
docker build --load --build-arg BUILDKIT_CONTEXT_KEEP_GIT_DIR=1 -f gateway/Dockerfile . -t tensorzero/gateway:sha-$SHORT_HASH
docker save tensorzero/gateway:sha-$SHORT_HASH > gateway-container.tar
