#!/bin/sh

set -euxo pipefail

# Get the short hash from the buildkite environment variable
SHORT_HASH=${BUILDKITE_COMMIT:0:7}
# Build and save container
docker build --load --build-arg BUILDKIT_CONTEXT_KEEP_GIT_DIR=1 -f ui/Dockerfile . -t tensorzero/ui:sha-$SHORT_HASH
docker save tensorzero/ui:sha-$SHORT_HASH > ui-container.tar
