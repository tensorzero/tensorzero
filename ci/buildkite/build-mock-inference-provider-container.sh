#!/bin/sh

set -euxo pipefail

# Get the short hash from the buildkite environment variable
SHORT_HASH=${BUILDKITE_COMMIT:0:7}
# Build and save container
docker build --load -f tensorzero-core/tests/mock-inference-provider/Dockerfile . -t tensorzero/mock-inference-provider:sha-$SHORT_HASH
docker save tensorzero/mock-inference-provider:sha-$SHORT_HASH > mock-inference-provider-container.tar
