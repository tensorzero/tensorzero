#!/bin/bash
# To be run from repo root
# Calls the ./ci/download-provider-proxy-cache.sh script
# then makes a tarball for Buildkite artifact

export R2_ACCESS_KEY_ID=$(buildkite-agent secret get R2_ACCESS_KEY_ID)
if [ -z "$R2_ACCESS_KEY_ID" ]; then
    echo "Error: R2_ACCESS_KEY_ID is not set"
    exit 1
fi

export R2_SECRET_ACCESS_KEY=$(buildkite-agent secret get R2_SECRET_ACCESS_KEY)
if [ -z "$R2_SECRET_ACCESS_KEY" ]; then
    echo "Error: R2_SECRET_ACCESS_KEY is not set"
    exit 1
fi

AWS_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY ./ci/download-provider-proxy-cache.sh

tar -czvf provider-proxy-cache.tar.gz ci/provider-proxy-cache
