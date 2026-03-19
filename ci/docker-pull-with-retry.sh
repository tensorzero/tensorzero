#!/usr/bin/env bash
# Pulls Docker images with retry logic to handle transient registry failures.
# Usage: ./ci/docker-pull-with-retry.sh IMAGE [IMAGE...]
set -euo pipefail

MAX_RETRIES=3
RETRY_DELAY=5

pull_with_retry() {
    local image="$1"
    for attempt in $(seq 1 "$MAX_RETRIES"); do
        if docker pull "$image"; then
            return 0
        fi
        if [ "$attempt" -lt "$MAX_RETRIES" ]; then
            echo "Failed to pull $image (attempt $attempt/$MAX_RETRIES), retrying in ${RETRY_DELAY}s..."
            sleep "$RETRY_DELAY"
        fi
    done
    echo "Failed to pull $image after $MAX_RETRIES attempts"
    return 1
}

for image in "$@"; do
    pull_with_retry "$image"
done
