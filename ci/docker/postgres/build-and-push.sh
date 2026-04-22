#!/usr/bin/env bash
set -euo pipefail

# Build and push multi-arch tensorzero/postgres images to Docker Hub.
# Each image bundles pgvector and pg_cron on top of the official postgres image.
#
# Prerequisites:
#   - docker buildx (with a builder that supports linux/amd64,linux/arm64)
#   - Logged in to Docker Hub (`docker login`)
#
# Usage:
#   ./build-and-push.sh          # build and push all versions
#   ./build-and-push.sh 17       # build and push only PG 17

REPO="tensorzero/postgres"
PLATFORMS="linux/amd64,linux/arm64"

# major:patch
ALL_VERSIONS="18:18.3 17:17.9 16:16.13 15:15.17 14:14.22"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

build_and_push() {
  local major="$1"
  local patch="$2"

  echo "==> Building ${REPO}:${patch} and ${REPO}:${major} for ${PLATFORMS}"
  docker buildx build \
    --platform "${PLATFORMS}" \
    --build-arg "PG_MAJOR=${major}" \
    --build-arg "PG_MINOR=${patch}" \
    --tag "${REPO}:${patch}" \
    --tag "${REPO}:${major}" \
    --push \
    "${SCRIPT_DIR}"
}

if [[ $# -ge 1 ]]; then
  FILTER="$1"
  for entry in $ALL_VERSIONS; do
    major="${entry%%:*}"
    patch="${entry##*:}"
    if [[ "$major" == "$FILTER" ]]; then
      build_and_push "$major" "$patch"
      exit 0
    fi
  done
  echo "Unknown version: ${FILTER}"
  echo "Available: $(echo "$ALL_VERSIONS" | tr ' ' '\n' | cut -d: -f1 | tr '\n' ' ')"
  exit 1
else
  for entry in $ALL_VERSIONS; do
    major="${entry%%:*}"
    patch="${entry##*:}"
    build_and_push "$major" "$patch"
  done
fi
