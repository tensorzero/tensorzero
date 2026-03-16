#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <image> <docker-buildx-args...>" >&2
  exit 1
fi

image="$1"
shift

echo "Checking whether ${image} already exists on Docker Hub..."
if docker manifest inspect "${image}" >/dev/null 2>&1; then
  echo "Image ${image} already exists; skipping rebuild because sha tags are immutable."
  exit 0
fi

echo "Building and pushing ${image}..."
set +e
docker buildx build "$@" -t "${image}" --push
status=$?
set -e

if [[ ${status} -eq 0 ]]; then
  echo "Built and pushed ${image}."
  exit 0
fi

if docker manifest inspect "${image}" >/dev/null 2>&1; then
  echo "Image ${image} became available after a failed push; continuing."
  exit 0
fi

exit "${status}"
