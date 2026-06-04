#!/usr/bin/env bash
# Download the YC Bench baseline rollout traces used by SKILL.md.
#
# The traces are too large to live in the tensorzero repo (98 MiB inferences.jsonl);
# they're hosted via Git LFS in a sibling data repo. This script fetches them into
# this directory and verifies the SHA-256 of each file before declaring success.
#
# Usage: bash baseline_data/fetch.sh

set -euo pipefail

cd "$(dirname "$0")"

BASE_URL="https://media.githubusercontent.com/media/anndvision/data/main/can-ai-agents-improve-ai-agents/baseline_data"

declare -a FILES=(
  "inferences.jsonl 9bac777bcedd790146ed082252ad77d41496f19f1beaecc8acac12cffe55d176"
  "feedback.jsonl   e59685147aea4679d6e617e39e12cd8474e052649f0eb48ccca9f6b2a6fe319d"
)

# Pick a sha256 binary that exists on this platform.
if command -v shasum >/dev/null 2>&1; then
  SHA="shasum -a 256"
elif command -v sha256sum >/dev/null 2>&1; then
  SHA="sha256sum"
else
  echo "error: need shasum or sha256sum on PATH" >&2
  exit 1
fi

verify() {
  local file="$1" expected="$2" actual
  actual=$($SHA "$file" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "  ✗ checksum mismatch for $file" >&2
    echo "    expected: $expected" >&2
    echo "    got:      $actual" >&2
    return 1
  fi
}

for entry in "${FILES[@]}"; do
  read -r name expected_sha <<<"$entry"
  if [[ -f "$name" ]] && verify "$name" "$expected_sha" 2>/dev/null; then
    echo "✓ $name (already present, checksum ok)"
    continue
  fi
  echo "→ downloading $name"
  curl -fL --progress-bar -o "$name" "$BASE_URL/$name"
  verify "$name" "$expected_sha"
  echo "✓ $name"
done

echo
echo "Baseline data ready in $(pwd)"
