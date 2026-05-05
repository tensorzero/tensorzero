#!/usr/bin/env bash
# Wait for the apt/dpkg locks to be released.
# GitHub-hosted Ubuntu runners sometimes run unattended apt operations in the
# background that race with workflow steps invoking apt (e.g. Playwright's
# `--with-deps` flag). Without this wait, the retry loop's short sleep often
# isn't enough to ride out a long apt-get run.
#
# Usage: wait-for-dpkg.sh [max_wait_seconds]
#
# Always exits 0, even on timeout — the caller should still attempt the
# install so any actual failure surfaces with a real error message.

set -u

max_wait="${1:-120}"
poll=5
elapsed=0

# In order: the lock the apt frontend takes first, the dpkg lock taken once
# the install proceeds, and the apt lists lock taken during `apt-get update`.
locks=(
  /var/lib/dpkg/lock-frontend
  /var/lib/dpkg/lock
  /var/lib/apt/lists/lock
)

while [ "$elapsed" -lt "$max_wait" ]; do
  if ! sudo fuser "${locks[@]}" >/dev/null 2>&1; then
    exit 0
  fi
  echo "dpkg/apt lock held; waited ${elapsed}s..."
  sleep "$poll"
  elapsed=$((elapsed + poll))
done

echo "dpkg/apt lock still held after ${max_wait}s — proceeding anyway" >&2
exit 0
