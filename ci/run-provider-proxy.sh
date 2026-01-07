#!/usr/bin/env bash
set -euxo pipefail

# Provider proxy cache mode can be configured via PROVIDER_PROXY_CACHE_MODE environment variable.
# Options: read-old-write-new (default), read-only, read-write
# See: https://github.com/tensorzero/tensorzero/issues/5380
CACHE_MODE="${PROVIDER_PROXY_CACHE_MODE:-read-old-write-new}"

cargo build --bin provider-proxy
if [[ "${1:-}" = "ci" ]]; then
  cargo run --bin provider-proxy -- --cache-path ./ci/provider-proxy-cache --mode "$CACHE_MODE" > provider_proxy_logs.txt 2>&1 &
else
  cargo run --bin provider-proxy -- --cache-path ./ci/provider-proxy-cache --mode "$CACHE_MODE"
fi
