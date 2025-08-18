#!/usr/bin/env bash
set -euxo pipefail


cargo build --bin provider-proxy
if [[ "${1:-}" = "ci" ]]; then
  cargo run --bin provider-proxy -- --cache-path ./ci/provider-proxy-cache --mode read-old-write-new > provider_proxy_logs.txt 2>&1 &
else
  cargo run --bin provider-proxy -- --cache-path ./ci/provider-proxy-cache --mode read-old-write-new
fi
