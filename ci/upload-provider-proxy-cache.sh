#!/bin/bash
set -euxo pipefail
cd $(dirname $0)/provider-proxy-cache

export AWS_RETRY_MODE=standard
export AWS_MAX_ATTEMPTS=10

# Avoid: "Reduce your concurrent request rate for the same object"
aws configure set default.s3.max_concurrent_requests 3
aws configure set default.s3.multipart_threshold 64MB
aws configure set default.s3.multipart_chunksize 64MB

# Use --delete to remove stale cache files from the bucket.
# This doesn't do anything for merge queue runs (since we start by downloading the entire bucket),
# but it will help for cron job runs, where we intentionally start with an empty provider-proxy cache.
aws s3 --endpoint-url https://19918a216783f0ac9e052233569aef60.r2.cloudflarestorage.com/ sync . s3://provider-proxy-cache --checksum-algorithm CRC32 --delete --no-progress
