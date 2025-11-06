#!/bin/bash
set -euxo pipefail

cd $(dirname $0)/provider-proxy-cache

export AWS_RETRY_MODE=standard
export AWS_MAX_ATTEMPTS=10

# The following settings attempt to avoid the error:
# "An error occurred (ServiceUnavailable) when calling the PutObject operation: Reduce your concurrent request rate for the same object."
export AWS_S3_MAX_CONCURRENT_REQUESTS=3
export AWS_S3_MULTIPART_THRESHOLD=67108864  # bytes (64MB)
export AWS_S3_MULTIPART_CHUNKSIZE=67108864  # bytes (64MB)

for i in {1..5}; do
    # Use --delete to remove stale cache files from the bucket.
    # This doesn't do anything for merge queue runs (since we start by downloading the entire bucket),
    # but it will help for cron job runs, where we intentionally start with an empty provider-proxy cache.
    if aws s3 --endpoint-url https://19918a216783f0ac9e052233569aef60.r2.cloudflarestorage.com/ sync . s3://${PROVIDER_PROXY_CACHE_BUCKET} --delete --no-progress; then
        break
    else
        echo "Attempt $i failed. Retrying..."
        if [ $i -eq 5 ]; then
            echo "All attempts failed."
            exit 1
        fi
        sleep 5
    fi
done
