#!/bin/bash
set -euxo pipefail
cd $(dirname $0)/provider-proxy-cache
for i in {1..5}; do
  if aws s3 --endpoint-url https://19918a216783f0ac9e052233569aef60.r2.cloudflarestorage.com/ sync s3://${PROVIDER_PROXY_CACHE_BUCKET} .; then
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