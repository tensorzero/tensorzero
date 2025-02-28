#!/usr/bin/env bash
set -euxo pipefail

docker run \
   -p 8000:9000 \
   -p 8001:9001 \
   -e "MINIO_ROOT_USER=tensorzero" \
   -e "MINIO_ROOT_PASSWORD=tensorzero" \
   quay.io/minio/minio server /data --console-address ":9001"
