#!/bin/bash
set -euxo pipefail

cd $(dirname $0)/s3-fixtures

aws s3 --endpoint-url https://19918a216783f0ac9e052233569aef60.r2.cloudflarestorage.com/ sync . s3://tensorzero-fixtures --checksum-algorithm CRC32
