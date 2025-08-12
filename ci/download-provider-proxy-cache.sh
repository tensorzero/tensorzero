#!/bin/bash
set -euxo pipefail
cd $(dirname $0)/provider-proxy-cache
aws s3 --endpoint-url https://19918a216783f0ac9e052233569aef60.r2.cloudflarestorage.com/ sync s3://provider-proxy-cache .
