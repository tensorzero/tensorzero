#!/usr/bin/env bash
set -uxo pipefail
script_path=$(dirname $(realpath -s $0))
dir_path=$(dirname "$1")
# Since the Cursor integration requires ngrok and ngrok will fail with fake credentials, we skip it
if [[ "$dir_path" == */cursor ]]; then
  exit 0
fi

# We already use these in CI, so no need to test it twice
if [[ "$1" == *ui/fixtures/docker-compose.yml || \
      "$1" == *ui/fixtures/docker-compose.e2e.yml || \
      "$1" == *tensorzero-core/tests/e2e/docker-compose.yml || \
      "$1" == *tensorzero-core/tests/e2e/docker-compose.replicated.yml || \
      "$1" == *tensorzero-core/tests/e2e/docker-compose.live.yml || \
      "$1" == *tensorzero-core/tests/e2e/docker-compose.clickhouse.yml || \
      "$1" == *tensorzero-core/tests/e2e/docker-compose-common.yml || \
      "$1" == *tensorzero-optimizers/tests/docker-compose.yml || \
      "$1" == *ui/docker-compose.yml || \
      "$1" == *ci/internal-network.yml ]]; then
  exit 0
fi

cd "$dir_path"
docker compose -f "$1" -f $script_path/internal-network.yml up --wait --wait-timeout 360
status=$?
if [ $status -ne 0 ]; then
  echo "Docker Compose failed for $1 with status $status"
  docker compose -f $1 logs
fi
docker compose -f $1 down --timeout 0
exit $status
