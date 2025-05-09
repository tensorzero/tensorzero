#!/usr/bin/env bash
set -uxo pipefail
dir_path=$(dirname "$1")
# Since the Cursor integration requires ngrok and ngrok will fail with fake credentials, we skip it
if [[ "$dir_path" == */cursor ]]; then
  exit 0
fi

cd "$dir_path"
docker compose -f "$1" up --wait --wait-timeout 120
status=$?
if [ $status -ne 0 ]; then
  echo "Docker Compose failed for $1 with status $status"
  docker compose -f $1 logs
fi
docker compose -f $1 down --timeout 0
exit $status
