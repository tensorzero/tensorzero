#!/usr/bin/env bash
set -uxo pipefail

cd $(dirname $1)
docker compose -f $1 up --wait --wait-timeout 30
status=$?
if [ $status -ne 0 ]; then
  echo "Docker Compose failed for $1 with status $status"
  docker compose -f $1 logs
fi
docker compose -f $1 kill
exit $status
