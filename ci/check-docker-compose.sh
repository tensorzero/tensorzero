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
      "$1" == *tensorzero-core/tests/optimization/docker-compose.yml || \
      "$1" == *ui/docker-compose.yml || \
      "$1" == *ci/internal-network.yml ]]; then
  exit 0
fi


# DEBUG ONLY DO NOT MERGE
if [[ "$1" != *examples/docs/guides/operations/enforce-custom-rate-limits/docker-compose.yml ]]; then
  exit 0
fi
echo "!!!!! A"

echo "!!!!! B"
# /DEBUG ONLY DO NOT MERGE

cd "$dir_path"

# Start monitoring docker events in background
echo "!!!!! Starting docker events monitor"
docker events --filter "type=container" &
EVENTS_PID=$!

# Start compose with verbose output
echo "!!!!! Starting docker compose up"
docker compose -f "$1" -f $script_path/internal-network.yml up --wait --wait-timeout 360 &
COMPOSE_PID=$!

# Monitor container status and ClickHouse logs while waiting
echo "!!!!! Monitoring containers"
for i in {1..180}; do
  echo "!!!!! === Status check $i ==="
  docker compose -f "$1" ps

  # Try to get clickhouse logs
  if docker ps --format '{{.Names}}' | grep -q clickhouse; then
    echo "!!!!! ClickHouse logs:"
    docker compose -f "$1" logs --tail=20 clickhouse || true

    echo "!!!!! ClickHouse container inspect:"
    docker inspect $(docker ps -qf "name=clickhouse") | grep -A 5 "Health" || true
  fi

  # Check if compose process is still running
  if ! kill -0 $COMPOSE_PID 2>/dev/null; then
    echo "!!!!! Compose process exited"
    break
  fi

  sleep 2
done

# Wait for compose to finish
wait $COMPOSE_PID
status=$?

# Stop events monitor
kill $EVENTS_PID 2>/dev/null || true

echo "!!!!! Final status: $status"
if [ $status -ne 0 ]; then
  echo "Docker Compose failed for $1 with status $status"
  echo "!!!!! All container logs:"
  docker compose -f $1 logs
fi
docker compose -f $1 down --timeout 0
exit $status
