#!/usr/bin/env bash
# Stream logs from the remote (docker compose and/or gateway) to the local terminal.
#
# Use same env as deploy: TENSORZERO_DEPLOY_REMOTE_HOST, TENSORZERO_DEPLOY_REMOTE_BASE,
#   TENSORZERO_DEPLOY_REMOTE_DIR (optional).
#
# Usage:
#   ./scripts/remote/logs.sh              # all docker compose services (follow)
#   ./scripts/remote/logs.sh compose      # same
#   ./scripts/remote/logs.sh gateway      # gateway process log (/tmp/e2e_gateway.log)
#   ./scripts/remote/logs.sh <service>   # one compose service (e.g. fixtures-postgres)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_HOST="${TENSORZERO_DEPLOY_REMOTE_HOST:-tensorzero}"
REMOTE_BASE="${TENSORZERO_DEPLOY_REMOTE_BASE:-/home/ubuntu}"
REMOTE_COMPOSE="tensorzero-core/tests/e2e/docker-compose.yml"

# Resolve REMOTE_DIR (same logic as deploy.sh: in-repo vs parent)
if [ -d "${SCRIPT_DIR}/../../tensorzero" ]; then
  PARENT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  REMOTE_DIR="${REMOTE_BASE}/tensorzero"
  if [ "${1:-}" = "1" ]; then
    REMOTE_DIR="${TENSORZERO_DEPLOY_REMOTE_DIR:-$REMOTE_BASE/tensorzero-1}"
    shift
  elif [ "${1:-}" = "0" ]; then
    shift
  fi
else
  REMOTE_DIR="${TENSORZERO_DEPLOY_REMOTE_DIR:-$REMOTE_BASE/$(basename "$(cd "${SCRIPT_DIR}/../.." && pwd)")}"
fi

WHAT="${1:-compose}"

case "$WHAT" in
  gateway)
    echo "--- Gateway log on ${REMOTE_HOST} (Ctrl+C to stop) ---"
    ssh "${REMOTE_HOST}" "tail -f /tmp/e2e_gateway.log 2>/dev/null || echo 'No gateway log found (start with: ./scripts/remote/deploy.sh gateway)'"
    ;;
  compose|"")
    echo "--- Docker compose logs on ${REMOTE_HOST}:${REMOTE_DIR} (Ctrl+C to stop) ---"
    ssh "${REMOTE_HOST}" "cd ${REMOTE_DIR} && docker compose -f ${REMOTE_COMPOSE} logs -f"
    ;;
  *)
    echo "--- Docker compose service '$WHAT' on ${REMOTE_HOST}:${REMOTE_DIR} (Ctrl+C to stop) ---"
    ssh "${REMOTE_HOST}" "cd ${REMOTE_DIR} && docker compose -f ${REMOTE_COMPOSE} logs -f ${WHAT}"
    ;;
esac
