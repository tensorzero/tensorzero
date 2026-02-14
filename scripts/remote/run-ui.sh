#!/usr/bin/env bash
# Run the TensorZero UI against the tunneled gateway (localhost:3000).
#
# Use in a separate terminal after starting the tunnel with:
#   ./scripts/remote/deploy.sh gateway
#
# Usage:  ./scripts/remote/run-ui.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UI_DIR="${REPO_ROOT}/ui"

if [ ! -f "${UI_DIR}/package.json" ]; then
  echo "Error: UI not found at ${UI_DIR}"
  exit 1
fi

export TENSORZERO_GATEWAY_URL="${TENSORZERO_GATEWAY_URL:-http://localhost:3000}"
echo "  TENSORZERO_GATEWAY_URL=${TENSORZERO_GATEWAY_URL}"
echo "  Starting UI (Ctrl+C to stop)"
echo ""
cd "${UI_DIR}" && exec pnpm dev
