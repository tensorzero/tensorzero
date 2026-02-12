#!/usr/bin/env bash
# Deploy a local tensorzero clone to a remote machine and run Rust tests.
#
# Usage:
#   ./deploy.sh              # defaults to workspace 0, sync + check
#   ./deploy.sh 0            # workspace 0: tensorzero/ <-> remote tensorzero/
#   ./deploy.sh 1            # workspace 1: tensorzero-1/ <-> remote tensorzero-1/
#   ./deploy.sh 0 sync       # sync only, no checks
#   ./deploy.sh 0 check      # sync + cargo check + clippy + unit tests (default)
#   ./deploy.sh 0 e2e        # sync + docker compose + gateway + e2e tests (all on remote)
#   ./deploy.sh 0 e2e cost   # same but only run tests matching "cost"
#   ./deploy.sh 1 cleanup    # stop docker compose and gateway on remote (no sync)
#
# From repo root:  ./ci/deploy.sh [check|e2e|sync|cleanup]  (no workspace number)
# Environment:     TENSORZERO_DEPLOY_REMOTE_HOST, TENSORZERO_DEPLOY_REMOTE_BASE,
#                  TENSORZERO_DEPLOY_REMOTE_DIR  (override for your remote)
#
# Local layout (next to this script when run from parent):
#   tensorzero/       <- workspace 0 (default clone)
#   tensorzero-1/     <- workspace 1 (second clone)
#
# Remote layout (/home/ubuntu/):
#   tensorzero/       <- mirror of workspace 0
#   tensorzero-1/     <- mirror of workspace 1
#
# E2E mode:
#   Everything runs on the remote: docker compose, gateway, and tests.
#   When switching workspaces, docker compose is torn down and restarted
#   on the remote to reset databases (different branches may have
#   different migrations).

set -euo pipefail

# Use Homebrew rsync (3.x) instead of macOS built-in (2.6.9)
RSYNC="/usr/local/bin/rsync"
if [ ! -x "$RSYNC" ]; then
  RSYNC="rsync"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_HOST="${TENSORZERO_DEPLOY_REMOTE_HOST:-tensorzero}"
REMOTE_BASE="${TENSORZERO_DEPLOY_REMOTE_BASE:-/home/ubuntu}"

# Run from repo (./ci/deploy.sh): use repo root, first arg is mode.
if [ -f "${SCRIPT_DIR}/../Cargo.toml" ] && [ "$(basename "$SCRIPT_DIR")" = "ci" ]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
  LOCAL_DIR="$REPO_ROOT"
  REMOTE_DIR="${TENSORZERO_DEPLOY_REMOTE_DIR:-$REMOTE_BASE/$(basename "$REPO_ROOT")}"
  STATE_FILE="${REPO_ROOT}/.active_workspace"
  WORKSPACE_NUM="1"
  MODE="${1:-check}"
  E2E_FILTER="${2:-}"
else
  STATE_FILE="${SCRIPT_DIR}/.active_workspace"
  WORKSPACE_NUM="${1:-0}"
  MODE="${2:-check}"
  E2E_FILTER="${3:-}"
  if [ "$WORKSPACE_NUM" = "0" ]; then
    LOCAL_DIR="${SCRIPT_DIR}/tensorzero"
    REMOTE_DIR="${REMOTE_BASE}/tensorzero"
  else
    LOCAL_DIR="${SCRIPT_DIR}/tensorzero-${WORKSPACE_NUM}"
    REMOTE_DIR="${TENSORZERO_DEPLOY_REMOTE_DIR:-$REMOTE_BASE/tensorzero-${WORKSPACE_NUM}}"
  fi
fi

REMOTE_COMPOSE="tensorzero-core/tests/e2e/docker-compose.yml"

if [ "$MODE" != "cleanup" ] && [ ! -d "$LOCAL_DIR" ]; then
  echo "Error: local directory '${LOCAL_DIR}' does not exist."
  echo ""
  echo "To create workspace ${WORKSPACE_NUM}:"
  echo "  git clone git@github.com:AntoineToussaint/tensorzero.git ${LOCAL_DIR}"
  exit 1
fi

echo "=== Workspace ${WORKSPACE_NUM} ==="
echo "  Local:  ${LOCAL_DIR}"
echo "  Remote: ${REMOTE_HOST}:${REMOTE_DIR}"
echo "  Mode:   ${MODE}"
echo "  Rsync:  $($RSYNC --version | head -1)"
echo ""

# ── Format locally (fast, text-only) ─────────────────────────────────────
echo "--- cargo fmt (local) ---"
(cd "$LOCAL_DIR" && cargo fmt)

# ── Rsync to remote ──────────────────────────────────────────────────────
echo ""
echo "--- rsync to ${REMOTE_HOST}:${REMOTE_DIR} ---"
ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}"

# For check mode we only need Rust workspace files.
# For e2e mode we also need Docker build contexts (ui/fixtures, etc.).
RSYNC_INCLUDES=(
  --include='Cargo.toml'
  --include='Cargo.lock'
  --include='clippy.toml'
  --include='deny.toml'
  --include='.cargo/***'
  --include='.config/***'
  --include='.sqlx/***'
  --include='.dockerignore'
  --include='tensorzero-core/***'
  --include='gateway/***'
  --include='internal/***'
  --include='clients/***'
  --include='provider-proxy/***'
  --include='evaluations/***'
  --include='tensorzero-optimizers/***'
)

if [ "$MODE" = "e2e" ] || [ "$MODE" = "gateway" ]; then
  # Extra dirs needed for Docker image builds on the remote (gateway Dockerfile expects .git)
  RSYNC_INCLUDES+=(
    --include='.git/***'
    --include='ui/fixtures/***'
    --include='ci/provider-proxy-cache/***'
  )
fi

"$RSYNC" -avz --checksum --delete \
  "${RSYNC_INCLUDES[@]}" \
  --exclude='**/target/' \
  --exclude='**/node_modules/' \
  --exclude='**/__pycache__/' \
  --exclude='**/.venv/' \
  --exclude='*' \
  "${LOCAL_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"

if [ "$MODE" = "sync" ]; then
  echo ""
  echo "=== Sync complete (skipping checks) ==="
  exit 0
fi

# ── E2E mode: docker + gateway + tests, all on remote ────────────────────
if [ "$MODE" = "e2e" ]; then

  # ── Workspace change detection: reset remote docker if needed ──────────
  PREV_WORKSPACE=""
  if [ -f "$STATE_FILE" ]; then
    PREV_WORKSPACE=$(cat "$STATE_FILE")
  fi

  if [ "$PREV_WORKSPACE" != "$WORKSPACE_NUM" ] && [ -n "$PREV_WORKSPACE" ]; then
    echo "=== Workspace changed (${PREV_WORKSPACE} -> ${WORKSPACE_NUM}) ==="

    # Determine the previous remote directory
    if [ "$PREV_WORKSPACE" = "0" ]; then
      PREV_REMOTE_DIR="${REMOTE_BASE}/tensorzero"
    else
      PREV_REMOTE_DIR="${REMOTE_BASE}/tensorzero-${PREV_WORKSPACE}"
    fi

    echo "--- docker compose down -v on remote (workspace ${PREV_WORKSPACE}) ---"
    ssh "${REMOTE_HOST}" bash -l <<TEARDOWN_EOF
      cd "${PREV_REMOTE_DIR}" 2>/dev/null && \
        docker compose -f "${REMOTE_COMPOSE}" down -v 2>/dev/null || true
TEARDOWN_EOF
  fi

  echo "$WORKSPACE_NUM" > "$STATE_FILE"

  echo ""
  echo "=== Running E2E tests on ${REMOTE_HOST}:${REMOTE_DIR} ==="

  # Determine the test command
  if [ -n "$E2E_FILTER" ]; then
    TEST_CMD="cargo test-e2e-fast ${E2E_FILTER}"
  else
    TEST_CMD="cargo test-e2e-fast"
  fi

  ssh "${REMOTE_HOST}" bash -l <<REMOTE_EOF
    set -euo pipefail
    cd "${REMOTE_DIR}"
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    export TENSORZERO_GATEWAY_BUILD_CACHE="${REMOTE_DIR}/.docker-gateway-cache"
    mkdir -p "\$TENSORZERO_GATEWAY_BUILD_CACHE"

    # Build gateway image only when missing (reuse existing compose setup; serial build avoids cache corruption)
    if ! docker image inspect tensorzero/gateway:latest &>/dev/null; then
      echo "--- docker compose build gateway (once) ---"
      docker compose -f ${REMOTE_COMPOSE} build gateway-clickhouse-migrations
    else
      echo "--- reusing existing gateway image ---"
    fi
    echo "--- docker compose up (remote) ---"
    docker compose -f ${REMOTE_COMPOSE} up -d --wait
    echo "  Docker services ready"

    export TENSORZERO_CLICKHOUSE_URL="http://chuser:chpassword@localhost:8123/tensorzero_e2e_tests"
    export TENSORZERO_POSTGRES_URL="postgres://postgres:postgres@localhost:5432/tensorzero-e2e-tests"
    export DATABASE_URL="postgres://postgres:postgres@localhost:5432/tensorzero-e2e-tests"

    # ── Build the gateway ──────────────────────────────────────────────────
    echo ""
    echo "--- cargo build-e2e ---"
    cargo build-e2e

    # ── Start the gateway ──────────────────────────────────────────────────
    echo ""
    echo "--- Starting gateway ---"
    ./target/debug/gateway --config-file 'tensorzero-core/tests/e2e/config/tensorzero.*.toml' > /tmp/e2e_gateway.log 2>&1 &
    GATEWAY_PID=\$!

    # Wait for gateway to become healthy
    count=0
    max_attempts=30
    while ! curl -sf http://localhost:3000/health >/dev/null 2>&1; do
      echo "  Waiting for gateway to be healthy... (\$count/\$max_attempts)"
      sleep 2
      count=\$((count + 1))
      if [ \$count -ge \$max_attempts ]; then
        echo "  Gateway failed to become healthy after \$max_attempts attempts"
        echo "  Last 30 lines of gateway log:"
        tail -30 /tmp/e2e_gateway.log
        kill \$GATEWAY_PID 2>/dev/null || true
        exit 1
      fi
    done
    echo "  Gateway is healthy (PID \$GATEWAY_PID)"

    # ── Run ClickHouse-specific tests first (lighter) ──────────────────────
    echo ""
    echo "--- cargo test-clickhouse-fast ---"
    cargo test-clickhouse-fast || true

    # ── Run the main E2E tests ─────────────────────────────────────────────
    echo ""
    echo "--- ${TEST_CMD} ---"
    ${TEST_CMD}
    TEST_EXIT=\$?

    # ── Stop the gateway ───────────────────────────────────────────────────
    echo ""
    echo "--- Stopping gateway (PID \$GATEWAY_PID) ---"
    kill \$GATEWAY_PID 2>/dev/null || true
    wait \$GATEWAY_PID 2>/dev/null || true

    if [ \$TEST_EXIT -ne 0 ]; then
      echo ""
      echo "  Last 50 lines of gateway log:"
      tail -50 /tmp/e2e_gateway.log
      exit \$TEST_EXIT
    fi

    echo ""
    echo "=== E2E tests passed (workspace ${WORKSPACE_NUM}) ==="
REMOTE_EOF

  exit $?
fi

# ── Gateway mode: docker + gateway on remote, gateway stays running ───────
# Use this when you want to run the UI locally and only tunnel the gateway.
# After this exits, run ./scripts/tunnel-and-ui.sh (tunnel in one terminal, UI in another).
#
# To use real OpenAI (not dummy): set OPENAI_API_KEY before running, or put it in
# scripts/.env (e.g. export OPENAI_API_KEY=sk-...). It will be passed to the remote
# so the gateway can call OpenAI. In the UI, use function basic_test, variant "openai".
if [ "$MODE" = "gateway" ]; then

  # Load OPENAI_API_KEY from scripts/.env: try workspace (LOCAL_DIR) first, then dir next to deploy.sh
  if [ -f "${LOCAL_DIR}/scripts/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "${LOCAL_DIR}/scripts/.env"
    set +a
  fi
  if [ -z "${OPENAI_API_KEY:-}" ] && [ -f "${SCRIPT_DIR}/scripts/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/scripts/.env"
    set +a
  fi
  if [ -n "${OPENAI_API_KEY:-}" ]; then
    echo "  OPENAI_API_KEY loaded for remote gateway (from scripts/.env or env)"
  else
    echo "  WARNING: OPENAI_API_KEY not set. Put it in ${LOCAL_DIR}/scripts/.env (e.g. OPENAI_API_KEY=sk-...) and re-run deploy so the gateway can call real OpenAI."
  fi

  # ── Workspace change detection: reset remote docker if needed ──────────
  PREV_WORKSPACE=""
  if [ -f "$STATE_FILE" ]; then
    PREV_WORKSPACE=$(cat "$STATE_FILE")
  fi

  if [ "$PREV_WORKSPACE" != "$WORKSPACE_NUM" ] && [ -n "$PREV_WORKSPACE" ]; then
    echo "=== Workspace changed (${PREV_WORKSPACE} -> ${WORKSPACE_NUM}) ==="
    if [ "$PREV_WORKSPACE" = "0" ]; then
      PREV_REMOTE_DIR="${REMOTE_BASE}/tensorzero"
    else
      PREV_REMOTE_DIR="${REMOTE_BASE}/tensorzero-${PREV_WORKSPACE}"
    fi
    echo "--- docker compose down -v on remote (workspace ${PREV_WORKSPACE}) ---"
    ssh "${REMOTE_HOST}" bash -l <<TEARDOWN_EOF
      cd "${PREV_REMOTE_DIR}" 2>/dev/null && \
        docker compose -f "${REMOTE_COMPOSE}" down -v 2>/dev/null || true
TEARDOWN_EOF
  fi

  echo "$WORKSPACE_NUM" > "$STATE_FILE"

  # Pass OPENAI_API_KEY to remote securely (temp file, then rm) so gateway can use real OpenAI
  REMOTE_ENV_FILE=""
  if [ -n "${OPENAI_API_KEY:-}" ]; then
    TMP_KEY_FILE=$(mktemp)
    echo "${OPENAI_API_KEY}" > "$TMP_KEY_FILE"
    REMOTE_ENV_FILE="/tmp/tz_gateway_env_$$"
    scp -q "$TMP_KEY_FILE" "${REMOTE_HOST}:${REMOTE_ENV_FILE}"
    rm -f "$TMP_KEY_FILE"
  fi

  echo ""
  echo "=== Starting Docker + gateway on ${REMOTE_HOST}:${REMOTE_DIR} (gateway will stay running) ==="

  # Pass rebuild flag: 1 if user said "rebuild" or "build", else 0 (reuse existing image when possible)
  REBUILD_GATEWAY=0
  [ "${3:-}" = "rebuild" ] || [ "${3:-}" = "build" ] && REBUILD_GATEWAY=1

  ssh "${REMOTE_HOST}" bash -l <<REMOTE_EOF
    set -euo pipefail
    cd "${REMOTE_DIR}"

    # Export OPENAI_API_KEY on remote if we passed it (for real OpenAI, not dummy)
    if [ -n "${REMOTE_ENV_FILE:-}" ] && [ -f "${REMOTE_ENV_FILE}" ]; then
      export OPENAI_API_KEY=\$(cat "${REMOTE_ENV_FILE}")
      rm -f "${REMOTE_ENV_FILE}"
    fi

    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    export TENSORZERO_GATEWAY_BUILD_CACHE="${REMOTE_DIR}/.docker-gateway-cache"
    mkdir -p "\$TENSORZERO_GATEWAY_BUILD_CACHE"

    # Build gateway image only when missing or when user asked for rebuild (avoids tearing down / full rebuild every time)
    need_build=${REBUILD_GATEWAY}
    if [ "\$need_build" -eq 0 ] && ! docker image inspect tensorzero/gateway:latest &>/dev/null; then
      need_build=1
    fi
    if [ "\$need_build" -eq 1 ]; then
      echo "--- docker compose build gateway (once) ---"
      docker compose -f ${REMOTE_COMPOSE} build gateway-clickhouse-migrations
    else
      echo "--- reusing existing gateway image ---"
    fi
    echo "--- docker compose up (remote) ---"
    docker compose -f ${REMOTE_COMPOSE} up -d --wait
    echo "  Docker services ready"

    export TENSORZERO_CLICKHOUSE_URL="http://chuser:chpassword@localhost:8123/tensorzero_e2e_tests"
    export TENSORZERO_POSTGRES_URL="postgres://postgres:postgres@localhost:5432/tensorzero-e2e-tests"
    export DATABASE_URL="postgres://postgres:postgres@localhost:5432/tensorzero-e2e-tests"

    echo ""
    echo "--- cargo build-e2e ---"
    cargo build-e2e

    # Kill any existing gateway on 3000 so we can start fresh
    pkill -f "gateway --config-file" 2>/dev/null || true
    sleep 2

    echo ""
    echo "--- Starting gateway (stays running in background) ---"
    # Run the binary directly with the config glob in single quotes so the shell does not expand it (gateway accepts one glob argument).
    nohup ./target/debug/gateway --config-file 'tensorzero-core/tests/e2e/config/tensorzero.*.toml' > /tmp/e2e_gateway.log 2>&1 &
    GATEWAY_PID=\$!
    echo \$GATEWAY_PID > /tmp/e2e_gateway.pid

    count=0
    max_attempts=30
    while ! curl -sf http://localhost:3000/health >/dev/null 2>&1; do
      echo "  Waiting for gateway to be healthy... (\$count/\$max_attempts)"
      sleep 2
      count=\$((count + 1))
      if [ \$count -ge \$max_attempts ]; then
        echo "  Gateway failed to become healthy"
        tail -30 /tmp/e2e_gateway.log
        exit 1
      fi
    done
    echo "  Gateway is healthy (PID \$GATEWAY_PID, log: /tmp/e2e_gateway.log)"
    echo "  To stop later: ssh ${REMOTE_HOST} 'kill \$(cat /tmp/e2e_gateway.pid 2>/dev/null) 2>/dev/null || true'"
REMOTE_EOF

  echo ""
  echo "=== Remote gateway is running. Starting tunnel (localhost:3000 -> ${REMOTE_HOST}:3000) ==="
  echo "  Gateway logs (in another terminal):  ssh ${REMOTE_HOST} 'tail -f /tmp/e2e_gateway.log'"
  echo "  Run the UI:  cd ${LOCAL_DIR}/ui && TENSORZERO_GATEWAY_URL=http://localhost:3000 pnpm dev"
  echo "  To use real OpenAI: pick function basic_test, variant 'openai'. Ctrl+C here stops the tunnel."
  echo ""
  ssh -L 3000:localhost:3000 "${REMOTE_HOST}" -N
fi

# ── Run only what was requested (check / clippy / test / all) ─────────────
run_remote_check() {
  ssh "${REMOTE_HOST}" bash -l -c "set -euo pipefail; cd ${REMOTE_DIR} && cargo check --all-targets --all-features"
}
run_remote_clippy() {
  ssh "${REMOTE_HOST}" bash -l -c "set -euo pipefail; cd ${REMOTE_DIR} && cargo clippy --all-targets --all-features -- -D warnings"
}
run_remote_test() {
  ssh "${REMOTE_HOST}" bash -l -c "set -euo pipefail; cd ${REMOTE_DIR} && cargo test-unit-fast"
}

case "$MODE" in
  check)
    echo ""
    echo "=== cargo check on ${REMOTE_HOST}:${REMOTE_DIR} ==="
    run_remote_check
    echo ""
    echo "=== Check passed (workspace ${WORKSPACE_NUM}) ==="
    ;;
  clippy)
    echo ""
    echo "=== cargo clippy on ${REMOTE_HOST}:${REMOTE_DIR} ==="
    run_remote_clippy
    echo ""
    echo "=== Clippy passed (workspace ${WORKSPACE_NUM}) ==="
    ;;
  test)
    echo ""
    echo "=== unit tests on ${REMOTE_HOST}:${REMOTE_DIR} ==="
    run_remote_test
    echo ""
    echo "=== Tests passed (workspace ${WORKSPACE_NUM}) ==="
    ;;
  all)
    echo ""
    echo "=== check + clippy + test on ${REMOTE_HOST}:${REMOTE_DIR} ==="
    run_remote_check
    echo ""
    run_remote_clippy
    echo ""
    run_remote_test
    echo ""
    echo "=== All passed (workspace ${WORKSPACE_NUM}) ==="
    ;;
  *)
    echo "Error: unknown mode '${MODE}'. Use: sync, check, clippy, test, all, e2e, gateway"
    exit 1
    ;;
esac
