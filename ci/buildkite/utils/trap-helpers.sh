#!/usr/bin/env bash

# NOTE: This file is intended to be sourced by CI scripts.
# Do not set -euo pipefail here to avoid altering caller shell options.

# tz_cleanup: common cleanup routine for CI scripts
# - Prints a clear section header
# - If TZ_COMPOSE_LOGS_FILE is set, prints docker compose logs for that file
tz_cleanup() {
  echo "==============================================================================="
  echo "Running cleanup and debug steps..."
  echo "==============================================================================="

  if [[ -n "${TZ_COMPOSE_LOGS_FILE:-}" ]]; then
    echo "Printing Docker Compose logs..."
    docker compose -f "$TZ_COMPOSE_LOGS_FILE" logs -t || true
  fi
}

# tz_setup_compose_logs_trap <compose_file>
# Registers an EXIT trap that calls tz_cleanup and prints logs for the compose file.
tz_setup_compose_logs_trap() {
  local compose_file="$1"
  export TZ_COMPOSE_LOGS_FILE="$compose_file"
  trap tz_cleanup EXIT
}
