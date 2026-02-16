#!/bin/bash
set -e

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --default-config)
      export TENSORZERO_UI_DEFAULT_CONFIG=1
      shift
      ;;
    --config-file)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --config-file requires a path argument"
        exit 1
      fi
      export TENSORZERO_UI_CONFIG_FILE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

cd /app/ui

# Launch React Router
exec ./node_modules/.bin/react-router-serve ./build/server/index.js
