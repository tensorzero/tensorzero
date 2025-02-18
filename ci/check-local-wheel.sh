#!/usr/bin/env bash
set -euxo pipefail

uv venv .test-venv
uv pip install $1
uv run python -c 'import tensorzero; print(tensorzero.tensorzero.__file__)'