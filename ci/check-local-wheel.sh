#!/usr/bin/env bash
set -euxo pipefail

uv venv .test-venv
source .test-venv/bin/activate
uv pip install $1
python -c 'import tensorzero; print(tensorzero.tensorzero.__file__)'