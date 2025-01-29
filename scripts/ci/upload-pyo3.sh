#!/usr/bin/env bash
set -euxo pipefail

cd "$(dirname "$0")"
curl -LsSf https://astral.sh/uv/0.5.24/install.sh | sh
cd ../../clients/python-pyo3
uv venv
uv pip sync requirements.txt
uv run maturin upload --repository testpypi --non-interactive --skip-existing wheels-*/*
