#!/bin/bash

set -euxo pipefail
# Avoid using 'uv run maturin develop', as this will build twice (once from uv when making the venv, and once from maturin)
uv run --config-setting 'build-args=--profile=dev --features e2e_tests' pytest -n auto $@
