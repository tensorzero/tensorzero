#!/bin/bash

set -euxo pipefail
# Set this to an intentionally invalid url, to make sure that `tensorzero` can still be imported
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=unix:///bad-tensorzero.sock uv run --config-setting 'build-args=--profile=dev --features e2e_tests' tests/import_failure.py

# Avoid using 'uv run maturin develop', as this will build twice (once from uv when making the venv, and once from maturin)
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4317 uv run --config-setting 'build-args=--profile=dev --features e2e_tests' pytest -n auto --reruns 3 "$@"
