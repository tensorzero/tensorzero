#!/bin/bash

set -euxo pipefail
# Set this to an intentionally invalid url, to make sure that `tensorzero` can still be imported
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=unix:///bad-tensorzero.sock
uv run tests/import_failure.py
# Use pre-built extension instead of rebuilding
uv run pytest -n auto --reruns 3 $@