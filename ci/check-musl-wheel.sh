#!/usr/bin/env bash
set -euxo pipefail

docker run -v $(pwd):/wheels python:3.10-alpine /bin/sh -c "
    pip install /wheels/$1
    python -c 'import tensorzero; print(tensorzero.tensorzero.__file__)'
"
