#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

echo 'POST http://localhost:4000/chat/completions' \
| vegeta attack \
    -header="Content-Type: application/json" \
    -body=$SCRIPT_DIR/body.json \
    -duration=30s \
    -rate=300 \
    -timeout=1s \
| vegeta report
