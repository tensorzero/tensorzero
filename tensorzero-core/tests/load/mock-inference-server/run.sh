#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

echo 'POST http://localhost:3030/openai/chat/completions' \
| vegeta attack \
    -header="Content-Type: application/json" \
    -body=$SCRIPT_DIR/body.json \
    -duration=30s \
    -rate=1000 \
    -timeout=1s \
| vegeta report
