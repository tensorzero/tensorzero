#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

echo 'POST http://localhost:3000/inference' \
| vegeta attack \
    -header="Content-Type: application/json" \
    -header="Authorization: Bearer budserve_ApTuiKIpZEjMytHt3nmCJqBQGaIlqc2TfoKYFusk" \
    -body=$SCRIPT_DIR/body.json \
    -duration=30s \
    -rate=1000 \
    -timeout=1s \
| vegeta report
