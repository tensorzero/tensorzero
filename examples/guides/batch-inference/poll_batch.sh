#!/bin/bash

# Check if `batch_id` argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <batch_id>"
    exit 1
fi

BATCH_ID=$1
GATEWAY_URL="http://localhost:3000"

# Poll batch inference endpoint with `batch_id`
curl -X GET "${GATEWAY_URL}/batch_inference/${BATCH_ID}"
