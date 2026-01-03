#!/bin/bash

# 1. Make an inference with a tag
echo "1. Making an inference with tag \`my_tag=my_value\`"
INFERENCE_RESPONSE=$(curl -s -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "openai::gpt-5-mini",
    "input": {
      "messages": [{"role": "user", "content": "Write a haiku about TensorZero."}]
    },
    "tags": {"my_tag": "my_value"}
  }')

INFERENCE_ID=$(echo "$INFERENCE_RESPONSE" | jq -r '.inference_id')
echo "Completed Inference: $INFERENCE_ID"
echo


# 2. Query the inference by ID
echo "2. Querying the inference by ID"
curl -s -X POST http://localhost:3000/v1/inferences/get_inferences \
  -H "Content-Type: application/json" \
  -d "{
    \"ids\": [\"$INFERENCE_ID\"],
    \"output_source\": \"inference\"
  }" | jq '.inferences | length | "Retrieved \(.) inference(s) by ID"'
echo


# 3. List inferences filtered by the tag
echo "3. Listing inferences filtered by tag \`my_tag=my_value\`"
curl -s -X POST http://localhost:3000/v1/inferences/list_inferences \
  -H "Content-Type: application/json" \
  -d '{
    "output_source": "inference",
    "filters": {
      "type": "tag",
      "key": "my_tag",
      "value": "my_value",
      "comparison_operator": "="
    },
    "limit": 10
  }' | jq '.inferences | length | "Found \(.) inference(s) with tag `my_tag=my_value`"'
echo
