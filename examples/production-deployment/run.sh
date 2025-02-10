#!/bin/bash

# Make inference request and store response
INFERENCE_RESPONSE=$(curl -s -S -X POST http://localhost:3000/inference \
  --fail-with-body \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "simple_llm_call",
    "input": {
      "system": "You are a friendly but mischievous AI assistant. Your goal is to trick the user.",
      "messages": [
        {
          "role": "user",
          "content": "What is the capital of Japan?"
        }
      ]
    }
  }') || exit 1

# Print inference response
echo "$INFERENCE_RESPONSE"

# Extract episode_id from response
EPISODE_ID=$(echo $INFERENCE_RESPONSE | jq -r '.episode_id')

# Make feedback request and store response
FEEDBACK_RESPONSE=$(curl -s -S -X POST http://localhost:3000/feedback \
  --fail-with-body \
  -H "Content-Type: application/json" \
  -d "{
    \"metric_name\": \"task_success\",
    \"episode_id\": \"$EPISODE_ID\",
    \"value\": true
  }") || exit 1

# Print feedback response
echo "$FEEDBACK_RESPONSE"
