#!/bin/bash

: "${OPENAI_API_KEY:?Environment variable OPENAI_API_KEY must be set.}"

curl -X POST "http://localhost:3000/openai/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tensorzero::model_name::openai::gpt-5-mini",
    "messages": [
      {
        "role": "user",
        "content": "Write a haiku about TensorZero."
      }
    ],
    "tensorzero::credentials": {
      "openai_api_key": "'"${OPENAI_API_KEY}"'"
    }
  }'
