#!/usr/bin/env bash

curl http://localhost:3000/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tensorzero::function_name::generate_haiku",
    "messages": [
      {
        "role": "user",
        "content": "Write a haiku about TensorZero."
      }
    ]
  }'
