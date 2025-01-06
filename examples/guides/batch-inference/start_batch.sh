#!/bin/bash

# Create the JSON payload
curl -X POST http://localhost:3000/batch_inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "generate_haiku",
    "variant_name": "gpt_4o_mini",
    "inputs": [{
      "messages": [
        {
          "role": "user",
          "content": "Write a haiku about artificial intelligence."
        }
      ]
    }]
  }'
