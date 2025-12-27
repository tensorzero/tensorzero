#!/bin/bash

# Create a batch inference job with three inputs
curl -X POST http://localhost:3000/batch_inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "generate_haiku",
    "variant_name": "gpt_4o_mini",
    "inputs": [
      {
        "messages": [
          {
            "role": "user",
            "content": "Write a haiku about TensorZero."
          }
        ]
      },
      {
        "messages": [
          {
            "role": "user",
            "content": "Write a haiku about general aviation."
          }
        ]
      },
      {
        "messages": [
          {
            "role": "user",
            "content": "Write a haiku about anime."
          }
        ]
      }
    ]
  }'
