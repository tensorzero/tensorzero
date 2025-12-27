#!/bin/bash

curl -X POST "http://localhost:3000/inference" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "openai::gpt-5-mini",
    "input": {
      "messages": [
        {
          "role": "user",
          "content": "Write a haiku about TensorZero."
        }
      ]
    }
  }'
