#!/bin/bash

curl -X POST http://localhost:3000/inference \
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
  }'

curl -X POST http://localhost:3000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "metric_name": "task_success",
    "inference_id": "00000000-0000-0000-0000-000000000000",
    "value": true
  }'
