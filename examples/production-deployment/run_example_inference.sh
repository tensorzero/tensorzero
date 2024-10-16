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
