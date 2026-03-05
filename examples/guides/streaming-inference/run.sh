#!/bin/bash

curl -X POST http://localhost:3000/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tensorzero::function_name::chatbot",
    "messages": [
      {
        "role": "user",
        "content": "Share an extensive list of fun facts about Japan."
      }
    ],
    "stream": true
  }'
