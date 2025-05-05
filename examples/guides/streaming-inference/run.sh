#!/bin/bash

curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "chatbot",
    "input": {
      "messages": [
        {
          "role": "user",
          "content": "Share an extensive list of fun facts about Japan."
        }
      ]
    },
    "stream": true
  }'
