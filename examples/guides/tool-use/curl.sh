#!/bin/bash

# First request: the model decides to call a tool
curl -X POST http://localhost:3000/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tensorzero::function_name::weather_chatbot",
    "messages": [
      {"role": "user", "content": "What is the weather in Tokyo (°F)?"}
    ]
  }'

echo "\n"

# Second request: provide the tool result and get the final response
curl -X POST http://localhost:3000/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tensorzero::function_name::weather_chatbot",
    "messages": [
      {
        "role": "user",
        "content": "What is the weather in Tokyo (°F)?"
      },
      {
        "role": "assistant",
        "tool_calls": [
          {
            "id": "123",
            "type": "function",
            "function": {
              "name": "get_temperature",
              "arguments": "{\"location\": \"Tokyo\"}"
            }
          }
        ]
      },
      {
        "role": "tool",
        "tool_call_id": "123",
        "content": "70"
      }
    ]
  }'
