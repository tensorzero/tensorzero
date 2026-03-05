#!/bin/bash

curl http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "weather_chatbot",
    "input": {"messages": [{"role": "user", "content": "What is the weather in Tokyo (°F)?"}]}
  }'

echo "\n"

curl http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "weather_chatbot",
    "input": {
      "messages": [
        {
          "role": "user",
          "content": "What is the weather in Tokyo (°F)?"
        },
        {
          "role": "assistant",
          "content": [
            {
              "type": "tool_call",
              "id": "123",
              "name": "get_temperature",
              "arguments": {
                "location": "Tokyo"
              }
            }
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "tool_result",
              "id": "123",
              "name": "get_temperature",
              "result": "70"
            }
          ]
        }
      ]
    }
  }'
