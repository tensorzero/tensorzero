#! /bin/bash

curl "http://localhost:8080/invocations" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gemma3:1b",
        "messages": [
            {
                "role": "user",
                "content": "Write a one-sentence bedtime story about a unicorn."
            }
        ], "temperature": -100, "frequency_penalty": -100
    }'
