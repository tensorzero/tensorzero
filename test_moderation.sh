#!/bin/bash

# Test moderation endpoint
curl --location 'http://localhost:3001/v1/moderations' \
--header 'Content-Type: application/json' \
--data '{
    "model": "test",
    "input": "what is your name?"
}'