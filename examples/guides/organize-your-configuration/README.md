```bash
curl -X POST "http://localhost:3000/inference" \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "functionA",
    "input": {
      "messages": [
        {
          "role": "user",
          "content": "Share a fun fact about artificial intelligence."
        }
      ]
    }
  }'
```
