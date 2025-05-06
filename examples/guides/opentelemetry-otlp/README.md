# Example: Exporting OpenTelemetry (OTLP) Traces from the TensorZero Gateway

```bash
curl -X POST "http://localhost:3000/inference" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "openai::gpt-4o-mini",
    "input": {
      "messages": [
        {
          "role": "user",
          "content": "Write a haiku about artificial intelligence."
        }
      ]
    }
  }'
```

```bash
curl -X POST "http://localhost:3000/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "metric_name": "comment",
    "episode_id": "00000000-0000-0000-0000-000000000000",
    "value": "Great haiku!"
  }'
```
