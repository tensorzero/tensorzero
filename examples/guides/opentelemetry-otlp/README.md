# Example: Exporting OpenTelemetry (OTLP) Traces from the TensorZero Gateway

This example shows how to export traces from the TensorZero Gateway to an external OpenTelemetry-compatible observability system.

Here, we'll export traces to a local instance of Jaeger.

## Getting Started

### Setup

1. Install Docker.
2. Generate an API key for OpenAI (`OPENAI_API_KEY`).
3. Set the `OPENAI_API_KEY` environment variable.
4. Launch the TensorZero Gateway, ClickHouse, and Jaeger: `docker compose up`

### Running the Example

First, let's make an inference request to the gateway.

```bash
curl -X POST "http://localhost:3000/inference" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "openai::gpt-4o-mini",
    "input": {
      "messages": [
        {
          "role": "user",
          "content": "Write a haiku about TensorZero."
        }
      ]
    }
  }'
```

Then, let's make a feedback request to the gateway.

> [!IMPORTANT]
>
> Make sure to replace the `episode_id` with the actual episode ID from the inference request above (not the inference ID!).

```bash
curl -X POST "http://localhost:3000/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "metric_name": "comment",
    "episode_id": "00000000-0000-0000-0000-000000000000",
    "value": "Great haiku!"
  }'
```

Finally, visit the Jaeger UI at `http://localhost:16686` to see the traces.

![Screenshot of TensorZero Gateway traces in Jaeger](https://github.com/user-attachments/assets/4f168a98-28a4-4721-95d3-acabfc156866)
