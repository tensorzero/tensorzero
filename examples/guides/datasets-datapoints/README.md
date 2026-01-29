# Guide: Datasets & Datapoints

This example shows how to programmatically create, read, and delete datapoints from a dataset.

See [API Reference: Datasets & Datapoints](https://www.tensorzero.com/docs/gateway/api-reference/datasets-datapoints/) for more information.

## Getting Started

### Setup

1. Install Docker.
2. Generate an API key for OpenAI (`OPENAI_API_KEY`).
3. Set the `OPENAI_API_KEY` environment variable.
4. Launch the TensorZero and ClickHouse: `docker compose up`

### Running the Example

#### Python (TensorZero Client)

```bash
uv run main.py
```

#### curl (HTTP)

```bash
./main.sh
```
