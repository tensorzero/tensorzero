# Guide: Multimodal Inference

This directory contains the code for the **[Multimodal Inference](https://www.tensorzero.com/docs/gateway/guides/multimodal-inference)** guide.

## Running the Example

1. Set the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY="sk-..." # Replace with your OpenAI API key
```

2. Launch the TensorZero Gateway, ClickHouse, and MinIO (a local S3-compatible object storage service):

```bash
docker compose up
```

> [!TIP]
>
> You can use any S3-compatible object storage service (e.g. AWS S3, GCP Storage, Cloudflare R2).
> We use a local MinIO instance in this example for convenience.

3. Install the Python dependencies. We recommend using [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync
```

4. Run the example:

```bash
uv run main.py
```
