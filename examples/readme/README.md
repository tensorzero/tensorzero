# TensorZero `README` Examples

This directory contains the code used for the absolutely minimal examples in the README at the repository root.

## Running the Example

1. Launch the TensorZero Gateway and ClickHouse database:

```bash
docker compose up
```

2. Install the Python dependencies. We recommend using [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync
```

3. Run the examples:

```bash
uv run tensorzero_sync_client.py
uv run tensorzero_async_client.py
uv run openai_client.py
```
