# Guide: Episodes

This directory contains the code for the **[Episodes](https://www.tensorzero.com/docs/gateway/guides/episodes)** guide.

## Running the Example

1. Install the Python dependencies. We recommend using [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync
```

2. Launch the TensorZero Gateway and ClickHouse database:

```bash
docker compose up
```

3. Run the example:

```bash
uv run python run.py
```
