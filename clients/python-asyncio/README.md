# TensorZero Python Client

This is an async Python client for the TensorZero inference API. Check out the [docs](https://tensorzero.com/docs/) for more information. This client allows you to easily make inference requests and assign feedback to them via the TensorZero Gateway.

## Installation

```bash
pip install tensorzero
```

## Basic Usage

### Non-Streaming Inference

```python
from tensorzero import AsyncTensorZero

with AsyncTensorZero("http://localhost:3000") as client:
    result = await client.inference(
        function_name="basic_test",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
episode_id = result.episode_id
output = result.output
print(output[0].text)  # Will return the text of the first content block returned by TensorZero
```

### Streaming Inference

```python
from tensorzero import AsyncTensorZero

async with AsyncTensorZero() as client:
    response = await client.chat.completions.create(
        model="firefunction-v2",
        messages=[{"role": "user", "content": "Hello, world!"}],
    )
    print(response)
```

## Development Setup

- Install `uv` [→](https://docs.astral.sh/uv/getting-started/installation/)
- Install `ruff` with `uv pip install ruff`.

You should now be able to run tests with `uv run pytest` and use pre-commit hooks which depend on `ruff`.
