# TensorZero Python Client

This is an async Python client for the TensorZero gateway. Check out the [docs](https://tensorzero.com/docs/) for more information. This client allows you to easily make inference requests and assign feedback to them via the TensorZero gateway.

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
print(output[0].text)  # Prints the text of the first content block returned by TensorZero
```

### Streaming Inference

```python
from tensorzero import AsyncTensorZero

async with AsyncTensorZero() as client:
    stream = await client.chat.completions.create(
        function_name="basic_test",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
        stream=True,
    )
    async for chunk in stream:
        print(chunk.content[0].text)  # Prints the text in each chunk returned by TensorZero
```
