# TensorZero Python Client

**[Website](https://www.tensorzero.com/)** ·
**[Docs](https://www.tensorzero.com/docs)** ·
**[Twitter](https://www.x.com/tensorzero)** ·
**[Slack](https://www.tensorzero.com/slack)** ·
**[Discord](https://www.tensorzero.com/discord)**

**[Quick Start (5min)](https://www.tensorzero.com/docs/gateway/tutorial)** ·
**[Comprehensive Tutorial](https://www.tensorzero.com/docs/gateway/tutorial)** ·
**[Deployment Guide](https://www.tensorzero.com/docs/gateway/deployment)** ·
**[API Reference](https://www.tensorzero.com/docs/gateway/api-reference)** ·
**[Configuration Reference](https://www.tensorzero.com/docs/gateway/deployment)**

The `tensorzero` package provides an async Python client for the TensorZero Gateway.
This client allows you to easily make inference requests and assign feedback to them via the gateway.

See our **[API Reference](https://www.tensorzero.com/docs/gateway/api-reference)** for more information.

## Installation

```bash
pip install tensorzero
```

## Basic Usage

### Non-Streaming Inference

```python
import asyncio

from tensorzero import AsyncTensorZeroGateway


async def run(topic):
    async with AsyncTensorZeroGateway("http://localhost:3000") as client:
        result = await client.inference(
            function_name="generate_haiku",
            input={
                "messages": [
                    {"role": "user", "content": {"topic": topic}},
                ],
            },
        )

        print(result)


if __name__ == "__main__":
    asyncio.run(run("artificial intelligence"))
```

### Streaming Inference

```python
import asyncio

from tensorzero import AsyncTensorZeroGateway


async def run(topic):
    async with AsyncTensorZeroGateway("http://localhost:3000") as client:
        stream = await client.inference(
            function_name="generate_haiku",
            input={
                "messages": [
                    {"role": "user", "content": {"topic": topic}},
                ],
            },
            stream=True,
        )

        async for chunk in stream:
            print(chunk)


if __name__ == "__main__":
    asyncio.run(run("artificial intelligence"))

```

### Feedback

```python
import asyncio

from tensorzero import AsyncTensorZeroGateway


async def run(inference_id):
    async with AsyncTensorZeroGateway("http://localhost:3000") as client:
        result = await client.feedback(
            metric_name="thumbs_up",
            inference_id=inference_id,
            value=True,  # 👍
        )

        print(result)


if __name__ == "__main__":
    asyncio.run(run("00000000-0000-0000-0000-000000000000"))
```
