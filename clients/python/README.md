# TensorZero Python Client

**[Website](https://www.tensorzero.com/)** ·
**[Docs](https://www.tensorzero.com/docs)** ·
**[Twitter](https://www.x.com/tensorzero)** ·
**[Slack](https://www.tensorzero.com/slack)** ·
**[Discord](https://www.tensorzero.com/discord)**

**[Quick Start (5min)](https://www.tensorzero.com/docs/quickstart)** ·
**[Comprehensive Tutorial](https://www.tensorzero.com/docs/gateway/tutorial)** ·
**[Deployment Guide](https://www.tensorzero.com/docs/gateway/deployment)** ·
**[API Reference](https://www.tensorzero.com/docs/gateway/api-reference/inference)** ·
**[Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference)**

The `tensorzero` package provides a Python client for the TensorZero Gateway.
This client allows you to easily make inference requests and assign feedback to them via the gateway.

See our **[API Reference](https://www.tensorzero.com/docs/gateway/api-reference)** for more information.

## Installation

```bash
pip install tensorzero
```

## Basic Usage

### Initialization

The TensorZero client offers synchronous (`TensorZeroGateway`) and asynchronous (`AsyncTensorZeroGateway`) variants.
Additionally, the client can launch an embedded (in-memory) gateway (`build_embedded`) or connect to an external HTTP gateway (`build_http`) - both of these methods return a gateway instance.

By default, the asynchronous client returns a `Future` when you call `build_http` or `build_embedded`, so you must `await` it.
If you prefer to avoid the `await`, you can set `async_setup=False` to initialize the client in a blocking way.

#### Synchronous HTTP Gateway

```python
from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    # ...
```

#### Asynchronous HTTP Gateway

```python
import asyncio

from tensorzero import AsyncTensorZeroGateway


async def run():
    async with await AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000",
        # async_setup=False  # optional: skip the `await` and run `build_http` synchronously (blocking)
    ) as client:
        # ...


if __name__ == "__main__":
    asyncio.run(run())
```

#### Synchronous Embedded Gateway

```python
from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_embedded(
    config_file="/path/to/tensorzero.toml",
    clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero"
) as client:
    # ...
```

#### Asynchronous Embedded Gateway

```python
import asyncio

from tensorzero import AsyncTensorZeroGateway


async def run():
    async with await AsyncTensorZeroGateway.build_embedded(
        config_file="/path/to/tensorzero.toml",
        clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero"
        # async_setup=False  # optional: skip the `await` and run `build_embedded` synchronously (blocking)
    ) as client:
        # ...


if __name__ == "__main__":
    asyncio.run(run())
```

### Inference

#### Non-Streaming Inference with Synchronous Client

```python
with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    response = client.inference(
        model_name="openai::gpt-4o-mini",
        input={
            "messages": [
                {"role": "user", "content": "What is the capital of Japan?"},
            ],
        },
    )

    print(response)
```

#### Non-Streaming Inference with Asynchronous Client

```python
async with await AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    response = await client.inference(
        model_name="openai::gpt-4o-mini",
        input={
            "messages": [
                {"role": "user", "content": "What is the capital of Japan?"},
            ],
        },
    )

    print(response)
```

#### Streaming Inference with Synchronous Client

```python
with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    stream = client.inference(
        model_name="openai::gpt-4o-mini",
        input={
            "messages": [
                {"role": "user", "content": "What is the capital of Japan?"},
            ],
        },
        stream=True,
    )

    for chunk in stream:
        print(chunk)
```

#### Streaming Inference with Asynchronous Client

```python
async with await AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    stream = await client.inference(
        model_name="openai::gpt-4o-mini",
        input={
            "messages": [{"role": "user", "content": "What is the capital of Japan?"}],
        },
        stream=True,
    )

    async for chunk in stream:
        print(chunk)
```

### Feedback

#### Synchronous

```python
with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    response = client.feedback(
        metric_name="thumbs_up",
        inference_id="00000000-0000-0000-0000-000000000000",
        value=True,  # 👍
    )

    print(response)
```

#### Asynchronous

```python
async with await AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    response = await client.feedback(
        metric_name="thumbs_up",
        inference_id="00000000-0000-0000-0000-000000000000",
        value=True,  # 👍
    )

    print(response)
```
