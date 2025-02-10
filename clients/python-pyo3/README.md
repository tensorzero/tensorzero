# Rust-based TensorZero Python Client

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

This is the future version of the Python client, which includes the Rust client as a native library

The `tensorzero` package provides an async Python client for the TensorZero Gateway.
This client allows you to easily make inference requests and assign feedback to them via the gateway.

See our **[API Reference](https://www.tensorzero.com/docs/gateway/api-reference)** for more information.

## Installation

This client is currently not published on `PyPI` - for production code, you most likely want to
use the Python client in `<repository root>/clients/python`

To try out the new Rust-based client implementation, run:

```bash
pip install -r requirements.txt
maturin develop
```

If using `uv`, then instead run:

```bash
uv venv
uv pip sync requirements.txt
uv run maturin develop --uv
uv run python
```

## Basic Usage

### Initializing the client

The TensorZero client comes in both synchronous (`TensorZeroGateway`) and asynchronous (`AsyncTensorZeroGateway`) variants.
Each of these classes can be constructed in one of two ways:

- HTTP gateway mode. This constructs a client that makes requests to the specified TensorZero HTTP gateway:

```python
from tensorzero import TensorZeroGateway, AsyncTensorZeroGateway

client = TensorZeroGateway(base_url="http://localhost:3000")
async_client = AsyncTensorZeroGateway(base_url="http://localhost:3000")
```

- Embedded gateway mode. This starts an in-memory TensorZero gateway using the provided config file and Clickhouse url
  Note that `AsyncTensorZeroGateway.create_embedded_gateway` returns a `Future`, which you must `await` to get the client.

```python
from tensorzero import TensorZeroGateway, AsyncTensorZeroGateway

client = TensorZeroGateway.create_embedded_gateway(config_path="/path/to/tensorzero.toml", clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero-python-e2e")
async_client = await AsyncTensorZeroGateway.create_embedded_gateway(config_path="/path/to/tensorzero.toml", clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero-python-e2e")
```

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

## Note on naming

There are several different names in use in this client:

- `python-pyo3` - this is _only_ used as the name of the top-level directory, to distinguish it from the pure-python implementation
  In the future, we'll delete the pure-python client and rename this to 'python'
- `tensorzero_python` - this is the rust _crate_ name, so that we get sensible output from running Cargo
- `tensorzero` - this is the name of the Python package (python code can use `import tensorzero`)
- `tensorzero_rust` - this is the (locally-renamed) Rust client package, which avoids conflicts with pyo3-generated code.
