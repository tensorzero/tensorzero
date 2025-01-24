import argparse
import asyncio
import sys

import tensorzero

parser = argparse.ArgumentParser()
parser.add_argument("--use-async", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--embedded", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

USE_ASYNC = args.use_async
USE_STREAM = args.stream
USE_EMBEDDED = args.embedded

print("USE_ASYNC: ", USE_ASYNC)
print("USE_STREAM: ", USE_STREAM)
print("USE_EMBEDDED: ", USE_EMBEDDED)


def sync_main():
    if USE_EMBEDDED:
        client = tensorzero.TensorZeroGateway.create_embedded_gateway(
            config_path="../../examples/haiku-hidden-preferences/config/tensorzero.toml",
            clickhouse_url="http://127.0.0.1:8123/tensorzero",
        )
    else:
        client = tensorzero.TensorZeroGateway("http://localhost:3000")

    with client as cli:
        res = cli.inference(
            function_name="judge_haiku",
            input={
                "messages": [
                    {
                        "role": "user",
                        "content": '{"topic": "Rivers", "haiku": "Endless roaring flow. Mountains weep streams for oceans. Carve earth like giants"}',
                    }
                ],
            },
            stream=USE_STREAM,
        )
        if USE_STREAM:
            for chunk in res:
                print(chunk.raw, file=sys.stderr, flush=True, end="")
            print()
        else:
            print("Result: ", res)


async def async_main():
    if USE_EMBEDDED:
        client = await tensorzero.AsyncTensorZeroGateway.create_embedded_gateway(
            config_path="../../examples/haiku-hidden-preferences/config/tensorzero.toml",
            clickhouse_url="http://127.0.0.1:8123/tensorzero",
        )
    else:
        client = tensorzero.AsyncTensorZeroGateway("http://localhost:3000")
    async with client:
        res = await client.inference(
            function_name="judge_haiku",
            input='{"topic": "Rivers", "haiku": "Endless roaring flow. Mountains weep streams for oceans. Carve earth like giants"}',
            stream=USE_STREAM,
        )

        if USE_STREAM:
            async for chunk in res:
                print(chunk.raw, file=sys.stderr, flush=True, end="")
            print()
        else:
            print("Result: ", res)


use_async = False
if use_async:
    asyncio.run(async_main())
else:
    sync_main()
