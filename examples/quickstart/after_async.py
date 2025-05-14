import asyncio

from tensorzero import AsyncTensorZeroGateway


async def main():
    async with await AsyncTensorZeroGateway.build_embedded(
        clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero",
        config_file="config/tensorzero.toml",
    ) as gateway:
        response = await gateway.inference(
            function_name="generate_haiku",
            input={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Write a haiku about artificial intelligence.",
                            },
                            {
                                "type": "text",
                                "text": "In german.",
                            },
                        ],
                    }
                ]
            },
        )

    print(response)


asyncio.run(main())
