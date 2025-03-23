import asyncio

from tensorzero import AsyncTensorZeroGateway


async def main():
    async with await AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000"
    ) as gateway:
        response = await gateway.inference(
            model_name="openai::gpt-4o-mini",
            # & many more e.g. "anthropic::claude-3-7-sonnet-20240229",
            input={
                "messages": [
                    {
                        "role": "user",
                        "content": "Write a haiku about artificial intelligence.",
                    }
                ]
            },
        )

    print(response)


asyncio.run(main())
