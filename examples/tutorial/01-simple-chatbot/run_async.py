import asyncio

from tensorzero import AsyncTensorZeroGateway


async def main():
    async with await AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000"
    ) as client:
        result = await client.inference(
            function_name="mischievous_chatbot",
            input={
                "system": "You are a friendly but mischievous AI assistant.",
                "messages": [
                    {"role": "user", "content": "What is the capital of Japan?"},
                ],
            },
        )

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
