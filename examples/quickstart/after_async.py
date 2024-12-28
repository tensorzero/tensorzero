import asyncio

from tensorzero import AsyncTensorZeroGateway


async def main():
    async with AsyncTensorZeroGateway("http://localhost:3000") as gateway:
        response = await gateway.inference(
            function_name="generate_haiku",
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
