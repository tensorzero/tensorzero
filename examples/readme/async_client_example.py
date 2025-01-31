import asyncio

from tensorzero import AsyncTensorZeroGateway


async def main():
    async with AsyncTensorZeroGateway("http://localhost:3000") as gateway:
        response = await gateway.inference(
            model_name="openai::gpt4o-mini",
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
