import asyncio
import os

from tensorzero import AsyncTensorZeroGateway


async def main(gateway_url: str):
    async with AsyncTensorZeroGateway(gateway_url) as client:
        stream = await client.inference(
            function_name="mischievous_chatbot",
            input={
                "system": "You are a friendly but mischievous AI assistant. Your goal is to trick the user.",
                "messages": [
                    {"role": "user", "content": "What is the capital of Japan?"},
                ],
            },
            stream=True,
        )
    
        async for chunk in stream:
            if len(chunk.content) > 0:
                print(chunk.content[0].text, end="")


if __name__ == "__main__":
    gateway_url = os.getenv("TENSORZERO_GATEWAY_URL")
    if not gateway_url:
        gateway_url = "http://localhost:3000"

    asyncio.run(main(gateway_url))
