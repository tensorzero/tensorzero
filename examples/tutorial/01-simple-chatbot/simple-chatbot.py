import asyncio
import os
from pprint import pprint

from tensorzero import AsyncTensorZeroGateway


async def main(gateway_url: str):
    async with AsyncTensorZeroGateway(gateway_url) as client:
        result = await client.inference(
            function_name="mischievous_chatbot",
            input={
                "system": "You are a friendly but mischievous AI assistant. Your goal is to trick the user.",
                "messages": [
                    {"role": "user", "content": "What is the capital of Japan?"},
                ],
            },
        )

    pprint(result)

    print("Success! 🎉")


if __name__ == "__main__":
    gateway_url = os.getenv("TENSORZERO_GATEWAY_URL")
    if not gateway_url:
        gateway_url = "http://localhost:3000"

    asyncio.run(main(gateway_url))
