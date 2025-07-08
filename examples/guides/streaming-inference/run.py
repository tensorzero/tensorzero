import asyncio
import os

from tensorzero import AsyncTensorZeroGateway


async def main(gateway_url: str):
    async with await AsyncTensorZeroGateway.build_http(
        gateway_url=gateway_url
    ) as client:
        stream = await client.inference(
            function_name="chatbot",
            input={
                "messages": [
                    {
                        "role": "user",
                        "content": "Share an extensive list of fun facts about Japan.",
                    },
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
