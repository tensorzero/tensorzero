import asyncio

from tensorzero import AsyncTensorZeroGateway


async def main():
    async with await AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000"
    ) as client:
        result = await client.inference(
            function_name="extract_email",
            input={
                "messages": [
                    {
                        "role": "user",
                        "content": "blah blah blah hello@tensorzero.com blah blah blah",
                    }
                ]
            },
        )

        print(result)


if __name__ == "__main__":
    asyncio.run(main())
