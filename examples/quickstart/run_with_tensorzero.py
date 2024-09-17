import asyncio

from tensorzero import AsyncTensorZeroGateway


async def run_with_tensorzero(topic):
    async with AsyncTensorZeroGateway("http://localhost:3000") as client:
        # Run the inference API call...
        result = await client.inference(
            function_name="generate_haiku",
            input={
                "messages": [
                    {
                        "role": "user",
                        "content": "Write a haiku about artificial intelligence.",
                    },
                ],
            },
        )

        print(result)


if __name__ == "__main__":
    asyncio.run(run_with_tensorzero("artificial intelligence"))


# from tensorzero import TensorZeroGateway


# result = TensorZeroGateway("http://localhost:3000").inference(
#     # Since our function has a single variant, the gateway will automatically select it
#     function_name="generate_haiku",
#     input={
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "Write a haiku about artificial intelligence.",
#             }
#         ]
#     },
# )

# print(result)
