import asyncio

from tensorzero import AsyncTensorZeroGateway


async def run_with_tensorzero(topic):
    async with AsyncTensorZeroGateway("http://localhost:3000") as client:
        # Run the inference API call...
        inference_result = await client.inference(
            function_name="generate_haiku",
            input={
                "messages": [
                    {"role": "user", "content": {"topic": topic}},
                ],
            },
        )

        print(inference_result)

        # ... and associate feedback to that inference using its ID
        feedback_result = await client.feedback(
            metric_name="thumbs_up",
            inference_id=inference_result.inference_id,
            value=True,  # 👍
        )

        print(feedback_result)


if __name__ == "__main__":
    asyncio.run(run_with_tensorzero("artificial intelligence"))
