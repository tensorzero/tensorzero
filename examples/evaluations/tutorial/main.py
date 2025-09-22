import asyncio
import random

from tensorzero import AsyncTensorZeroGateway


async def generate_haiku(t0, topic, semaphore):
    async with semaphore:
        response = await t0.inference(
            function_name="write_haiku",
            variant_name="gpt_4o_mini",
            input={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "arguments": {
                                    "topic": topic,
                                },
                            }
                        ],
                    }
                ]
            },
        )
        print(response)


async def main():
    # Set the random seed for reproducibility
    random.seed(0)

    # Build the TensorZero Gateway client
    t0 = await AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000",
    )

    # Load the topics to generate haikus for
    with open("data/nounlist.txt", "r") as f:
        topics = f.read().splitlines()
        random.shuffle(topics)
        topics = topics[:100]

    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(10)

    # Create tasks for all topics
    tasks = [generate_haiku(t0, topic, semaphore) for topic in topics]

    # Run all tasks concurrently
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
