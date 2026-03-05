import asyncio
import json

import httpx
from ner import Row, compute_exact_match, compute_jaccard_similarity, load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

NUM_SAMPLES = 500
MAX_CONCURRENCY = 50  # lower this value if you get rate limited

GATEWAY_URL = "http://localhost:3000"


async def process_datapoint(
    datapoint: Row,
    client: AsyncOpenAI,
    http_client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
):
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="tensorzero::function_name::extract_entities",
                messages=[
                    {
                        "role": "user",
                        "content": datapoint.input,
                    }
                ],
                extra_body={
                    "tensorzero::cache_options": {"enabled": "on"},
                },
            )
        except Exception as e:
            print(f"Error occurred: {e}")
            return None

        inference_id = response.id

        # Get the predicted output
        content = response.choices[0].message.content
        predicted = json.loads(content) if content else {}

        # Compute metrics
        exact_match = compute_exact_match(predicted, datapoint.label)
        jaccard_similarity = compute_jaccard_similarity(predicted, datapoint.label)

        # Send feedback to TensorZero
        await http_client.post(
            f"{GATEWAY_URL}/feedback",
            json={
                "metric_name": "exact_match",
                "value": exact_match,
                "inference_id": inference_id,
            },
        )

        await http_client.post(
            f"{GATEWAY_URL}/feedback",
            json={
                "metric_name": "jaccard_similarity",
                "value": jaccard_similarity,
                "inference_id": inference_id,
            },
        )

        return response


async def main():
    client = AsyncOpenAI(
        base_url=f"{GATEWAY_URL}/openai/v1",
        api_key="not-used",
    )

    async with httpx.AsyncClient() as http_client:
        # Load datapoints
        dataset = load_dataset()
        datapoints = []
        for i in range(NUM_SAMPLES):
            try:
                datapoints.append(next(dataset))
            except StopIteration:
                print(f"Dataset exhausted after {i} samples")
                break

        # Run inferences in parallel with semaphore
        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        tasks = [process_datapoint(dp, client, http_client, semaphore) for dp in datapoints]
        await tqdm.gather(*tasks, desc="Processing samples")


if __name__ == "__main__":
    asyncio.run(main())
