import asyncio
import json

from ner import Row, compute_exact_match, compute_jaccard_similarity, load_dataset
from openai import AsyncOpenAI
from tensorzero import AsyncTensorZeroGateway
from tqdm.asyncio import tqdm

NUM_SAMPLES = 500
MAX_CONCURRENCY = 50  # lower this value if you get rate limited


async def process_datapoint(
    datapoint: Row,
    openai_client: AsyncOpenAI,
    t0: AsyncTensorZeroGateway,
    semaphore: asyncio.Semaphore,
):
    async with semaphore:
        try:
            response = await openai_client.chat.completions.create(
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
        await t0.feedback(
            metric_name="exact_match",
            value=exact_match,
            inference_id=inference_id,
        )

        await t0.feedback(
            metric_name="jaccard_similarity",
            value=jaccard_similarity,
            inference_id=inference_id,
        )

        return response


async def main():
    openai_client = AsyncOpenAI(
        base_url="http://localhost:3000/openai/v1",
        api_key="not-used",
    )

    t0 = await AsyncTensorZeroGateway.build_http(  # type: ignore
        gateway_url="http://localhost:3000",
        timeout=30,
    )

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
    tasks = [process_datapoint(dp, openai_client, t0, semaphore) for dp in datapoints]
    await tqdm.gather(*tasks, desc="Processing samples")


if __name__ == "__main__":
    asyncio.run(main())
