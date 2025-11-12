import asyncio

from ner import Row, compute_exact_match, compute_jaccard_similarity, load_dataset
from tensorzero import AsyncTensorZeroGateway, JsonInferenceResponse
from tqdm.asyncio import tqdm

NUM_SAMPLES = 2000
MAX_CONCURRENCY = 50  # lower this value if you get rate limited


async def process_datapoint(datapoint: Row, t0: AsyncTensorZeroGateway, semaphore: asyncio.Semaphore):
    async with semaphore:
        try:
            response = await t0.inference(
                function_name="extract_entities",
                input={
                    "messages": [
                        {
                            "role": "user",
                            "content": datapoint.input,
                        }
                    ]
                },
                cache_options={"enabled": "on"},
            )
            assert isinstance(response, JsonInferenceResponse)
        except Exception as e:
            print(f"Error occurred: {e}")
            return None

        # Get the predicted output
        predicted = response.output.parsed if response.output.parsed else {}

        # Compute metrics
        exact_match = compute_exact_match(predicted, datapoint.label)
        jaccard_similarity = compute_jaccard_similarity(predicted, datapoint.label)

        # Send feedback to TensorZero
        await t0.feedback(
            metric_name="exact_match",
            value=exact_match,
            inference_id=response.inference_id,
        )

        await t0.feedback(
            metric_name="jaccard_similarity",
            value=jaccard_similarity,
            inference_id=response.inference_id,
        )

        return response


async def main():
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
    tasks = [process_datapoint(dp, t0, semaphore) for dp in datapoints]
    await tqdm.gather(*tasks, desc="Processing samples")


if __name__ == "__main__":
    asyncio.run(main())
