import asyncio
import time

from ner import Row, compute_exact_match, compute_jaccard_similarity, load_dataset
from tensorzero import (
    AsyncTensorZeroGateway,
    JsonInferenceResponse,
)
from tqdm.asyncio import tqdm

FUNCTION_NAME = "extract_entities"
EVALUATION_NAME = "extract_entities_eval"
DATASET_NAME = "extract_entities_dataset"

# Models to use for analyzing inferences and generating prompt mutations
ANALYSIS_MODEL = "openai::gpt-5.2"
MUTATION_MODEL = "openai::gpt-5.2"

# Initial variants to start the optimization from
INITIAL_VARIANTS = ["baseline"]

# Number of evolution iterations (each iteration evaluates, analyzes, and mutates variants)
MAX_ITERATIONS = 10

NUM_SAMPLES = 500
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

        # Compute metrics and send feedback to TensorZero
        # Note: these metrics are not used by GEPA but can be used for other purposes
        exact_match = compute_exact_match(predicted, datapoint.label)
        jaccard_similarity = compute_jaccard_similarity(predicted, datapoint.label)

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
    t0 = await AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000",
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

    # Create a dataset from the inferences we just ran
    await t0.create_datapoints_from_inferences(
        dataset_name=DATASET_NAME,
        params={
            "type": "inference_query",
            "function_name": FUNCTION_NAME,
            "output_source": "inference",
        },
    )

    # Launch GEPA optimization
    result = await t0.optimization.gepa.launch(
        function_name=FUNCTION_NAME,
        dataset_name=DATASET_NAME,
        evaluation_name=EVALUATION_NAME,
        analysis_model=ANALYSIS_MODEL,
        mutation_model=MUTATION_MODEL,
        initial_variants=INITIAL_VARIANTS,
        max_iterations=MAX_ITERATIONS,
    )

    task_id = result.task_id
    print(f"GEPA task launched: {task_id}")

    # Poll for results
    while True:
        response = await t0.optimization.gepa.get(task_id=task_id)

        if response["status"] == "completed":
            print("GEPA optimization completed!")
            break
        elif response["status"] == "error":
            print(f"GEPA optimization failed: {response['error']}")
            return
        else:
            progress = response.get("progress")
            if progress:
                print(
                    f"  Iteration {progress['current_iteration']}/{progress['max_iterations']}"
                    f" — {progress['current_step']}"
                )
            time.sleep(10)

    # Print optimized variants and their evaluation statistics
    for variant_name, stats in response["statistics"].items():
        print(f"\n# Variant: {variant_name}")
        for evaluator_name, evaluator_stats in stats.items():
            print(
                f"  {evaluator_name}: mean={evaluator_stats['mean']:.3f}"
                f" stdev={evaluator_stats['stdev']:.3f}"
                f" (n={evaluator_stats['count']})"
            )

    for variant_name, variant_config in response["variants"].items():
        print(f"\n# Optimized variant: {variant_name}")
        for template_name, template in variant_config["templates"].items():
            print(f"## '{template_name}' template:")
            print(template["path"]["__data"])


if __name__ == "__main__":
    asyncio.run(main())
