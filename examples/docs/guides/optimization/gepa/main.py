import asyncio
import os

from ner import Row, compute_exact_match, compute_jaccard_similarity, load_dataset
from tensorzero import (
    AsyncTensorZeroGateway,
    JsonInferenceResponse,
    OptimizationJobStatus,
)
from tqdm.asyncio import tqdm

FUNCTION_NAME = "extract_entities"
EVALUATION_NAME = "extract_entities_eval"

TEMPLATE_VARIANT_NAME = "baseline"

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
    t0 = await AsyncTensorZeroGateway.build_embedded(  # type: ignore[misc]
        config_file="config/tensorzero.toml",
        clickhouse_url=os.environ.get("TENSORZERO_CLICKHOUSE_URL"),
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

    job_handle = await t0.experimental_launch_optimization_workflow(
        params={
            "function_name": FUNCTION_NAME,
            "template_variant_name": TEMPLATE_VARIANT_NAME,
            "output_source": "inference",
            "limit": NUM_SAMPLES,
            "val_fraction": 0.5,
            "optimizer_config": {
                "type": "gepa",
                "function_name": FUNCTION_NAME,
                "evaluation_name": EVALUATION_NAME,
                "analysis_model": ANALYSIS_MODEL,
                "mutation_model": MUTATION_MODEL,
                "initial_variants": INITIAL_VARIANTS,
                "max_iterations": MAX_ITERATIONS,
                "max_concurrency": MAX_CONCURRENCY,
                "max_tokens": 16384,
            },
        },
    )

    print("GEPA optimization launched. Polling for results...")
    while True:
        job_info = await t0.experimental_poll_optimization(job_handle=job_handle)
        if job_info.status == OptimizationJobStatus.Completed:
            break
        if job_info.status == OptimizationJobStatus.Failed:
            raise RuntimeError(f"GEPA optimization failed: {job_info}")
        print(f"  Status: {job_info.status} — waiting 30s...")
        await asyncio.sleep(30)

    assert job_info.output is not None
    variant_configs = job_info.output["content"]

    for variant_name, variant_config in variant_configs.items():
        print(f"\n# Optimized variant: {variant_name}")
        for template_name, template in variant_config["templates"].items():
            print(f"## '{template_name}' template:")
            print(template["path"]["__data"].strip())


if __name__ == "__main__":
    asyncio.run(main())
