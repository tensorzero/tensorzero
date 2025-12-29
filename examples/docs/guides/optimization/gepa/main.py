import asyncio
import os
import random

from ner import Row, compute_exact_match, compute_jaccard_similarity, load_dataset
from tensorzero import (
    AsyncTensorZeroGateway,
    GEPAConfig,
    JsonInferenceResponse,
    ListInferencesRequest,
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

    inferences_response = await t0.list_inferences(
        request=ListInferencesRequest(
            function_name=FUNCTION_NAME,
            output_source="inference",  # could also be "demonstration"
            limit=NUM_SAMPLES,
        ),
    )

    rendered_samples = await t0.experimental_render_samples(
        stored_samples=inferences_response.inferences,
        variants={FUNCTION_NAME: TEMPLATE_VARIANT_NAME},
    )

    random.shuffle(rendered_samples)
    split_idx = len(rendered_samples) // 2
    train_samples = rendered_samples[:split_idx]
    val_samples = rendered_samples[split_idx:]

    optimization_config = GEPAConfig(
        function_name=FUNCTION_NAME,
        evaluation_name=EVALUATION_NAME,
        analysis_model=ANALYSIS_MODEL,
        mutation_model=MUTATION_MODEL,
        initial_variants=INITIAL_VARIANTS,
        max_iterations=MAX_ITERATIONS,
        max_concurrency=MAX_CONCURRENCY,
        max_tokens=16384,
    )

    job_handle = await t0.experimental_launch_optimization(
        train_samples=train_samples,
        val_samples=val_samples,
        optimization_config=optimization_config,
    )

    job_info = await t0.experimental_poll_optimization(job_handle=job_handle)

    assert job_info.output is not None
    variant_configs = job_info.output["content"]

    for variant_name, variant_config in variant_configs.items():
        print(f"\n# Optimized variant: {variant_name}")
        for template_name, template in variant_config["templates"].items():
            print(f"## '{template_name}' template:")
            print(template["path"]["__data"])


if __name__ == "__main__":
    asyncio.run(main())
