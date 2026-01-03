import asyncio
import os

from ner import Row, compute_exact_match, compute_jaccard_similarity, load_dataset
from tensorzero import (
    AsyncTensorZeroGateway,
    DICLOptimizationConfig,
    JsonInferenceResponse,
    ListInferencesRequest,
)
from tqdm.asyncio import tqdm

FUNCTION_NAME = "extract_entities"

# The variant to use for initial inferences and as a template for DICL examples
BASELINE_VARIANT_NAME = "baseline"

# DICL variant name (will be created by the optimization)
DICL_VARIANT_NAME = "dicl"

# Embedding model to use for DICL (must be configured in tensorzero.toml)
EMBEDDING_MODEL = "text_embedding_3_small"

# Number of nearest neighbors to retrieve at inference time
K = 10

# Model to use for the DICL variant
MODEL = "openai::gpt-5-mini"

NUM_SAMPLES = 500
MAX_CONCURRENCY = 50  # lower this value if you get rate limited


async def process_datapoint(datapoint: Row, t0: AsyncTensorZeroGateway, semaphore: asyncio.Semaphore):
    """
    Run inference on a datapoint and submit demonstration feedback using the ground-truth label.
    This simulates a human correcting the model's output.
    """
    async with semaphore:
        try:
            response = await t0.inference(
                function_name=FUNCTION_NAME,
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
            print(f"Error occurred during inference: {e}")
            return None

        # Get the predicted output
        predicted = response.output.parsed if response.output.parsed else {}

        # Compute metrics for tracking (optional, not used by DICL)
        exact_match = compute_exact_match(predicted, datapoint.label)
        jaccard_similarity = compute_jaccard_similarity(predicted, datapoint.label)

        # Send metric feedback (optional, for tracking performance)
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

        # Submit demonstration feedback using the ground-truth label.
        # This is the key step for DICL: we're providing the "correct" output
        # that will be used as an example for similar inputs at inference time.
        await t0.feedback(
            metric_name="demonstration",
            value=datapoint.label,  # Pass the dict directly, not JSON string
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

    print(f"Processing {len(datapoints)} samples...")

    # Run inferences and submit demonstration feedback in parallel
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [process_datapoint(dp, t0, semaphore) for dp in datapoints]
    await tqdm.gather(*tasks, desc="Processing samples and submitting demonstrations")

    # Query inferences that have demonstrations (ground-truth corrected outputs)
    # We use output_source="demonstration" to get examples where the output has been corrected.
    print("\nQuerying inferences with demonstrations...")
    inferences_response = await t0.list_inferences(
        request=ListInferencesRequest(
            function_name=FUNCTION_NAME,
            output_source="demonstration",  # Use demonstrations as the gold standard
            limit=NUM_SAMPLES,
        ),
    )

    print(f"Found {len(inferences_response.inferences)} inferences with demonstrations")

    # Render samples using the baseline variant as a template
    # This prepares the data in the format needed for DICL optimization
    rendered_samples = await t0.experimental_render_samples(
        stored_samples=inferences_response.inferences,
        variants={FUNCTION_NAME: BASELINE_VARIANT_NAME},
    )

    print(f"Rendered {len(rendered_samples)} samples")

    # Configure DICL optimization
    # DICL uses embeddings to find similar examples (no evaluation required)
    optimization_config = DICLOptimizationConfig(
        function_name=FUNCTION_NAME,
        variant_name=DICL_VARIANT_NAME,
        embedding_model=EMBEDDING_MODEL,
        k=K,
        model=MODEL,
        batch_size=128,  # Batch size for embedding generation
        max_concurrency=10,  # Max concurrent embedding requests
    )

    print("\nLaunching DICL optimization...")
    print(f"  - Embedding model: {EMBEDDING_MODEL}")
    print(f"  - k (nearest neighbors): {K}")
    print(f"  - Generation model: {MODEL}")

    # Launch the optimization
    # DICL only needs train_samples (no validation split needed)
    job_handle = await t0.experimental_launch_optimization(
        train_samples=rendered_samples,
        optimization_config=optimization_config,
    )

    # Poll for completion
    job_info = await t0.experimental_poll_optimization(job_handle=job_handle)

    print("\n" + "=" * 60)
    print("DICL Optimization Complete!")
    print("=" * 60)

    # Print the generated DICL variant configuration
    assert job_info.output is not None
    variant_config = job_info.output

    print("\nGenerated DICL variant configuration:")
    print("-" * 40)

    # The output contains the variant configuration parameters
    if "content" in variant_config:
        content = variant_config["content"]
        print(f"  embedding_model: {content.get('embedding_model', EMBEDDING_MODEL)}")
        print(f"  k: {content.get('k', K)}")
        print(f"  model: {content.get('model', MODEL)}")

    print("\n" + "-" * 40)
    print("To use this DICL variant, add the following to your tensorzero.toml:")
    print("-" * 40)
    print(f"""
[functions.{FUNCTION_NAME}.variants.{DICL_VARIANT_NAME}]
type = "experimental_dynamic_in_context_learning"
embedding_model = "{EMBEDDING_MODEL}"
k = {K}
model = "{MODEL}"
json_mode = "strict"
""")

    print("The DICL examples have been stored in ClickHouse.")
    print("At inference time, the k most similar examples will be retrieved")
    print("and included as context for in-context learning.")


if __name__ == "__main__":
    asyncio.run(main())
