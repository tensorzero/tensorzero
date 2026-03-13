import asyncio
import json

from ner import Row, load_dataset
from tensorzero import (
    AsyncTensorZeroGateway,
    CreateDatapointRequestJson,
    DICLOptimizationConfig,
    JsonDatapointOutputUpdate,
)

FUNCTION_NAME = "extract_entities"

# The variant to use as a template for DICL examples
BASELINE_VARIANT_NAME = "baseline"

# DICL variant name (will be created by the optimization)
DICL_VARIANT_NAME = "dicl"

# Embedding model to use for DICL (must be configured in tensorzero.toml)
EMBEDDING_MODEL = "openai::text-embedding-3-small"

# Number of nearest neighbors to retrieve at inference time
K = 10

# Model to use for the DICL variant
MODEL = "openai::gpt-5-mini"

DATASET_NAME = "extract_entities_dataset"

NUM_SAMPLES = 500


def make_datapoint(row: Row) -> CreateDatapointRequestJson:
    """Create a CreateDatapointRequestJson from a dataset row."""
    return CreateDatapointRequestJson(
        function_name=FUNCTION_NAME,
        input={
            "messages": [
                {
                    "role": "user",
                    "content": row.input,
                }
            ]
        },
        output=JsonDatapointOutputUpdate(raw=json.dumps(row.label)),
    )


async def main():
    t0 = await AsyncTensorZeroGateway.build_http(  # type: ignore[misc]
        gateway_url="http://localhost:3000",
    )

    # Load datapoints from the NER dataset
    dataset = load_dataset()
    rows: list[Row] = []
    for i in range(NUM_SAMPLES):
        try:
            rows.append(next(dataset))
        except StopIteration:
            print(f"Dataset exhausted after {i} samples")
            break

    print(f"Creating {len(rows)} datapoints in dataset `{DATASET_NAME}`...")

    # Create datapoints in a TensorZero dataset
    datapoints = [make_datapoint(row) for row in rows]
    await t0.create_datapoints(
        dataset_name=DATASET_NAME,
        requests=datapoints,
    )

    print(f"Created {len(datapoints)} datapoints")

    # Configure DICL optimization
    optimization_config = DICLOptimizationConfig(
        function_name=FUNCTION_NAME,
        variant_name=DICL_VARIANT_NAME,
        embedding_model=EMBEDDING_MODEL,
        k=K,
        model=MODEL,
    )

    print("\nLaunching DICL optimization...")
    print(f"  - Embedding model: {EMBEDDING_MODEL}")
    print(f"  - k (nearest neighbors): {K}")
    print(f"  - Generation model: {MODEL}")

    # Launch the optimization workflow
    job_handle = await t0.experimental_launch_optimization_workflow(
        function_name=FUNCTION_NAME,
        template_variant_name=BASELINE_VARIANT_NAME,
        dataset_name=DATASET_NAME,
        optimizer_config=optimization_config,
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
