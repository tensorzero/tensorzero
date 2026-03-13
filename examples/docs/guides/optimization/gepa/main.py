import asyncio
import json

from ner import Row, load_dataset
from tensorzero import (
    AsyncTensorZeroGateway,
    CreateDatapointRequestJson,
    GEPAConfig,
    JsonDatapointOutputUpdate,
)

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

DATASET_NAME = "extract_entities_dataset"

NUM_SAMPLES = 500
MAX_CONCURRENCY = 50  # lower this value if you get rate limited

VAL_FRACTION = 0.5


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

    # Configure GEPA optimization
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

    print("\nLaunching GEPA optimization...")

    # Launch the optimization workflow
    job_handle = await t0.experimental_launch_optimization_workflow(
        function_name=FUNCTION_NAME,
        template_variant_name=TEMPLATE_VARIANT_NAME,
        dataset_name=DATASET_NAME,
        optimizer_config=optimization_config,
        val_fraction=VAL_FRACTION,
    )

    # Poll for completion
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
