import asyncio
import json

from ner import Row, load_dataset
from tensorzero import (
    AsyncTensorZeroGateway,
    CreateDatapointRequestJson,
    JsonDatapointOutputUpdate,
)

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

    # Launch GEPA optimization
    result = await t0.optimization.gepa.launch(
        function_name=FUNCTION_NAME,
        dataset_name=DATASET_NAME,
        evaluation_name=EVALUATION_NAME,
        analysis_model=ANALYSIS_MODEL,
        mutation_model=MUTATION_MODEL,
        initial_variants=INITIAL_VARIANTS,
        max_iterations=MAX_ITERATIONS,
        max_concurrency=MAX_CONCURRENCY,
    )

    task_id = result.task_id
    print(f"GEPA task launched: {task_id}")

    # Poll for results
    last_status = None
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
                status = (
                    f"  Iteration {progress['current_iteration']}/{progress['max_iterations']}"
                    f" — {progress['current_step']}"
                )
            else:
                status = "  Pending..."
            if status != last_status:
                print(status)
                last_status = status
            await asyncio.sleep(10)

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
