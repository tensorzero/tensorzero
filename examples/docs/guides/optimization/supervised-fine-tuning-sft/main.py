import asyncio
import json

from ner import Row, load_dataset
from tensorzero import (
    AsyncTensorZeroGateway,
    CreateDatapointRequestJson,
    JsonDatapointOutputUpdate,
    OpenAISFTConfig,
    OptimizationJobStatus,
)

FUNCTION_NAME = "extract_entities"

# The variant to use as a template for fine-tuning examples
BASELINE_VARIANT_NAME = "baseline"

# Fine-tuning configuration
MODEL_NAME = "gpt-4.1-2025-04-14"  # OpenAI model to fine-tune
VAL_FRACTION = 0.2  # 20% of data for validation

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

    # Configure SFT optimization
    optimization_config = OpenAISFTConfig(
        model=MODEL_NAME,
    )

    print("\nLaunching SFT optimization...")
    print(f"  - Base model: {MODEL_NAME}")
    print(f"  - Validation fraction: {VAL_FRACTION}")

    # Launch the optimization workflow
    job_handle = await t0.experimental_launch_optimization_workflow(
        function_name=FUNCTION_NAME,
        template_variant_name=BASELINE_VARIANT_NAME,
        dataset_name=DATASET_NAME,
        optimizer_config=optimization_config,
        val_fraction=VAL_FRACTION,
    )

    print("Job launched!")

    # Poll for completion
    print("\nWaiting for fine-tuning to complete...")

    job_info = await t0.experimental_poll_optimization(job_handle=job_handle)

    # For long-running jobs, poll periodically:
    while job_info.status == OptimizationJobStatus.Pending:
        print(f"  Status: {job_info.status}")
        await asyncio.sleep(60)  # wait 1 minute between polls
        job_info = await t0.experimental_poll_optimization(job_handle=job_handle)

    print("\n" + "=" * 60)
    if job_info.status == OptimizationJobStatus.Completed:
        print("SFT Optimization Complete!")
        print("=" * 60)

        # Extract the fine-tuned model name from the job output
        assert job_info.output is not None
        fine_tuned_model = job_info.output["routing"][0]
        print(f"\nFine-tuned model: {fine_tuned_model}")

        print("\n" + "-" * 40)
        print("To use this model, add to your tensorzero.toml:")
        print("-" * 40)
        print(f"""
[models.{FUNCTION_NAME}_fine_tuned]
routing = ["openai"]

[models.{FUNCTION_NAME}_fine_tuned.providers.openai]
type = "openai"
model_name = "{fine_tuned_model}"

[functions.{FUNCTION_NAME}.variants.fine_tuned]
type = "chat_completion"
model = "{FUNCTION_NAME}_fine_tuned"
templates.system.path = "functions/{FUNCTION_NAME}/initial_prompt/system_template.minijinja"
json_mode = "strict"
""")
    else:
        print("SFT Optimization Failed!")
        print("=" * 60)
        print(f"Status: {job_info.status}")
        if job_info.message:
            print(f"Message: {job_info.message}")


if __name__ == "__main__":
    asyncio.run(main())
