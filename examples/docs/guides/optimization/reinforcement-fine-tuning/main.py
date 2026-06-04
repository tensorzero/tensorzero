import asyncio
import json

from ner import Row, load_dataset
from tensorzero import (
    AsyncTensorZeroGateway,
    CreateDatapointRequestJson,
    JsonDatapointOutputUpdate,
    OpenAIRFTConfig,
    OptimizationJobStatus,
)

FUNCTION_NAME = "extract_entities"

# The variant to use as a template for fine-tuning examples
BASELINE_VARIANT_NAME = "baseline"

# Fine-tuning configuration
MODEL_NAME = "o4-mini-2025-04-16"  # OpenAI reasoning model to fine-tune (RFT requires o-series)
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

    # Define the grader — the reward function that guides reinforcement fine-tuning.
    # This ScoreModel grader uses an LLM judge to evaluate NER quality.
    grader = {
        "type": "score_model",
        "name": "ner_judge",
        "model": "gpt-4.1-mini-2025-04-14",
        "input": [
            {
                "role": "developer",
                "content": (
                    "You are an impartial grader for a Named Entity Recognition (NER) task.\n"
                    "You will receive the message history, the model's generated output, "
                    "and a reference output.\n"
                    "Compare the generated output against the reference output and return a JSON object "
                    "with a single key `score` whose value is **-1**, **0**, or **1**.\n\n"
                    "# Task Description\n"
                    "Extract named entities from text into four categories:\n"
                    "- **person**: Names of specific people\n"
                    "- **organization**: Names of companies, institutions, agencies, or groups\n"
                    "- **location**: Names of geographical locations (countries, cities, landmarks)\n"
                    "- **miscellaneous**: Other named entities (events, products, nationalities, etc.)\n\n"
                    "# Evaluation Criteria (in priority order)\n\n"
                    "## 1. Correctness\n"
                    "- Only **proper nouns** should be extracted (specific people, places, organizations, things)\n"
                    "- Do NOT extract: common nouns, category labels, numbers, statistics, metadata, or headers\n\n"
                    "## 2. Verbatim Extraction\n"
                    "- Entities must appear **exactly** as written in the input text\n"
                    "- Preserve original spelling, capitalization, and formatting\n\n"
                    "## 3. No Duplicates\n"
                    "- Each entity should appear **exactly once** in the output\n\n"
                    "## 4. Completeness\n"
                    "- All valid named entities from the input should be captured\n\n"
                    "## 5. Correct Categorization\n"
                    "- Entities should be placed in the appropriate category\n\n"
                    "# Scoring\n"
                    "- **1 (better)**: Generated output is materially better than reference.\n"
                    "- **0 (similar)**: Outputs are comparable or differences are minor.\n"
                    "- **-1 (worse)**: Generated output is materially worse.\n\n"
                    'Return **only** a JSON object: {"score": <value>}'
                ),
            },
            {
                "role": "user",
                "content": "Message History:\n{{item.messages}}\n\nGenerated Output:\n{{sample.output_text}}\n\nReference Output:\n{{item.reference_text}}",
            },
        ],
        "range": [-1.0, 1.0],
    }

    # Configure RFT optimization
    optimization_config = OpenAIRFTConfig(
        model=MODEL_NAME,
        grader=grader,
    )

    print("\nLaunching RFT optimization...")
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
    print("\nWaiting for reinforcement fine-tuning to complete...")
    print("(This may take longer than SFT due to the reinforcement learning steps)")

    job_info = await t0.experimental_poll_optimization(job_handle=job_handle)

    # For long-running jobs, poll periodically:
    while job_info.status == OptimizationJobStatus.Pending:
        print(f"  Status: {job_info.status}")
        await asyncio.sleep(60)  # wait 1 minute between polls
        job_info = await t0.experimental_poll_optimization(job_handle=job_handle)

    print("\n" + "=" * 60)
    if job_info.status == OptimizationJobStatus.Completed:
        print("RFT Optimization Complete!")
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
        print("RFT Optimization Failed!")
        print("=" * 60)
        print(f"Status: {job_info.status}")
        if job_info.message:
            print(f"Message: {job_info.message}")


if __name__ == "__main__":
    asyncio.run(main())
