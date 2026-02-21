import asyncio
import os
import random

import toml
from ner import Row, compute_exact_match, compute_jaccard_similarity, load_dataset
from tensorzero import (
    AsyncTensorZeroGateway,
    JsonInferenceResponse,
    ListInferencesRequest,
    OpenAIRFTConfig,
    OptimizationJobStatus,
)
from tqdm.asyncio import tqdm

FUNCTION_NAME = "extract_entities"

# The variant to use for initial inferences and as a template for fine-tuning examples
BASELINE_VARIANT_NAME = "baseline"

# Fine-tuning configuration
MODEL_NAME = "o4-mini-2025-04-16"  # OpenAI reasoning model to fine-tune (RFT requires o-series)
SEED = 42  # Seed for reproducible subsampling of train/val sets
NUM_TRAIN = 10  # Number of training samples to subsample
NUM_VAL = 10  # Number of validation samples to subsample

NUM_SAMPLES = 500  # Number of inferences to run (we subsample from these for RFT)
MAX_CONCURRENCY = 50  # lower this value if you get rate limited


async def process_datapoint(datapoint: Row, t0: AsyncTensorZeroGateway, semaphore: asyncio.Semaphore):
    """
    Run inference on a datapoint and submit metric feedback.
    Unlike SFT, RFT does not require demonstration feedback — the grader
    evaluates outputs during training.
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

        # Compute metrics for tracking (optional, not used by RFT directly)
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

        # Note: No demonstration feedback needed for RFT.
        # The grader will evaluate outputs during training.

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

    # Run inferences and submit metric feedback in parallel
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [process_datapoint(dp, t0, semaphore) for dp in datapoints]
    await tqdm.gather(*tasks, desc="Processing samples and submitting metrics")

    # Query inferences (using output_source="inference" since RFT uses graders, not demonstrations)
    print("\nQuerying inferences...")
    inferences_response = await t0.list_inferences(
        request=ListInferencesRequest(
            function_name=FUNCTION_NAME,
            output_source="inference",  # Use model's original outputs (no demonstrations needed)
        ),
    )

    print(f"Found {len(inferences_response.inferences)} inferences")

    # Render samples using the baseline variant as a template
    # This prepares the data in the format needed for RFT optimization
    rendered_samples = await t0.experimental_render_samples(
        stored_samples=inferences_response.inferences,
        variants={FUNCTION_NAME: BASELINE_VARIANT_NAME},
    )

    print(f"Rendered {len(rendered_samples)} samples")

    # Subsample training and validation sets from all rendered samples.
    # We use a fixed seed for reproducibility — adjust the seed if needed.
    rng = random.Random(SEED)
    rng.shuffle(rendered_samples)
    train_samples = rendered_samples[:NUM_TRAIN]
    val_samples = rendered_samples[NUM_TRAIN : NUM_TRAIN + NUM_VAL]

    print(f"Train samples: {len(train_samples)}, Validation samples: {len(val_samples)}")

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
                    "You will receive **Input** (source text), **Generated Output**, and **Reference Output**.\n"
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
                "content": "Message History:\n{{item.messages}}\n\nGenerated Output:\n{{sample.output_json}}\n\nReference Output:\n{{item.reference_text}}",
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
    print(f"  - Train samples: {len(train_samples)}")
    print(f"  - Validation samples: {len(val_samples)}")

    # Launch the optimization job
    job_handle = await t0.experimental_launch_optimization(
        train_samples=train_samples,
        val_samples=val_samples,
        optimization_config=optimization_config,
    )

    print("\nJob launched!")
    print(job_handle)

    # Poll for completion
    print("\nWaiting for reinforcement fine-tuning to complete...")
    print("(This may take longer than SFT due to the reinforcement learning steps)")

    job_info = await t0.experimental_poll_optimization(job_handle=job_handle)

    # For long-running jobs, you may want to poll periodically:
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

        # Generate the configuration for the fine-tuned model
        model_name = f"{FUNCTION_NAME}_fine_tuned"
        config = {
            "models": {
                model_name: {
                    "routing": ["openai"],
                    "providers": {"openai": {"type": "openai", "model_name": fine_tuned_model}},
                }
            },
            "functions": {
                FUNCTION_NAME: {
                    "variants": {
                        "fine_tuned": {
                            "type": "chat_completion",
                            "model": model_name,
                            "templates": {
                                "system": {
                                    "path": f"functions/{FUNCTION_NAME}/initial_prompt/system_template.minijinja"
                                }
                            },
                            "json_mode": "strict",
                        }
                    }
                }
            },
        }

        print("\n" + "-" * 40)
        print("To use this model, add to your tensorzero.toml:")
        print("-" * 40)
        print(toml.dumps(config))
    else:
        print("RFT Optimization Failed!")
        print("=" * 60)
        print(f"Status: {job_info.status}")
        if job_info.message:
            print(f"Message: {job_info.message}")


if __name__ == "__main__":
    asyncio.run(main())
