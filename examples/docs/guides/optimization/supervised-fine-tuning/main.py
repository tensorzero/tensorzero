import asyncio
import os
import random
import time

import toml
from ner import Row, compute_exact_match, compute_jaccard_similarity, load_dataset
from tensorzero import (
    AsyncTensorZeroGateway,
    JsonInferenceResponse,
    ListInferencesRequest,
    OpenAISFTConfig,
    OptimizationJobStatus,
)
from tqdm.asyncio import tqdm

FUNCTION_NAME = "extract_entities"

# The variant to use for initial inferences and as a template for fine-tuning examples
BASELINE_VARIANT_NAME = "baseline"

# Fine-tuning configuration
MODEL_NAME = "gpt-4.1-2025-04-14"  # OpenAI model to fine-tune
VAL_FRACTION = 0.2  # 20% of data for validation

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

        # Compute metrics for tracking (optional, not used by SFT)
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
        # This is the key step for SFT: we're providing the "correct" output
        # that will be used as training data for the fine-tuned model.
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
    # This prepares the data in the format needed for SFT optimization
    rendered_samples = await t0.experimental_render_samples(
        stored_samples=inferences_response.inferences,
        variants={FUNCTION_NAME: BASELINE_VARIANT_NAME},
    )

    print(f"Rendered {len(rendered_samples)} samples")

    # Split into training and validation sets
    random.shuffle(rendered_samples)
    split_idx = int(len(rendered_samples) * (1 - VAL_FRACTION))
    train_samples = rendered_samples[:split_idx]
    val_samples = rendered_samples[split_idx:]

    print(f"Train samples: {len(train_samples)}, Validation samples: {len(val_samples)}")

    # Configure SFT optimization
    optimization_config = OpenAISFTConfig(
        model=MODEL_NAME,
    )

    print("\nLaunching SFT optimization...")
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
    print(f"  - Job ID: {job_handle.job_id}")
    print(f"  - Monitor at: {job_handle.job_url}")

    # Poll for completion
    print("\nWaiting for fine-tuning to complete...")
    print("(This may take 10-30 minutes depending on dataset size)")

    job_info = await t0.experimental_poll_optimization(job_handle=job_handle)

    # For long-running jobs, you may want to poll periodically:
    while job_info.status == OptimizationJobStatus.Pending:
        print(f"  Status: {job_info.status}")
        time.sleep(60)  # wait 1 minute between polls
        job_info = await t0.experimental_poll_optimization(job_handle=job_handle)

    print("\n" + "=" * 60)
    if job_info.status == OptimizationJobStatus.Completed:
        print("SFT Optimization Complete!")
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
        print("SFT Optimization Failed!")
        print("=" * 60)
        print(f"Status: {job_info.status}")
        if job_info.message:
            print(f"Message: {job_info.message}")


if __name__ == "__main__":
    asyncio.run(main())
