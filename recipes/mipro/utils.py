import asyncio
from typing import Any, Dict, Optional

from tensorzero import AsyncTensorZeroGateway, InferenceResponse


async def get_instructions(
    client: AsyncTensorZeroGateway,
    example_instructions: str,
    example_schema: str,
    semaphore: asyncio.Semaphore,
    variant_name: str = "baseline",
    dryrun: bool = True,
) -> Optional["InferenceResponse"]:
    """
    Get instructions from the client with retries.
    """
    input_args = {
        "system": {
            "example_instructions": example_instructions,
        }
    }

    if example_schema:
        input_args["system"]["example_schema"] = example_schema

    try:
        async with semaphore:
            return await client.inference(
                function_name="generate_instruction",
                input=input_args,
                variant_name=variant_name,
                dryrun=dryrun,
            )
    except Exception as e:
        print(f"Error generating instructions: {e}")
        return None


async def candidate_inference(
    client: AsyncTensorZeroGateway,
    function_name: str,
    input: Dict[str, Any],
    variant_name: str,
    semaphore: asyncio.Semaphore,
    dryrun: bool = True,
) -> Optional["InferenceResponse"]:
    try:
        async with semaphore:
            return await client.inference(
                function_name=function_name,
                input=input,
                variant_name=variant_name,
                dryrun=dryrun,
            )
    except Exception as e:
        print(f"Error generating answer: {e}")
        return None


async def judge_answer(
    client: AsyncTensorZeroGateway,
    task_description: str,
    metric_properties: str,
    prediction: str,
    truth: str,
    semaphore: asyncio.Semaphore,
    variant_name: str = "baseline",
    dryrun: bool = True,
) -> Optional["InferenceResponse"]:
    try:
        async with semaphore:
            return await client.inference(
                function_name="judge_answer",
                input={
                    "system": {
                        "task_description": task_description,
                        "metric_properties": metric_properties,
                    },
                    "messages": [
                        {
                            "role": "user",
                            "content": {"prediction": prediction, "truth": truth},
                        },
                    ],
                },
                variant_name=variant_name,
                dryrun=dryrun,
            )
    except Exception as e:
        print(f"Error judging answer: {e}")
        return None
