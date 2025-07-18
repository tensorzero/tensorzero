import asyncio
from typing import Any, AsyncIterator, Dict, Optional, Union

from tensorzero import (
    AsyncTensorZeroGateway,
    InferenceChunk,
    InferenceInput,
    InferenceResponse,
)


async def get_instructions(
    client: AsyncTensorZeroGateway,
    example_instructions: str,
    example_schema: str,
    semaphore: asyncio.Semaphore,
    variant_name: str = "baseline",
    dryrun: bool = True,
) -> Optional[Union[InferenceResponse, AsyncIterator[InferenceChunk]]]:
    """
    Get instructions from the client with retries.
    """
    system_args: Dict[str, Any] = {
        "example_instructions": example_instructions,
    }

    if example_schema:
        system_args["example_schema"] = example_schema

    inputs = InferenceInput(system=system_args, messages=[])

    try:
        async with semaphore:
            return await client.inference(
                function_name="generate_instruction",
                input=inputs,
                variant_name=variant_name,
                dryrun=dryrun,
            )
    except Exception as e:
        print(f"Error generating instructions: {e}")
        return None


async def candidate_inference(
    client: AsyncTensorZeroGateway,
    input: InferenceInput,
    system_prompt: str,
    model_name: str,
    semaphore: asyncio.Semaphore,
    dryrun: bool = True,
) -> Optional[Union[InferenceResponse, AsyncIterator[InferenceChunk]]]:
    messages = []
    for message in input.messages:  # type: ignore
        messages.append({"role": message.role, "content": message.content})  # type: ignore
    adjusted_input = {"system": system_prompt, "messages": messages}  # type: ignore
    try:
        async with semaphore:
            return await client.inference(
                input=adjusted_input,  # type: ignore
                model_name=model_name,
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
    ground_truth: str,
    semaphore: asyncio.Semaphore,
    variant_name: str = "baseline",
    dryrun: bool = True,
) -> Optional[Union[InferenceResponse, AsyncIterator[InferenceChunk]]]:
    try:
        async with semaphore:
            system_args: Dict[str, Any] = {
                "task_description": task_description,
                "metric_properties": metric_properties,
            }

            inputs = InferenceInput(
                system=system_args,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "arguments": {
                                    "prediction": prediction,
                                    "ground_truth": ground_truth,
                                },
                            }
                        ],
                    },
                ],
            )
            return await client.inference(
                function_name="judge_answer",
                input=inputs,
                variant_name=variant_name,
                dryrun=dryrun,
            )
    except Exception as e:
        print(f"Error judging answer: {e}")
        return None
