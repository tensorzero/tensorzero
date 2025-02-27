import asyncio
from typing import Any, Dict, Optional

from minijinja import Environment
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


async def generate_answer(
    client: AsyncTensorZeroGateway,
    function_name: str,
    instruction: str,
    demonstrations: str,
    query: str,
    semaphore: asyncio.Semaphore,
    output_schema: Dict[str, Any] = None,
    system_args: Dict[str, Any] = None,
    variant_name: str = "gpt_4o_mini",
    dryrun: bool = True,
) -> Optional["InferenceResponse"]:
    """
    Generate an answer from the client with retries.
    """

    if system_args:
        env = Environment(templates={"system": instruction})
        instruction = env.render_template("system", **system_args)

    try:
        async with semaphore:
            return await client.inference(
                function_name=function_name,
                input={
                    "system": {
                        "instructions": instruction,
                        "demonstrations": demonstrations,
                    },
                    "messages": [
                        {
                            "role": "user",
                            "content": {"query": query},
                        },
                    ],
                },
                variant_name=variant_name,
                output_schema=output_schema,
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
