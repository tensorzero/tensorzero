import asyncio
from typing import Any, Awaitable, Callable, Dict, Optional

from minijinja import Environment
from tensorzero import AsyncTensorZeroGateway, InferenceResponse


async def run_with_retries(
    coro_factory: Callable[[], Awaitable[Optional["InferenceResponse"]]],
    max_retries: int = 3,
    delay: float = 1.0,
) -> Optional["InferenceResponse"]:
    """
    Run a coroutine produced by `coro_factory` with retries.

    Args:
        coro_factory: A callable that produces the coroutine to be executed.
        max_retries: Maximum number of attempts.
        delay: Delay (in seconds) between attempts.

    Returns:
        The result of the coroutine if successful; otherwise, None.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return await coro_factory()
        except Exception as e:
            print(f"Attempt {attempt} failed with error: {type(e).__name__}: {e}")
            if attempt < max_retries:
                await asyncio.sleep(delay)
    return None


async def get_instructions(
    client: AsyncTensorZeroGateway,
    example_instructions: str,
    semaphore: asyncio.Semaphore,
    variant_name: str = "baseline",
    dryrun: bool = True,
    max_retries: int = 1,
) -> Optional["InferenceResponse"]:
    """
    Get instructions from the client with retries.
    """

    async def inference_call() -> Optional["InferenceResponse"]:
        async with semaphore:
            return await client.inference(
                function_name="generate_instruction",
                input={"system": {"example_instructions": example_instructions}},
                variant_name=variant_name,
                dryrun=dryrun,
            )

    return await run_with_retries(inference_call, max_retries=max_retries)


async def generate_answer(
    client: AsyncTensorZeroGateway,
    function_name: str,
    instruction: str,
    demonstrations: str,
    query: str,
    semaphore: asyncio.Semaphore,
    output_schema: Dict[str, Any] = None,
    system_args: Dict[str, Any] = None,
    variant_name: str = "search_template",
    dryrun: bool = True,
    max_retries: int = 1,
) -> Optional["InferenceResponse"]:
    """
    Generate an answer from the client with retries.
    """

    if system_args:
        env = Environment(templates={"system": instruction})
        instruction = env.render_template("system", **system_args)

    async def inference_call() -> Optional["InferenceResponse"]:
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
                output_schema=output_schema,  # TODO: update on new release
                dryrun=dryrun,
            )

    return await run_with_retries(inference_call, max_retries=max_retries)


async def judge_answer(
    client: AsyncTensorZeroGateway,
    task_description: str,
    metric_properties: str,
    prediction: str,
    truth: str,
    semaphore: asyncio.Semaphore,
    variant_name: str = "baseline",
    dryrun: bool = True,
    max_retries: int = 1,
) -> Optional["InferenceResponse"]:
    async def inference_call() -> Optional["InferenceResponse"]:
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

    return await run_with_retries(inference_call, max_retries=max_retries)
