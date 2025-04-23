import asyncio
import itertools
from asyncio import Semaphore

from agent import ask_question
from dataset import load_beerqa
from judge import judge_answer
from tensorzero import AsyncTensorZeroGateway

MAX_SAMPLES = 1
CONCURRENCY = 10


async def main():
    # We initialize an embedded TensorZero client with our configuration file.
    t0 = await AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000",
    )
    semaphore = Semaphore(CONCURRENCY)
    data = load_beerqa()
    agent_variants = ["baseline", "gpt-4.1-mini", "claude-3.5-haiku"]
    compact_context_variants = ["baseline", "gpt-4.1-nano"]
    # We want to evaluate all combinations of agent and compact_context variants
    tasks = []
    for agent_variant, compact_context_variant in itertools.product(
        agent_variants, compact_context_variants
    ):
        variant_pins = {
            "multi_hop_rag_agent": agent_variant,
            "compact_context": compact_context_variant,
        }
        tasks.append(evaluate_variant_pins(t0, semaphore, data, variant_pins))
    await asyncio.gather(*tasks)


async def evaluate_variant_pins(
    t0: AsyncTensorZeroGateway,
    semaphore: Semaphore,
    data: list[dict],
    variant_pins: dict[str, str],
):
    # data is a list of dictionaries, notably with a "question" key with a string value
    # and a 'answers' key with a list of strings value
    # We want to evaluate the agent on each key
    run_info = await t0.dynamic_evaluation_run(variants=variant_pins)

    # Create tasks for each question
    question_tasks = []
    for question in data[:MAX_SAMPLES]:  # Apply MAX_SAMPLES limit here
        question_tasks.append(
            evaluate_question(t0, semaphore, question, run_info.run_id)
        )

    # Run all question evaluations concurrently
    await asyncio.gather(*question_tasks)


async def evaluate_question(
    t0: AsyncTensorZeroGateway, semaphore: Semaphore, question: dict, run_id: str
):
    episode_info = await t0.dynamic_evaluation_run_episode(run_id=run_id)
    episode_id = episode_info.episode_id
    ai_answer = await ask_question(
        t0, semaphore, question["question"], episode_id=episode_id, verbose=False
    )
    await judge_answer(t0, semaphore, question, ai_answer, episode_id)


if __name__ == "__main__":
    asyncio.run(main())
