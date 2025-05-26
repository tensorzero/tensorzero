import asyncio
import itertools
from asyncio import Semaphore
from tqdm.asyncio import tqdm_asyncio

from agent import ask_question
from dataset import load_beerqa
from judge import judge_answer
from tensorzero import AsyncTensorZeroGateway


MAX_SAMPLES = 10
CONCURRENCY = 10


async def main():
    # We initialize a TensorZero client with our configuration file.
    t0 = await AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000",
    )
    semaphore = Semaphore(CONCURRENCY)
    data = load_beerqa()

    agent_variants = ["baseline", "gpt-4.1-mini", "gemini-2.5-flash"]
    compact_context_variants = ["baseline", "gemini-2.5-flash"]

    # We want to evaluate all combinations of agent and compact_context variants
    for agent_variant, compact_context_variant in itertools.product(
        agent_variants, compact_context_variants
    ):
        variant_pins = {
            "multi_hop_rag_agent": agent_variant,
            "compact_context": compact_context_variant,
        }
        print(f"Evaluating: {variant_pins}")
        await evaluate_variant_pins(t0, semaphore, data, variant_pins)


async def evaluate_variant_pins(
    t0: AsyncTensorZeroGateway,
    semaphore: Semaphore,
    data: list[dict],
    variant_pins: dict[str, str],
):
    display_name = f"agent-{variant_pins['multi_hop_rag_agent']}-compact_context-{variant_pins['compact_context']}"

    run_info = await t0.dynamic_evaluation_run(
        variants=variant_pins,
        project_name="beerqa-agentic-rag",
        display_name=display_name,
    )

    # Create tasks for each question
    # `data` is a list of dictionaries. Each dictionary has a `question` (string) and `answers` (list of strings).
    question_tasks = []
    for question in data[:MAX_SAMPLES]:  # Apply MAX_SAMPLES limit here
        question_tasks.append(
            evaluate_question(t0, semaphore, question, run_info.run_id)
        )

    # Run all question evaluations concurrently
    await tqdm_asyncio.gather(*question_tasks)


async def evaluate_question(
    t0: AsyncTensorZeroGateway, semaphore: Semaphore, question: dict, run_id: str
):
    try:
        episode_info = await t0.dynamic_evaluation_run_episode(
            run_id=run_id, task_name=question["id"]
        )
        episode_id = episode_info.episode_id
        result = await ask_question(
            t0, semaphore, question["question"], episode_id=episode_id, verbose=False
        )
        await judge_answer(t0, semaphore, question, result.answer, episode_id, result.t)
    except Exception as e:
        print(f"Error evaluating question {question['id']}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
