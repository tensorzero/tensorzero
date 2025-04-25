from asyncio import Semaphore, gather

from tensorzero import AsyncTensorZeroGateway


async def judge_answer(
    t0: AsyncTensorZeroGateway,
    semaphore: Semaphore,
    question: dict,
    ai_answer: str,
    episode_id: str,
    t: int,
):
    async with semaphore:
        response = await t0.inference(
            function_name="judge_answer",
            input={
                "system": {
                    "question": question["question"],
                    "answers": question["answers"],
                },
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "arguments": {"answer": ai_answer}}
                        ],
                    }
                ],
            },
        )
    score = response.output.parsed["score"]
    # Run both feedback calls concurrently
    await gather(
        t0.feedback(
            metric_name="judge_score",
            episode_id=episode_id,
            value=score,
        ),
        t0.feedback(
            metric_name="num_iterations",
            value=t,
            episode_id=episode_id,
        ),
    )
    return score
