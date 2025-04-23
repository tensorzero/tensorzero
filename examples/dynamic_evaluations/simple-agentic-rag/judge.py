from asyncio import Semaphore

from tensorzero import AsyncTensorZeroGateway


async def judge_answer(
    t0: AsyncTensorZeroGateway,
    semaphore: Semaphore,
    question: dict,
    ai_answer: str,
    episode_id: str,
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
    await t0.feedback(
        metric_name="judge_score",
        episode_id=episode_id,
        value=score,
    )
    return score
