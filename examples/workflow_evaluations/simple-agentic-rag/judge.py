from asyncio import Semaphore, gather

from tensorzero import AsyncTensorZeroGateway, JsonInferenceResponse
from tensorzero.util import UUID


async def judge_answer(
    t0: AsyncTensorZeroGateway,
    semaphore: Semaphore,
    question: dict,
    ai_answer: str,
    episode_id: UUID,
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
                        "content": [{"type": "text", "arguments": {"answer": ai_answer}}],
                    }
                ],
            },
        )
    assert isinstance(response, JsonInferenceResponse)

    if response.output.parsed is None:
        raise ValueError("The judge failed to generate a valid output.")

    score = response.output.parsed.get("score")

    if score is None:
        raise ValueError("The judge output is missing the 'score' field.")

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
