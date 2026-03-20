import json
from asyncio import Semaphore, gather

from openai import AsyncOpenAI
from tensorzero import AsyncTensorZeroGateway
from tensorzero.util import UUID


async def judge_answer(
    openai_client: AsyncOpenAI,
    t0: AsyncTensorZeroGateway,
    semaphore: Semaphore,
    question: dict,
    ai_answer: str,
    episode_id: UUID,
    t: int,
):
    async with semaphore:
        response = await openai_client.chat.completions.create(
            model="tensorzero::function_name::judge_answer",
            messages=[
                {
                    "role": "system",
                    "content": [  # type: ignore
                        {
                            "type": "text",
                            "tensorzero::arguments": {
                                "question": question["question"],
                                "answers": question["answers"],
                            },
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [  # type: ignore
                        {
                            "type": "text",
                            "tensorzero::arguments": {"answer": ai_answer},
                        }
                    ],
                },
            ],
            extra_body={"tensorzero::episode_id": str(episode_id)},
        )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("The judge failed to generate a valid output.")

    parsed = json.loads(content)
    score = parsed.get("score")

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
