import asyncio
import random
from asyncio import Semaphore

from agent import ask_question
from dataset import load_beerqa
from tensorzero import AsyncTensorZeroGateway
from tensorzero.util import uuid7

MAX_SAMPLES = 10
CONCURRENCY = 10


async def main():
    # We initialize an embedded TensorZero client with our configuration file.
    t0 = await AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000",
    )

    semaphore = Semaphore(CONCURRENCY)

    data = load_beerqa()
    # data is a list of dictionaries, notably with a "question" key with a string value
    # and a 'answers' key with a list of strings value
    # We want to evaluate the agent on each key
    episode_id = uuid7()
    random.shuffle(data)
    for question in data:
        print(f"Question: {question['question']}")
        ai_answer = await ask_question(
            t0, semaphore, question["question"], episode_id=episode_id, verbose=True
        )
        print(f"AI Answer: {ai_answer}")
        print(f"Correct Answer: {question['answers']}")
        score = await judge_answer(t0, semaphore, question, ai_answer, episode_id)
        print(f"Score: {score}")
        break
        # TODO: evaluate the answer


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


if __name__ == "__main__":
    asyncio.run(main())
