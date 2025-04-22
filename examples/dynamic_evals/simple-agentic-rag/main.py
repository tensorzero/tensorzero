import asyncio
from asyncio import Semaphore
import os

from dataset import load_beerqa
from agent import ask_question
from tensorzero import AsyncTensorZeroGateway

MAX_SAMPLES = 10
CONCURRENCY = 10

async def main():
    # We initialize an embedded TensorZero client with our configuration file.
    t0 = await AsyncTensorZeroGateway.build_embedded(
        config_file="config/tensorzero.toml",
        clickhouse_url=os.getenv("TENSORZERO_CLICKHOUSE_URL"),
    )

    semaphore = Semaphore(CONCURRENCY)

    data = load_beerqa()
    # data is a list of dictionaries, notably with a "question" key with a string value
    # and a 'answers' key with a list of strings value
    # We want to evaluate the agent on each key

    for question in data:
        ai_answer = await ask_question(t0, semaphore, question["question"])
        # TODO: evaluate the answer


    breakpoint()
if __name__ == "__main__":
    asyncio.run(main())
