import asyncio
import time
from typing import Optional

from tensorzero import AsyncTensorZeroGateway

semaphore = asyncio.Semaphore(20)


async def play_21_questions(
    client: AsyncTensorZeroGateway,
    semaphore: asyncio.Semaphore,
    episode_id: str,
) -> bool:
    start_time = time.time()
    async with semaphore:
        generate_secret_response = await client.inference(
            function_name="generate_secret",
            input={},
            episode_id=episode_id,
        )
    secret = generate_secret_response.output.parsed["secret"]
    message_history = []
    for i in range(21):
        async with semaphore:
            ask_question_response = await client.inference(
                function_name="ask_question",
                episode_id=episode_id,
                input={
                    "system": {"remaining_questions": 21 - i},
                    "messages": message_history,
                },
            )
        question = ask_question_response.output.parsed["question"]
        message_history.append({"role": "user", "content": question})
        async with semaphore:
            answer_question_response = await client.inference(
                function_name="answer_question",
                episode_id=episode_id,
                input={"system": {"secret": secret}, "messages": message_history},
            )
        full_answer = answer_question_response.content[0].text
        short_answer = full_answer.split()[-1].lower()
        if "solved" in short_answer:
            elapsed_ms = (time.time() - start_time) * 1000
            client.feedback(episode_id=episode_id, metric_name="elapsed_ms", value=elapsed_ms)
            client.feedback(episode_id=episode_id, metric_name="solved", value=True)
            return True
        message_history.append({"role": "assistant", "content": short_answer})
    elapsed_ms = (time.time() - start_time) * 1000
    client.feedback(episode_id=episode_id, metric_name="elapsed_ms", value=elapsed_ms)
    await client.feedback(episode_id=episode_id, metric_name="solved", value=False)
    return False


async def safe_play_21_questions(
    client: AsyncTensorZeroGateway, semaphore: asyncio.Semaphore, run_id: str
) -> Optional[bool]:
    run_episode_response = await client.dynamic_evaluation_run_episode(run_id=run_id, tags={"baz": "bat"})
    episode_id = run_episode_response.episode_id
    try:
        return await play_21_questions(client, semaphore, episode_id)
    except Exception as e:
        print(f"Error during 21 questions game: {e}")
        return None


async def run_dynamic_evaluation(
    t0: AsyncTensorZeroGateway,
    display_name: str,
    ask_question_variant: str,
):
    num_games = 50
    run_info = await t0.dynamic_evaluation_run(
        variants={"ask_question": ask_question_variant, "answer_question": "baseline"},
        project_name="21_questions",
        display_name=display_name,
        tags={"foo": "bar"},
    )
    result = [
        x
        for x in await asyncio.gather(
            *[safe_play_21_questions(t0, semaphore, run_info.run_id) for _ in range(num_games)]
        )
        if x is not None
    ]
    print("Asking question variant: ", ask_question_variant)
    print("display name: ", display_name)
    print(f"Solved {sum(result)} / {num_games} games")
    print(f"Run ID: {run_info.run_id}")


async def main():
    t0 = await AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000",
        timeout=60,
    )

    await run_dynamic_evaluation(t0, "gpt-4.1-nano", "gpt-4.1-nano")
    await run_dynamic_evaluation(t0, "baseline", "baseline")
    await run_dynamic_evaluation(t0, "gpt-4.1-mini", "gpt-4.1-mini")


if __name__ == "__main__":
    asyncio.run(main())
