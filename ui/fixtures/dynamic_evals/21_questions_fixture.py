import asyncio
from typing import Optional

from tensorzero import AsyncTensorZeroGateway




semaphore = asyncio.Semaphore(20)

async def play_21_questions(
    client: AsyncTensorZeroGateway,
    semaphore: asyncio.Semaphore,
    episode_id: str,
) -> bool:
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
            client.feedback(episode_id=episode_id, metric_name="solved", value=True)
            return True
        message_history.append({"role": "assistant", "content": short_answer})
    await client.feedback(episode_id=episode_id, metric_name="solved", value=False)
    return False

async def safe_play_21_questions(
    client: AsyncTensorZeroGateway, semaphore: asyncio.Semaphore, run_id: str
) -> Optional[bool]:
    run_episode_response = await client.dynamic_evaluation_run_episode(run_id=run_id)
    episode_id = run_episode_response.episode_id
    try:
        return await play_21_questions(client, semaphore, episode_id)
    except Exception as e:
        print(f"Error during 21 questions game: {e}")
        return None

async def main():
    num_games = 50
    t0= AsyncTensorZeroGateway(
        "http://localhost:3000",
        timeout=60,
    )
    run_info = await t0.dynamic_evaluation_run(
        variants={"ask_question": "gpt-4.1-nano"},
        project_name="21_questions",
        display_name="gpt-4.1-nano",
    )


    result = [
        x
        for x in await asyncio.gather(
            *[
                safe_play_21_questions(t0, semaphore, run_info.run_id)
                for _ in range(num_games)
            ]
        )
        if x is not None
    ]
    print(f"Solved {sum(result)} / {num_games} games")

if __name__ == "__main__":
    asyncio.run(main())
