import argparse
import asyncio
import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple
from uuid import UUID

import pandas as pd
import yaml
from balrog.environments import make_env
from omegaconf import OmegaConf
from tensorzero import AsyncTensorZeroGateway
from tensorzero.util import uuid7
from tqdm import trange

logger = logging.getLogger(__name__)

ACTION_SPACE = [
    "turn left",
    "turn right",
    "go forward",
    "pick up",
    "drop",
    "toggle",
]

MAX_CONCURRENT_T0_REQUESTS = 50


async def run_episode(
    client: AsyncTensorZeroGateway,
    variant_name: str,
    env_name: str,
    task_name: str,
    episode_idx: int,
    config: OmegaConf,
    semaphore: asyncio.Semaphore,
    history_length: int = 2,
) -> Tuple[float, float, Optional[UUID]]:
    episode_log = {
        "variant": variant_name,
        "task": task_name,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    use_history = "history" in variant_name
    episode_id = uuid7()
    env = make_env(env_name, task_name, config)
    obs, _ = env.reset(seed=episode_idx)
    mission = obs["mission"]
    episode_return = 0
    history = []
    for step in range(env.max_steps):
        # Generate action
        try:
            async with semaphore:
                # Generate message content
                state = obs["text"]["long_term_context"]
                # Generate action given message content
                raw_response = await client.inference(
                    function_name="generate_action",
                    variant_name=variant_name,
                    input={
                        "system": {
                            "mission": mission,
                        },
                        "messages": [
                            {
                                "role": "user",
                                "content": {
                                    "observation": state,
                                    "history": "\n".join(history[-history_length:]),
                                },
                            }
                        ],
                    },
                    episode_id=episode_id,
                )
                episode_log["input_tokens"] += raw_response.usage.input_tokens
                episode_log["output_tokens"] += raw_response.usage.output_tokens
            # Extract action from raw response
            async with semaphore:
                action_response = await client.inference(
                    function_name="extract_action",
                    input={
                        "messages": [{"role": "user", "content": raw_response.content}]
                    },
                    episode_id=episode_id,
                )
                episode_log["input_tokens"] += action_response.usage.input_tokens
                episode_log["output_tokens"] += action_response.usage.output_tokens
            action = action_response.output.parsed["action"]
            # Check if action is valid and set to default if not
            action = env.check_action_validity(action)
        except Exception as e:
            # Handle error
            logger.error(f"Error occurred: {type(e).__name__}: {e}")
            logger.info("Choosing a random legal move as fallback.")
            action = random.choice(ACTION_SPACE)
        # update history
        if use_history:
            history.append(f"Observation:{state}\n\nYour Response:\n{action}\n")
        # Interact with environment
        obs, reward, terminated, truncated, info = env.step(action)
        # Update episode return
        episode_return += reward
        # Check if episode is done and break if so
        done = terminated or truncated
        if done:
            break
    # See if episode is successful
    progression = env.get_stats()["progression"]
    # Log feedback
    _ = await client.feedback(
        metric_name="episode_return",
        episode_id=episode_id,
        value=episode_return,
    )
    _ = await client.feedback(
        metric_name="progression",
        episode_id=episode_id,
        value=progression,
    )
    episode_log["episode_return"] = episode_return
    episode_log["num_steps"] = step + 1
    episode_log["failed_candidates"] = env.failed_candidates
    episode_log.update(env.get_stats())
    episode_log["seed"] = episode_idx
    episode_log["episode_id"] = episode_id
    return episode_log


async def run_episodes(
    client: AsyncTensorZeroGateway,
    variant_name: str,
    env_name: str,
    task_name: str,
    num_episodes: int,
    config: OmegaConf,
    semaphore: asyncio.Semaphore,
    disable_progress_bar: bool = False,
    history_length: int = 2,
) -> Tuple[List[float], List[float]]:
    progress_bar = trange(
        num_episodes,
        desc=f"{env_name} {task_name} {variant_name}",
        disable=disable_progress_bar,
    )

    tasks = [
        asyncio.create_task(
            run_episode(
                client=client,
                variant_name=variant_name,
                env_name=env_name,
                task_name=task_name,
                episode_idx=episode_idx,
                config=config,
                semaphore=semaphore,
                history_length=history_length,
            )
        )
        for episode_idx in range(num_episodes)
    ]

    num_successes = 0
    episode_logs = []
    for task in asyncio.as_completed(tasks):
        episode_log = await task
        if episode_log["progression"] == 1.0:
            num_successes += 1
        episode_logs.append(episode_log)
        current = len(episode_logs)
        progress_bar.update(1)
        progress_bar.set_postfix(
            {"Success": f"{num_successes}/{current}"},
            refresh=True,
        )
    progress_bar.close()
    return episode_logs


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--variant", type=str, default="baseline")
    parser.add_argument("--history-length", type=int, default=8)
    args = parser.parse_args()
    with open(
        "config.yml",
        "r",
    ) as f:
        config_dict = yaml.safe_load(f)
    config = OmegaConf.create(config_dict)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_T0_REQUESTS)

    task_names = config.tasks.babyai_tasks

    results = []
    for task_name in task_names:
        async with AsyncTensorZeroGateway(
            "http://localhost:3000", timeout=180.0
        ) as client:
            results_task = await run_episodes(
                client=client,
                variant_name=args.variant,
                env_name="babyai",
                task_name=task_name,
                num_episodes=args.num_episodes,
                config=config,
                semaphore=semaphore,
                disable_progress_bar=False,
                history_length=args.history_length,
            )
            results.extend(results_task)

    df = pd.DataFrame(results)
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / Path(
        f"{args.variant}_{args.num_episodes}_{args.history_length}.csv"
    )
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    asyncio.run(main())
