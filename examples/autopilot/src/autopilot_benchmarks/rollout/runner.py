"""Rollout logic for running LLM Gym episodes against an HTTP TensorZero gateway.

Uses build_http() instead of embedded gateway.
"""

import asyncio
import json
import logging
import math
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import random
from random import SystemRandom
from typing import Any, Optional

import llmgym
from llmgym.agents import TensorZeroAgent
from tensorzero import AsyncTensorZeroGateway, TensorZeroError, TensorZeroInternalError

from autopilot_benchmarks.infra.config_generator import parse_env_config

# Retry settings for transient inference failures
INFERENCE_MAX_RETRIES = 3
INFERENCE_RETRY_BASE_DELAY = 1.0  # seconds
INFERENCE_RETRY_MAX_DELAY = 30.0  # seconds

# Errors considered transient and worth retrying
_TRANSIENT_EXCEPTIONS = (
    TensorZeroInternalError,  # 5xx from gateway/upstream
    ConnectionError,
    OSError,  # covers network-level errors
    asyncio.TimeoutError,
)

logger = logging.getLogger(__name__)


class _InferenceTracker:
    """Wraps a gateway to capture variant_name from inference responses."""

    def __init__(
        self,
        gateway: AsyncTensorZeroGateway,
        *,
        episode_num: int,
        progress: "EpisodeProgress",
    ):
        self._gateway = gateway
        self.episode_num = episode_num
        self.progress = progress
        self.variant_name: Optional[str] = None
        self.current_step: int = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._gateway, name)

    async def inference(self, **kwargs: Any) -> Any:
        function_name = kwargs.get("function_name", "<unknown>")
        inference_num = self.progress.inference_count + 1
        self.progress.inference_count = inference_num
        self.progress.mark(event=f"inference:{function_name}:start")
        started_at = time.perf_counter()
        logger.info(
            "Episode %d step %d inference %d: start function=%s",
            self.episode_num,
            self.current_step,
            inference_num,
            function_name,
        )

        last_exc: Optional[Exception] = None
        for attempt in range(1, INFERENCE_MAX_RETRIES + 1):
            try:
                response = await self._gateway.inference(**kwargs)
                break
            except _TRANSIENT_EXCEPTIONS as exc:
                last_exc = exc
                duration_seconds = time.perf_counter() - started_at
                if attempt < INFERENCE_MAX_RETRIES:
                    delay = min(
                        INFERENCE_RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                        INFERENCE_RETRY_MAX_DELAY,
                    )
                    logger.warning(
                        "Episode %d step %d inference %d attempt %d/%d failed after %.2fs: %s: %s — retrying in %.1fs",
                        self.episode_num,
                        self.current_step,
                        inference_num,
                        attempt,
                        INFERENCE_MAX_RETRIES,
                        duration_seconds,
                        type(exc).__name__,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    self.progress.record_inference_duration(duration_seconds)
                    self.progress.mark(
                        event=f"inference:{function_name}:error:{type(exc).__name__}"
                    )
                    logger.warning(
                        "Episode %d step %d inference %d failed after %d attempts (%.2fs): %s: %s",
                        self.episode_num,
                        self.current_step,
                        inference_num,
                        INFERENCE_MAX_RETRIES,
                        duration_seconds,
                        type(exc).__name__,
                        exc,
                    )
                    raise
            except Exception as exc:
                # Non-transient errors (e.g. TensorZeroError / 4xx) — fail immediately
                duration_seconds = time.perf_counter() - started_at
                self.progress.record_inference_duration(duration_seconds)
                self.progress.mark(
                    event=f"inference:{function_name}:error:{type(exc).__name__}"
                )
                logger.warning(
                    "Episode %d step %d inference %d failed after %.2fs: %s: %s",
                    self.episode_num,
                    self.current_step,
                    inference_num,
                    duration_seconds,
                    type(exc).__name__,
                    exc,
                )
                raise
        else:
            # Should not reach here, but just in case
            assert last_exc is not None
            raise last_exc

        if attempt > 1:
            logger.info(
                "Episode %d step %d inference %d: succeeded on attempt %d/%d",
                self.episode_num,
                self.current_step,
                inference_num,
                attempt,
                INFERENCE_MAX_RETRIES,
            )

        response_variant = getattr(response, "variant_name", None)
        if response_variant is not None and self.variant_name != response_variant:
            if self.variant_name is not None:
                logger.warning(
                    "Episode %d step %d inference %d changed variant from %s to %s",
                    self.episode_num,
                    self.current_step,
                    inference_num,
                    self.variant_name,
                    response_variant,
                )
            self.variant_name = response_variant
            self.progress.variant_name = response_variant

        duration_seconds = time.perf_counter() - started_at
        self.progress.record_inference_duration(duration_seconds)
        self.progress.mark(event=f"inference:{function_name}:complete")
        logger.info(
            "Episode %d step %d inference %d: complete variant=%s processing_time_ms=%s duration=%.2fs",
            self.episode_num,
            self.current_step,
            inference_num,
            getattr(response, "variant_name", None),
            getattr(response, "processing_time_ms", None),
            duration_seconds,
        )
        return response


@dataclass
class EpisodeProgress:
    """Mutable progress state for an in-flight episode."""

    step_num: int = 0
    inference_count: int = 0
    variant_name: Optional[str] = None
    last_event: str = "episode:not_started"
    last_progress_at: Optional[str] = None
    total_inference_seconds: float = 0.0
    total_agent_act_seconds: float = 0.0
    total_env_step_seconds: float = 0.0
    total_execute_command_seconds: float = 0.0
    total_submit_solution_seconds: float = 0.0
    total_other_env_step_seconds: float = 0.0
    slowest_inference_seconds: float = 0.0
    slowest_agent_act_seconds: float = 0.0
    slowest_env_step_seconds: float = 0.0
    last_action_kind: Optional[str] = None
    last_tool_name: Optional[str] = None
    last_tool_timeout: Optional[float] = None
    last_command_preview: Optional[str] = None

    def mark(self, *, event: str, step_num: Optional[int] = None) -> None:
        if step_num is not None:
            self.step_num = step_num
        self.last_event = event
        self.last_progress_at = datetime.now(timezone.utc).isoformat()

    def record_inference_duration(self, duration_seconds: float) -> None:
        self.total_inference_seconds += duration_seconds
        self.slowest_inference_seconds = max(
            self.slowest_inference_seconds, duration_seconds
        )

    def record_agent_act_duration(self, duration_seconds: float) -> None:
        self.total_agent_act_seconds += duration_seconds
        self.slowest_agent_act_seconds = max(
            self.slowest_agent_act_seconds, duration_seconds
        )

    def record_env_step_duration(
        self,
        duration_seconds: float,
        *,
        action_kind: str,
        tool_name: Optional[str],
    ) -> None:
        self.total_env_step_seconds += duration_seconds
        self.slowest_env_step_seconds = max(
            self.slowest_env_step_seconds, duration_seconds
        )
        if action_kind == "tool_call" and tool_name == "execute_command":
            self.total_execute_command_seconds += duration_seconds
        elif action_kind == "tool_call" and tool_name == "submit_solution":
            self.total_submit_solution_seconds += duration_seconds
        else:
            self.total_other_env_step_seconds += duration_seconds

    def timing_summary(self, *, timeout: bool = False) -> dict[str, Any]:
        return {
            "timeout": timeout,
            "total_inference_seconds": self.total_inference_seconds,
            "total_agent_act_seconds": self.total_agent_act_seconds,
            "total_env_step_seconds": self.total_env_step_seconds,
            "total_execute_command_seconds": self.total_execute_command_seconds,
            "total_submit_solution_seconds": self.total_submit_solution_seconds,
            "total_other_env_step_seconds": self.total_other_env_step_seconds,
            "slowest_inference_seconds": self.slowest_inference_seconds,
            "slowest_agent_act_seconds": self.slowest_agent_act_seconds,
            "slowest_env_step_seconds": self.slowest_env_step_seconds,
            "last_action_kind": self.last_action_kind,
            "last_tool_name": self.last_tool_name,
            "last_tool_timeout": self.last_tool_timeout,
            "last_command_preview": self.last_command_preview,
        }


@dataclass
class EpisodeResult:
    """Result of a single episode execution."""

    episode_num: int
    success: bool
    variant_name: Optional[str] = None
    feedback_values: dict[str, list] = field(default_factory=dict)
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    timestamp: Optional[str] = None
    timing: dict[str, Any] = field(default_factory=dict)


def _summarize_action(action: Any) -> dict[str, Any]:
    if hasattr(action, "content") and isinstance(getattr(action, "content"), dict):
        return {"kind": "json", "tool_name": None, "tool_timeout": None, "command_preview": None}

    if not isinstance(action, list) or not action:
        return {
            "kind": "unknown",
            "tool_name": None,
            "tool_timeout": None,
            "command_preview": None,
        }

    tool_calls = [
        content
        for content in action
        if hasattr(content, "name") and hasattr(content, "arguments")
    ]
    text_blocks = [content for content in action if hasattr(content, "content")]

    if len(tool_calls) == 1 and len(action) == 1:
        tool_call = tool_calls[0]
        arguments = getattr(tool_call, "arguments", {}) or {}
        command = arguments.get("command")
        command_preview = None
        if isinstance(command, str):
            command_preview = command.replace("\n", "\\n")
            if len(command_preview) > 120:
                command_preview = command_preview[:117] + "..."
        return {
            "kind": "tool_call",
            "tool_name": getattr(tool_call, "name", None),
            "tool_timeout": arguments.get("timeout"),
            "command_preview": command_preview,
        }

    if text_blocks and not tool_calls:
        return {
            "kind": "text",
            "tool_name": None,
            "tool_timeout": None,
            "command_preview": None,
        }

    if tool_calls:
        tool_names = ",".join(
            str(getattr(tool_call, "name", "<unknown>")) for tool_call in tool_calls
        )
        return {
            "kind": "mixed",
            "tool_name": tool_names,
            "tool_timeout": None,
            "command_preview": None,
        }

    return {
        "kind": "unknown",
        "tool_name": None,
        "tool_timeout": None,
        "command_preview": None,
    }


async def run_episode(
    env_name: str,
    gateway: AsyncTensorZeroGateway,
    dryrun: bool,
    agent_semaphore: asyncio.Semaphore,
    episode_num: int,
    task_split: Optional[str] = None,
    env_config_extra: Optional[dict] = None,
    progress: Optional[EpisodeProgress] = None,
    seed: Optional[int] = None,
    task_idx: Optional[int] = None,
) -> EpisodeResult:
    """Run a single episode with its own agent instance.

    Args:
        env_name: Name of the LLM Gym environment.
        gateway: TensorZero gateway for LLM function calls.
        dryrun: If True, run in dryrun mode.
        agent_semaphore: Semaphore for controlling agent-level concurrency.
        episode_num: Episode number (for logging).
        task_split: Optional dataset split (e.g., "train", "test").
        env_config_extra: Optional dict of extra params for llmgym.make().
        task_idx: Optional explicit task index. If provided, overrides random
            task sampling. Used by unique_tasks mode to ensure each task runs
            exactly once.

    Returns:
        EpisodeResult indicating success or failure with error details.
    """
    env = None
    progress = progress or EpisodeProgress()
    try:
        logger.info("Starting episode %d", episode_num)

        parsed_env_name, env_config = parse_env_config(
            env_name,
            task_split=task_split,
            env_config_extra=env_config_extra,
        )

        tracker = _InferenceTracker(
            gateway,
            episode_num=episode_num,
            progress=progress,
        )
        agent = TensorZeroAgent(
            env_name=parsed_env_name,
            gateway=tracker,  # type: ignore[arg-type]
            dryrun=dryrun,
            semaphore=agent_semaphore,
        )

        try:
            env = llmgym.make(parsed_env_name, config=env_config)
        except TypeError:
            # Some envs don't accept task_split — retry without it
            env_config.pop("task_split", None)
            env = llmgym.make(parsed_env_name, config=env_config)

        # Select task index: use explicit task_idx if provided, otherwise sample randomly
        if task_idx is not None:
            logger.debug(
                "Episode %d: using explicit task_idx %d",
                episode_num,
                task_idx,
            )
        elif hasattr(env, "num_tasks") and env.num_tasks is not None:
            if seed is not None:
                rng = random.Random(seed + episode_num)
            else:
                rng = SystemRandom()
            task_idx = rng.randint(0, env.num_tasks - 1)
            logger.debug(
                "Episode %d: selected task %d of %d (seed=%s)",
                episode_num,
                task_idx,
                env.num_tasks,
                seed,
            )

        reset_data = await env.reset(task_idx=task_idx)
        obs = reset_data.observation

        feedback_values: dict[str, list] = defaultdict(list)

        max_steps = env.horizon
        for step_num in range(1, max_steps + 1):
            progress.mark(event="agent_act:start", step_num=step_num)
            tracker.current_step = step_num
            logger.info("Episode %d step %d: agent.act start", episode_num, step_num)
            agent_act_started_at = time.perf_counter()
            action = await agent.act(obs)
            agent_act_duration = time.perf_counter() - agent_act_started_at
            progress.record_agent_act_duration(agent_act_duration)
            action_summary = _summarize_action(action)
            progress.last_action_kind = action_summary["kind"]
            progress.last_tool_name = action_summary["tool_name"]
            progress.last_tool_timeout = action_summary["tool_timeout"]
            progress.last_command_preview = action_summary["command_preview"]
            progress.variant_name = tracker.variant_name
            progress.mark(event="agent_act:complete", step_num=step_num)
            logger.info(
                "Episode %d step %d: agent.act complete variant=%s inferences=%d duration=%.2fs action=%s tool=%s timeout=%s command=%s",
                episode_num,
                step_num,
                tracker.variant_name,
                progress.inference_count,
                agent_act_duration,
                action_summary["kind"],
                action_summary["tool_name"],
                action_summary["tool_timeout"],
                action_summary["command_preview"],
            )
            progress.mark(event="env_step:start", step_num=step_num)
            logger.info("Episode %d step %d: env.step start", episode_num, step_num)
            env_step_started_at = time.perf_counter()
            step_data = await env.step(action)
            env_step_duration = time.perf_counter() - env_step_started_at
            progress.record_env_step_duration(
                env_step_duration,
                action_kind=action_summary["kind"],
                tool_name=action_summary["tool_name"],
            )
            obs = step_data.observation
            done = step_data.terminated or step_data.truncated
            progress.mark(
                event=(
                    "env_step:complete:"
                    f"terminated={step_data.terminated}:truncated={step_data.truncated}"
                ),
                step_num=step_num,
            )
            logger.info(
                "Episode %d step %d: env.step complete terminated=%s truncated=%s duration=%.2fs action=%s tool=%s",
                episode_num,
                step_num,
                step_data.terminated,
                step_data.truncated,
                env_step_duration,
                action_summary["kind"],
                action_summary["tool_name"],
            )

            # Capture feedback values locally
            for fb in step_data.feedback.inference:
                if isinstance(fb.value, (int, float, bool)):
                    feedback_values[fb.name].append(float(fb.value))
            for fb in step_data.feedback.episode:
                if isinstance(fb.value, (int, float, bool)):
                    feedback_values[fb.name].append(float(fb.value))

            # Only send feedback to gateway for non-dryrun (train) rollouts.
            # Dryrun inferences aren't persisted, so feedback would fail.
            if not dryrun:
                await agent.give_feedback(step_data.feedback)
            if done:
                break

        logger.info("Completed episode %d", episode_num)
        progress.variant_name = tracker.variant_name
        progress.mark(event="episode:complete")
        return EpisodeResult(
            episode_num=episode_num,
            success=True,
            variant_name=tracker.variant_name,
            feedback_values=dict(feedback_values),
            timing=progress.timing_summary(),
        )

    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e)
        stack = traceback.format_exc()
        ts = datetime.now(timezone.utc).isoformat()

        logger.error(
            "Episode %d failed with %s at step=%d variant=%s inferences=%d last_event=%s: %s",
            episode_num,
            error_type,
            progress.step_num,
            progress.variant_name,
            progress.inference_count,
            progress.last_event,
            error_message,
        )

        return EpisodeResult(
            episode_num=episode_num,
            success=False,
            variant_name=progress.variant_name,
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack,
            timestamp=ts,
            timing=progress.timing_summary(),
        )

    finally:
        if env is not None:
            try:
                # Call async cleanup directly instead of env.close().
                # env.close() is sync and spawns a new thread with asyncio.run(),
                # which creates a NEW event loop. This breaks Daytona sandbox
                # deletion because the SDK's HTTP client is bound to the original
                # loop ("Future attached to a different loop" error), causing
                # sandboxes to leak.
                if hasattr(env, "_session") and hasattr(env._session, "cleanup"):
                    await env._session.cleanup()
                    env._episode = type(env._episode)()
                    env._function = type(env._function)()
                else:
                    env.close()
            except Exception:
                logger.warning(
                    "Episode %d: failed to close environment",
                    episode_num,
                    exc_info=True,
                )


async def run_rollout(
    gateway: AsyncTensorZeroGateway,
    env_name: str,
    num_episodes: int,
    concurrency: int = 5,
    dryrun: bool = False,
    output_dir: Optional[Path] = None,
    agent_concurrency: Optional[int] = None,
    task_split: Optional[str] = None,
    env_config_extra: Optional[dict] = None,
    episode_timeout: Optional[float] = None,
    seed: Optional[int] = None,
    unique_tasks: bool = False,
) -> dict:
    """Run multiple episodes concurrently using a shared gateway.

    Args:
        gateway: TensorZero gateway (HTTP mode).
        env_name: Name of the LLM Gym environment.
        num_episodes: Number of episodes to run.
        concurrency: Maximum number of concurrent episodes.
        dryrun: If True, run in dryrun mode.
        output_dir: Optional directory to write error logs.
        agent_concurrency: Maximum concurrent agent operations (defaults to concurrency * 10).
        task_split: Optional dataset split (e.g., "train", "test").
        env_config_extra: Optional dict of extra params for llmgym.make().
        episode_timeout: Optional timeout in seconds per episode. Episodes exceeding this are cancelled.
        seed: Optional RNG seed for deterministic task selection. Each episode uses seed + episode_num.
        unique_tasks: If True, run each task at most once. Creates a probe env to
            discover num_tasks, caps episodes to min(num_episodes, num_tasks), and
            assigns sequential task indices. Environments without num_tasks are
            unaffected.

    Returns:
        Summary statistics dict with keys: total, succeeded, failed, error_counts,
        variant_metrics.
    """
    episode_semaphore = asyncio.Semaphore(concurrency)
    agent_semaphore = asyncio.Semaphore(agent_concurrency or concurrency * 10)

    # When unique_tasks is enabled, discover num_tasks and assign indices sequentially
    task_indices: Optional[list[int]] = None
    if unique_tasks:
        parsed_env_name, env_config = parse_env_config(
            env_name,
            task_split=task_split,
            env_config_extra=env_config_extra,
        )
        try:
            probe_env = llmgym.make(parsed_env_name, config=env_config)
        except TypeError:
            env_config.pop("task_split", None)
            probe_env = llmgym.make(parsed_env_name, config=env_config)

        if hasattr(probe_env, "num_tasks") and probe_env.num_tasks is not None:
            n_tasks = probe_env.num_tasks
            actual_episodes = min(num_episodes, n_tasks)
            logger.info(
                "unique_tasks: env has %d tasks, capping episodes from %d to %d",
                n_tasks,
                num_episodes,
                actual_episodes,
            )
            num_episodes = actual_episodes
            task_indices = list(range(num_episodes))
        else:
            logger.info(
                "unique_tasks: env has no num_tasks, falling back to random sampling"
            )
        probe_env.close()

    async def _run_with_timeout(
        episode_num: int,
        task_idx: Optional[int] = None,
    ) -> EpisodeResult:
        progress = EpisodeProgress()
        async with episode_semaphore:
            coro = run_episode(
                env_name=env_name,
                gateway=gateway,
                dryrun=dryrun,
                agent_semaphore=agent_semaphore,
                episode_num=episode_num,
                task_split=task_split,
                env_config_extra=env_config_extra,
                progress=progress,
                seed=seed,
                task_idx=task_idx,
            )
            if episode_timeout is not None:
                try:
                    return await asyncio.wait_for(coro, timeout=episode_timeout)
                except asyncio.TimeoutError:
                    logger.error(
                        "Episode %d timed out after %.0fs",
                        episode_num,
                        episode_timeout,
                    )
                    return EpisodeResult(
                        episode_num=episode_num,
                        success=False,
                        variant_name=progress.variant_name,
                        error_type="TimeoutError",
                        error_message=(
                            f"Episode timed out after {episode_timeout}s "
                            f"(step={progress.step_num}, inferences={progress.inference_count}, "
                            f"last_event={progress.last_event}, "
                            f"last_progress_at={progress.last_progress_at}, "
                            f"agent_act_s={progress.total_agent_act_seconds:.2f}, "
                            f"inference_s={progress.total_inference_seconds:.2f}, "
                            f"env_step_s={progress.total_env_step_seconds:.2f}, "
                            f"execute_command_s={progress.total_execute_command_seconds:.2f}, "
                            f"submit_solution_s={progress.total_submit_solution_seconds:.2f}, "
                            f"other_env_step_s={progress.total_other_env_step_seconds:.2f}, "
                            f"last_tool={progress.last_tool_name}, "
                            f"last_tool_timeout={progress.last_tool_timeout}, "
                            f"last_command={progress.last_command_preview})"
                        ),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        timing=progress.timing_summary(timeout=True),
                    )
            return await coro

    if task_indices is not None:
        tasks = [
            _run_with_timeout(i + 1, task_idx=task_indices[i])
            for i in range(num_episodes)
        ]
    else:
        tasks = [_run_with_timeout(i + 1) for i in range(num_episodes)]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    episode_results: list[EpisodeResult] = []
    for result in results:
        if isinstance(result, EpisodeResult):
            episode_results.append(result)
        elif isinstance(result, BaseException):
            logger.error(
                "Unexpected exception in episode: %s: %s",
                type(result).__name__,
                result,
            )
            episode_results.append(
                EpisodeResult(
                    episode_num=-1,
                    success=False,
                    error_type=type(result).__name__,
                    error_message=str(result),
                    stack_trace="".join(
                        traceback.format_exception(
                            type(result), result, result.__traceback__
                        )
                    ),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )

    successes = [r for r in episode_results if r.success]
    failures = [r for r in episode_results if not r.success]

    if failures and output_dir:
        _write_failed_episodes(failures, output_dir)
    if output_dir:
        _write_episode_timings(episode_results, output_dir)

    _log_summary(episode_results, successes, failures)

    error_counts: dict[str, int] = defaultdict(int)
    for failure in failures:
        if failure.error_type:
            error_counts[failure.error_type] += 1

    variant_metrics = _compute_variant_metrics(successes)

    _warn_all_zero_metrics(successes)

    return {
        "total": len(episode_results),
        "succeeded": len(successes),
        "failed": len(failures),
        "error_counts": dict(error_counts),
        "variant_metrics": variant_metrics,
        "episode_results": episode_results,
    }


def _compute_variant_metrics(
    successes: list[EpisodeResult],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute per-variant, per-metric statistics from successful episodes.

    Returns:
        {metric_name: {variants: {variant_name: {mean, std, n, ci_error}}, n: total}}
    """
    # Count episodes per variant
    episode_counts: dict[str, int] = defaultdict(int)
    for ep in successes:
        vname = ep.variant_name or "unknown"
        episode_counts[vname] += 1

    # Group feedback values by variant and metric
    # variant -> metric -> [values]
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for ep in successes:
        vname = ep.variant_name or "unknown"
        for metric_name, values in ep.feedback_values.items():
            grouped[vname][metric_name].extend(values)

    # Collect all metric names across all variants
    all_metrics: set[str] = set()
    for per_metric in grouped.values():
        all_metrics.update(per_metric.keys())

    total_episodes = len(successes)

    result: dict[str, dict[str, Any]] = {}
    for metric_name in all_metrics:
        variants: dict[str, Any] = {}
        total_n = 0
        for vname, per_metric in grouped.items():
            values = per_metric.get(metric_name, [])
            if not values:
                continue
            n = len(values)
            total_n += n
            mean = sum(values) / n
            if n > 1:
                variance = sum((v - mean) ** 2 for v in values) / (n - 1)
                std = math.sqrt(variance)
            else:
                std = 0.0
            ci_error = 1.96 * std / math.sqrt(n) if n > 0 else None
            variants[vname] = {
                "mean": mean,
                "std": std,
                "n": n,
                "n_episodes": episode_counts[vname],
                "ci_error": ci_error,
            }
        result[metric_name] = {
            "variants": variants,
            "n": total_n,
            "n_episodes": total_episodes,
        }

    return result


def _write_failed_episodes(failures: list[EpisodeResult], output_dir: Path) -> None:
    """Write failed episodes to a JSONL file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    error_file = output_dir / "failed_episodes.jsonl"
    try:
        with open(error_file, "a") as f:
            for failure in failures:
                f.write(json.dumps(asdict(failure)) + "\n")
        logger.info("Wrote %d failed episodes to %s", len(failures), error_file)
    except Exception as e:
        logger.error("Failed to write error file: %s", e)


def _write_episode_timings(episode_results: list[EpisodeResult], output_dir: Path) -> None:
    """Write timing summaries for all episodes to a JSONL file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timing_file = output_dir / "episode_timings.jsonl"
    try:
        with open(timing_file, "w") as f:
            for episode in episode_results:
                f.write(
                    json.dumps(
                        {
                            "episode_num": episode.episode_num,
                            "success": episode.success,
                            "variant_name": episode.variant_name,
                            "error_type": episode.error_type,
                            "timing": episode.timing,
                        }
                    )
                    + "\n"
                )
        logger.info("Wrote %d episode timing rows to %s", len(episode_results), timing_file)
    except Exception as e:
        logger.error("Failed to write timing file: %s", e)


def _warn_all_zero_metrics(successes: list[EpisodeResult]) -> None:
    """Warn loudly if every successful episode got 0.0 for any metric.

    This is a strong signal that external scoring infrastructure (e.g. the
    archipelago grading subprocess) is broken, silently assigning zero scores.
    """
    if len(successes) < 3:
        return

    # Collect all values per metric across successful episodes
    metric_values: dict[str, list[float]] = defaultdict(list)
    for ep in successes:
        for metric_name, values in ep.feedback_values.items():
            metric_values[metric_name].extend(values)

    for metric_name, values in metric_values.items():
        if values and all(v == 0.0 for v in values):
            logger.warning("!" * 72)
            logger.warning(
                "ALL %d values for metric '%s' across %d successful episodes are 0.0!",
                len(values),
                metric_name,
                len(successes),
            )
            logger.warning(
                "This almost certainly indicates a broken scoring/grading pipeline "
                "(e.g. grading subprocess failed, missing ARCHIPELAGO_DIR, missing "
                "GEMINI_API_KEY). Check the episode logs for grading errors."
            )
            logger.warning("!" * 72)


def _log_summary(
    all_results: list[EpisodeResult],
    successes: list[EpisodeResult],
    failures: list[EpisodeResult],
) -> None:
    """Log summary statistics about the rollout."""
    total = len(all_results)
    if total == 0:
        logger.info("No episodes ran")
        return

    logger.info("=" * 60)
    logger.info("Rollout Summary")
    logger.info("=" * 60)
    logger.info("Total episodes: %d", total)
    logger.info("Successful: %d (%.1f%%)", len(successes), len(successes) / total * 100)
    logger.info("Failed: %d (%.1f%%)", len(failures), len(failures) / total * 100)

    if failures:
        error_counts: dict[str, int] = defaultdict(int)
        for failure in failures:
            if failure.error_type:
                error_counts[failure.error_type] += 1
        logger.info("-" * 60)
        logger.info("Error breakdown:")
        for error_type, count in sorted(
            error_counts.items(), key=lambda x: x[1], reverse=True
        ):
            logger.info("  %s: %d", error_type, count)

    logger.info("=" * 60)
