"""Main eval orchestrator — runs the full iteration loop for an environment.

Steps per environment:
  1. Generate T0 config from llmgym environment
  2. Reset gateway Postgres for a clean run
  3. Start gateway binary as subprocess (runs PG migrations)
  4. Create HTTP gateway client
  5. Start a second embedded gateway for persisted test rollouts
  6. For each iteration:
     a. Run train rollout on the primary gateway for autopilot observability
     b. Run test rollout on the isolated test gateway
     c. Record iteration to eval Postgres with both train and test metrics
     d. Run autopilot session (create → stream → approve → idle)
     e. Apply config writes via config-applier CLI
     f. Restart both gateways
"""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, urlparse, urlunparse

import asyncpg
import httpx
from tensorzero import AsyncTensorZeroGateway
from tensorzero.util import uuid7

from autopilot_benchmarks.autopilot.config_applier import apply_config_writes
from autopilot_benchmarks.autopilot.interlocutor import Interlocutor
from autopilot_benchmarks.autopilot.session import AutopilotSessionManager
from autopilot_benchmarks.config import EnvironmentConfig, EvalConfig
from autopilot_benchmarks.infra.config_generator import write_initial_env_config
from autopilot_benchmarks.infra.gateway_process import GatewayProcess
from autopilot_benchmarks.infra.postgres_io import import_all_tables
from autopilot_benchmarks.results.recorder import JsonResultRecorder
from autopilot_benchmarks.rollout.runner import run_rollout
from autopilot_benchmarks.snapshot import (
    create_snapshot,
    load_baseline_data,
    restore_config,
    validate_snapshot,
)

logger = logging.getLogger(__name__)


def _build_model_notes(available_models: list[str]) -> list[str]:
    """Return model-specific usage constraints for the initial autopilot prompt."""
    notes: list[str] = [
        "- Do not set the `temperature` parameter on any variant."
        " Some models reject non-default temperature values and"
        " the evaluations will fail.",
    ]

    return notes


# Maps model provider prefixes to the env var(s) that must be set.
_PROVIDER_KEY_ENV_VARS: dict[str, list[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "google_ai_studio_gemini": ["GEMINI_API_KEY", "GOOGLE_AI_STUDIO_API_KEY"],
    "fireworks": ["FIREWORKS_API_KEY"],
}


def _filter_available_models(models: list[str]) -> list[str]:
    """Filter models to only those whose provider API key is set."""
    filtered = []
    for model in models:
        provider = model.split("::")[0] if "::" in model else ""
        required_keys = _PROVIDER_KEY_ENV_VARS.get(provider)
        if required_keys is None:
            # Unknown provider — keep it (don't block unknown providers)
            filtered.append(model)
        elif any(os.environ.get(k) for k in required_keys):
            filtered.append(model)
        else:
            logger.warning(
                "Dropping model %s from available_models: none of %s are set in the environment",
                model,
                required_keys,
            )
    return filtered


def _build_initial_message(env_config: EnvironmentConfig) -> str:
    """Build the initial message for an autopilot session.

    Appends available model list and strategy guidance to the
    user-provided autopilot_initial_message.
    """
    parts = [env_config.autopilot_initial_message]

    if env_config.available_models:
        models = _filter_available_models(env_config.available_models)
        model_list = "\n".join(f"- {m}" for m in models)
        parts.append(f"\nYou have access to the following models:\n{model_list}")
        model_notes = _build_model_notes(models)
        if model_notes:
            parts.append("\nModel-specific constraints:\n" + "\n".join(model_notes))

    parts.append(
        "\nYou should create many variant candidates using different models"
        " and prompting strategies, then use topK evaluation to compare them"
        " all in parallel. Use the evaluation results to winnow the field down"
        " to roughly 4 of the most promising variants, then put those into"
        " production using experimentation config (e.g. track_and_stop)."
        " Do NOT put all variants into track_and_stop — only the top ~4."
        " When you configure experimentation, set `update_period_s` to `5`."
        " so the eval loop stays fast."
        """In particular you should set up an experiment like:
            {
              "function_name": "my_function",
              "experimentation": {
                "type": "track_and_stop",
                "metric": "success",
                "candidate_variants": ["top-variant-1", "top-variant-2", "top-variant-3", "top-variant-4"],
                "fallback_variants": ["baseline-v1"],
                "min_samples_per_variant": 5, // keep this low
                "update_period_s": 5,
                "delta": 0.1 // this can be high
              }
            }"""
    )

    return "\n".join(parts)


def _merge_error_counts(*stats_dicts: Optional[dict[str, Any]]) -> dict[str, int]:
    """Merge rollout error counts across phases/splits."""
    merged: dict[str, int] = {}
    for stats in stats_dicts:
        if not stats:
            continue
        for error_type, count in stats.get("error_counts", {}).items():
            merged[error_type] = merged.get(error_type, 0) + int(count)
    return merged


def _rollout_output_dir(
    work_dir: Path,
    *,
    iteration: int,
    phase: str,
    split: str,
) -> Path:
    """Return the artifact directory for one rollout."""
    return work_dir / "rollouts" / f"iteration_{iteration:03d}" / phase / split


def _test_episode_concurrency(env_config: EnvironmentConfig) -> int:
    """Return the configured concurrency for test rollouts."""
    return env_config.test_episode_concurrency or env_config.episode_concurrency


def _test_episodes_per_iteration(env_config: EnvironmentConfig) -> int:
    """Return the configured episode count for test rollouts."""
    return env_config.test_episodes_per_iteration or env_config.episodes_per_iteration


def _test_episode_timeout(env_config: EnvironmentConfig) -> Optional[float]:
    """Return the configured timeout for test rollouts."""
    if env_config.test_episode_timeout is not None:
        return env_config.test_episode_timeout
    return env_config.episode_timeout


def _database_name_from_url(url: str) -> str:
    """Extract the database name from a Postgres URL."""
    db_name = urlparse(url).path.lstrip("/")
    if not db_name:
        raise ValueError(f"Database URL is missing a database path: {url}")
    return db_name


def _replace_database_name(url: str, db_name: str) -> str:
    """Return a copy of the URL with the database path replaced."""
    parsed = urlparse(url)
    return urlunparse(parsed._replace(path=f"/{db_name}"))


def _scoped_database_name(base_name: str, scope: str, max_length: int = 63) -> str:
    """Build a run-scoped database name within Postgres' length limit."""
    suffix = f"_{scope}"
    if len(base_name) + len(suffix) <= max_length:
        return f"{base_name}{suffix}"
    return f"{base_name[: max_length - len(suffix)]}{suffix}"


def _should_scope_db() -> bool:
    """Check if per-run database scoping is enabled (default: yes)."""
    return os.environ.get("TENSORZERO_SCOPE_DB", "1").strip() in ("1", "true", "yes")


def _build_primary_gateway_urls(run_id: uuid.UUID) -> dict[str, str]:
    """Build isolated per-run database URLs for the primary (train) gateway."""
    base_pg_url = os.environ.get("TENSORZERO_POSTGRES_URL", "")
    if not base_pg_url:
        raise RuntimeError("TENSORZERO_POSTGRES_URL is required")

    if not _should_scope_db():
        return {"postgres_url": base_pg_url}

    scope = run_id.hex[-12:]
    base_db = _database_name_from_url(base_pg_url)
    scoped_db = _scoped_database_name(base_db, scope)
    return {
        "postgres_url": _replace_database_name(base_pg_url, scoped_db),
    }


def _build_test_gateway_urls(primary_pg_url: str, run_id: uuid.UUID) -> dict[str, str]:
    """Build isolated per-run database URLs for persisted test rollouts."""
    base_test_pg_url = os.environ.get("TENSORZERO_TEST_POSTGRES_URL") or (
        _replace_database_name(primary_pg_url, f"{_database_name_from_url(primary_pg_url)}_test")
    )

    if not _should_scope_db():
        return {"postgres_url": base_test_pg_url}

    scope = run_id.hex[-12:]
    test_pg_db = _scoped_database_name(_database_name_from_url(base_test_pg_url), scope)
    return {
        "postgres_url": _replace_database_name(base_test_pg_url, test_pg_db),
    }


def _write_gateway_runtime_metadata(
    output_dir: Path,
    *,
    gateway_url: str,
    postgres_url: str,
) -> None:
    """Persist the runtime database mapping for later inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "runtime.json"
    metadata = {
        "gateway_url": gateway_url,
        "postgres_database": _database_name_from_url(postgres_url),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def _write_json_artifact(path: Path, payload: Any) -> None:
    """Write a JSON artifact with stable formatting for debugging."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _persist_session_debug_artifacts(
    *,
    work_dir: Path,
    iteration: int,
    initial_message: str,
    session_result: Any,
) -> None:
    """Persist the autopilot session inputs/outputs for postmortem debugging."""
    session_dir = work_dir / "autopilot" / f"iteration_{iteration:03d}"
    session_dir.mkdir(parents=True, exist_ok=True)

    (session_dir / "initial_message.txt").write_text(initial_message)
    _write_json_artifact(
        session_dir / "session_result.json",
        {
            "session_id": session_result.session_id,
            "final_status": session_result.final_status,
            "turns": session_result.turns,
            "error": session_result.error,
            "config_writes_count": len(session_result.config_writes),
            "raw_config_writes_count": len(session_result.raw_config_writes),
        },
    )
    _write_json_artifact(
        session_dir / "config_writes.flattened.json",
        session_result.config_writes,
    )
    _write_json_artifact(
        session_dir / "config_writes.raw.json",
        session_result.raw_config_writes,
    )


async def _close_gateway_client(gateway: AsyncTensorZeroGateway | None) -> None:
    """Close an HTTP gateway client if it exists."""
    if gateway is None:
        return
    try:
        await gateway.close()
    except Exception:
        logger.debug("Failed to close gateway client", exc_info=True)


async def _restart_managed_gateways(
    primary_proc: GatewayProcess,
    *,
    primary_env: dict[str, str],
    test_proc: GatewayProcess | None = None,
    test_env: dict[str, str] | None = None,
) -> None:
    """Restart the active embedded gateways from disk config."""
    if test_proc is not None and test_proc.is_running:
        await test_proc.stop()
    if primary_proc.is_running:
        await primary_proc.stop()

    await primary_proc.start(env_overrides=primary_env)
    if test_proc is not None:
        await test_proc.start(env_overrides=test_env)


async def _apply_session_config_writes_and_restart(
    *,
    config_glob: str,
    config_writes: list[dict[str, Any]],
    gateway_proc: GatewayProcess,
    gateway_env: dict[str, str],
    gateway: AsyncTensorZeroGateway | None,
    test_gateway_proc: GatewayProcess | None = None,
    test_gateway_env: dict[str, str] | None = None,
    test_gateway: AsyncTensorZeroGateway | None = None,
) -> tuple[AsyncTensorZeroGateway, AsyncTensorZeroGateway | None]:
    """Apply config writes, sync them, and restart all managed gateways.

    This intentionally does not rollback the written config on restart failure.
    If the updated config is invalid, the failure should remain visible on disk
    for debugging rather than being silently masked by restoration logic.
    """
    await apply_config_writes(
        config_writes=config_writes,
        config_glob=config_glob,
    )

    await _close_gateway_client(test_gateway)
    await _close_gateway_client(gateway)
    await _restart_managed_gateways(
        gateway_proc,
        primary_env=gateway_env,
        test_proc=test_gateway_proc,
        test_env=test_gateway_env,
    )
    primary_gateway = await AsyncTensorZeroGateway.build_http(
        gateway_url=gateway_proc.url,
    )
    restarted_test_gateway = None
    if test_gateway_proc is not None:
        restarted_test_gateway = await AsyncTensorZeroGateway.build_http(
            gateway_url=test_gateway_proc.url,
        )
    return primary_gateway, restarted_test_gateway


async def _start_test_gateway(
    *,
    run_id: uuid.UUID,
    primary_pg_url: str,
    eval_config: EvalConfig,
    config_file: str,
    work_dir: Path,
) -> tuple[GatewayProcess | None, AsyncTensorZeroGateway | None, dict[str, str] | None]:
    """Start a second embedded gateway for persisted test rollouts."""
    test_urls = _build_test_gateway_urls(primary_pg_url, run_id)
    await _reset_gateway_postgres(test_urls["postgres_url"])
    test_gateway_env = _build_gateway_env(
        eval_config,
        postgres_url=test_urls["postgres_url"],
    )
    test_gateway_proc = GatewayProcess(
        binary_path=eval_config.infra.gateway_binary_path,
        config_path=config_file,
        port=eval_config.infra.test_gateway_port,
        startup_timeout=eval_config.infra.gateway_startup_timeout,
        shutdown_timeout=eval_config.infra.gateway_shutdown_timeout,
        log_dir=work_dir / "gateway" / "test",
    )
    await test_gateway_proc.start(env_overrides=test_gateway_env)
    _write_gateway_runtime_metadata(
        work_dir / "gateway" / "test",
        gateway_url=test_gateway_proc.url,
        postgres_url=test_urls["postgres_url"],
    )
    logger.info(
        "Persisting test rollouts via %s (Postgres=%s)",
        test_gateway_proc.url,
        _database_name_from_url(test_urls["postgres_url"]),
    )
    test_gateway = await AsyncTensorZeroGateway.build_http(
        gateway_url=test_gateway_proc.url,
    )
    return test_gateway_proc, test_gateway, test_gateway_env


async def run_environment(
    env_config: EnvironmentConfig,
    eval_config: EvalConfig,
    recorder: JsonResultRecorder,
    work_dir: Path,
    snapshot_path: Path | None = None,
    seed: Optional[int] = None,
) -> uuid.UUID:
    """Run the full evaluation loop for a single environment.

    Args:
        env_config: Configuration for this environment.
        eval_config: Top-level eval configuration.
        recorder: Result recorder for persisting metrics.
        work_dir: Working directory for config files and artifacts.
        snapshot_path: If provided, restore baseline from this snapshot
            instead of running the baseline rollout.
        seed: Optional RNG seed for deterministic task selection.

    Returns:
        The eval run UUID.
    """
    logger.info("=" * 60)
    logger.info("Starting evaluation for environment: %s (seed=%s)", env_config.name, seed)
    logger.info("=" * 60)

    # Create eval run record
    run_id = await recorder.create_run(
        environment_name=env_config.name,
        autopilot_target=eval_config.autopilot_target.kind,
        autopilot_base_url=eval_config.autopilot_target.base_url or "default",
        eval_config=env_config.model_dump(),
        seed=seed,
    )

    gateway = None
    test_gateway = None
    interlocutor = None
    gateway_proc = None
    test_gateway_proc = None
    test_gateway_env = None

    try:
        # Scope the primary (train) gateway database per-run
        primary_urls = _build_primary_gateway_urls(run_id)
        pg_url = primary_urls["postgres_url"]
        gateway_env = _build_gateway_env(eval_config, postgres_url=pg_url)

        # Track per-iteration metrics for the end-of-run summary
        iteration_summaries: list[dict[str, Any]] = []

        if snapshot_path is not None:
            # --- Restore from snapshot ---
            logger.info("Restoring baseline from snapshot: %s", snapshot_path)
            validate_snapshot(snapshot_path, env_config)

            # 1. Restore config
            config_dir = restore_config(snapshot_path, work_dir)
            config_file = str(config_dir / "tensorzero.toml")
            config_glob = f"{config_dir}/**/*.toml"

            # 2-3. Reset databases
            await _reset_gateway_postgres(pg_url)

            # 4. Start gateway (runs migrations, creates empty CH tables)
            gateway_proc = GatewayProcess(
                binary_path=eval_config.infra.gateway_binary_path,
                config_path=config_file,
                port=eval_config.infra.gateway_port,
                startup_timeout=eval_config.infra.gateway_startup_timeout,
                shutdown_timeout=eval_config.infra.gateway_shutdown_timeout,
                log_dir=work_dir / "gateway",
            )
            await gateway_proc.start(env_overrides=gateway_env)

            # 5. Import Postgres data from snapshot
            await import_all_tables(pg_url, snapshot_path / "postgres")

            # 6. Create HTTP gateway client
            gateway = await AsyncTensorZeroGateway.build_http(
                gateway_url=gateway_proc.url,
            )
            (
                test_gateway_proc,
                test_gateway,
                test_gateway_env,
            ) = await _start_test_gateway(
                run_id=run_id,
                primary_pg_url=pg_url,
                eval_config=eval_config,
                config_file=config_file,
                work_dir=work_dir,
            )

            # 7. Load baseline data from snapshot
            baseline_data, baseline_stats, _metadata = load_baseline_data(snapshot_path)
            train_metrics = baseline_data["train"]
            test_metrics = baseline_data["test"]
            train_stats = baseline_stats["train"]
            test_stats = baseline_stats["test"]

            await recorder.record_iteration(
                run_id=run_id,
                iteration_number=0,
                phase="baseline_restored",
                num_episodes=train_stats["total"] + test_stats["total"],
                num_succeeded=train_stats["succeeded"] + test_stats["succeeded"],
                num_failed=train_stats["failed"] + test_stats["failed"],
                error_counts=_merge_error_counts(train_stats, test_stats),
                metrics={"train": train_metrics, "test": test_metrics},
                active_variants=_extract_active_variants(test_metrics),
            )
            iteration_summaries.append(
                {
                    "iteration": 0,
                    "phase": "baseline_restored",
                    "train": train_metrics,
                    "test": test_metrics,
                    "active_variants": _extract_active_variants(test_metrics),
                }
            )
        else:
            # --- Normal baseline phase ---
            # 1. Generate T0 config from llmgym environment
            config_dir = write_initial_env_config(
                env_name=env_config.effective_llmgym_env,
                output_path=str(work_dir),
                model_name=env_config.initial_model,
                function_description=env_config.function_description,
                env_config_extra=env_config.env_config or None,
            )
            config_file = str(config_dir / "tensorzero.toml")
            config_glob = f"{config_dir}/**/*.toml"

            # 2. Reset gateway Postgres for a clean run
            await _reset_gateway_postgres(pg_url)

            # 3. Start gateway binary as subprocess
            gateway_proc = GatewayProcess(
                binary_path=eval_config.infra.gateway_binary_path,
                config_path=config_file,
                port=eval_config.infra.gateway_port,
                startup_timeout=eval_config.infra.gateway_startup_timeout,
                shutdown_timeout=eval_config.infra.gateway_shutdown_timeout,
                log_dir=work_dir / "gateway",
            )

            await gateway_proc.start(env_overrides=gateway_env)

            # 4. Create HTTP gateway client
            gateway = await AsyncTensorZeroGateway.build_http(
                gateway_url=gateway_proc.url,
            )
            (
                test_gateway_proc,
                test_gateway,
                test_gateway_env,
            ) = await _start_test_gateway(
                run_id=run_id,
                primary_pg_url=pg_url,
                eval_config=eval_config,
                config_file=config_file,
                work_dir=work_dir,
            )

            # Write runtime metadata for the primary gateway
            _write_gateway_runtime_metadata(
                work_dir / "gateway",
                gateway_url=gateway_proc.url,
                postgres_url=pg_url,
            )

            # 5. Run baseline rollouts (train + test)
            logger.info(
                "Running baseline train rollout (%d episodes)",
                env_config.episodes_per_iteration,
            )
            train_stats = await run_rollout(
                gateway=gateway,
                env_name=env_config.effective_llmgym_env,
                num_episodes=env_config.episodes_per_iteration,
                concurrency=env_config.episode_concurrency,
                dryrun=False,
                output_dir=_rollout_output_dir(
                    work_dir,
                    iteration=0,
                    phase="baseline",
                    split="train",
                ),
                task_split=env_config.task_split,
                env_config_extra=env_config.env_config or None,
                episode_timeout=env_config.episode_timeout,
                seed=seed,
            )
            train_metrics = await _collect_metrics(gateway_proc.url, env_config.metric_name, env_config.function_name)

            logger.info(
                "Running baseline test rollout (%d episodes)",
                _test_episodes_per_iteration(env_config),
            )
            test_stats = await run_rollout(
                gateway=test_gateway or gateway,
                env_name=env_config.effective_llmgym_env,
                num_episodes=_test_episodes_per_iteration(env_config),
                concurrency=_test_episode_concurrency(env_config),
                dryrun=False,
                output_dir=_rollout_output_dir(
                    work_dir,
                    iteration=0,
                    phase="baseline",
                    split="test",
                ),
                task_split="test",
                env_config_extra=env_config.env_config or None,
                episode_timeout=_test_episode_timeout(env_config),
                seed=seed,
                unique_tasks=True,
            )
            test_metrics = test_stats["variant_metrics"]

            await recorder.record_iteration(
                run_id=run_id,
                iteration_number=0,
                phase="baseline",
                num_episodes=train_stats["total"] + test_stats["total"],
                num_succeeded=train_stats["succeeded"] + test_stats["succeeded"],
                num_failed=train_stats["failed"] + test_stats["failed"],
                error_counts=_merge_error_counts(train_stats, test_stats),
                metrics={"train": train_metrics, "test": test_metrics},
                active_variants=_extract_active_variants(test_metrics),
            )
            iteration_summaries.append(
                {
                    "iteration": 0,
                    "phase": "baseline",
                    "train": train_metrics,
                    "test": test_metrics,
                    "active_variants": _extract_active_variants(test_metrics),
                }
            )

        # 5. Set up interlocutor
        try:
            interlocutor = await Interlocutor.create(eval_config.interlocutor.config_file)
        except FileNotFoundError:
            logger.warning("Interlocutor config not found, running without interlocutor")

        # 7. Iteration loop
        for iteration in range(1, env_config.num_iterations + 1):
            logger.info("-" * 60)
            logger.info("Iteration %d / %d", iteration, env_config.num_iterations)
            logger.info("-" * 60)

            # a. Run autopilot session
            session_mgr = AutopilotSessionManager(
                gateway_url=gateway_proc.url,
                interlocutor=interlocutor,
                max_turns=env_config.autopilot_max_turns,
                timeout=env_config.autopilot_session_timeout,
            )

            initial_message = _build_initial_message(env_config)
            session_result = await session_mgr.run_session(
                initial_message=initial_message,
            )
            _persist_session_debug_artifacts(
                work_dir=work_dir,
                iteration=iteration,
                initial_message=initial_message,
                session_result=session_result,
            )

            logger.info(
                "Autopilot session %s finished: status=%s, turns=%d, writes=%d",
                session_result.session_id,
                session_result.final_status,
                session_result.turns,
                len(session_result.config_writes),
            )

            # b. Apply config writes and restart gateway
            if session_result.config_writes:
                gateway, test_gateway = await _apply_session_config_writes_and_restart(
                    config_glob=config_glob,
                    config_writes=session_result.config_writes,
                    gateway_proc=gateway_proc,
                    gateway_env=gateway_env,
                    gateway=gateway,
                    test_gateway_proc=test_gateway_proc,
                    test_gateway_env=test_gateway_env,
                    test_gateway=test_gateway,
                )

            # d. Run post-autopilot rollouts. For snapshot-backed single-iteration
            # runs, skip the non-dryrun train rollout because there is no later
            # iteration that would consume the newly written observability data.
            skip_post_autopilot_train = snapshot_path is not None and env_config.num_iterations == 1
            if skip_post_autopilot_train:
                logger.info("Skipping post-autopilot train rollout for snapshot-backed single-iteration run")
                rollout_train = {
                    "total": 0,
                    "succeeded": 0,
                    "failed": 0,
                    "error_counts": {},
                }
                iter_train_metrics = None
            else:
                logger.info(
                    "Running post-autopilot train rollout (%d episodes)",
                    env_config.episodes_per_iteration,
                )
                rollout_train = await run_rollout(
                    gateway=gateway,
                    env_name=env_config.effective_llmgym_env,
                    num_episodes=env_config.episodes_per_iteration,
                    concurrency=env_config.episode_concurrency,
                    dryrun=False,
                    output_dir=_rollout_output_dir(
                        work_dir,
                        iteration=iteration,
                        phase="post_autopilot",
                        split="train",
                    ),
                    task_split=env_config.task_split,
                    env_config_extra=env_config.env_config or None,
                    episode_timeout=env_config.episode_timeout,
                    seed=seed,
                )
                iter_train_metrics = await _collect_metrics(
                    gateway_proc.url, env_config.metric_name, env_config.function_name
                )

            logger.info(
                "Running post-autopilot test rollout (%d episodes)",
                _test_episodes_per_iteration(env_config),
            )
            rollout_test = await run_rollout(
                gateway=test_gateway or gateway,
                env_name=env_config.effective_llmgym_env,
                num_episodes=_test_episodes_per_iteration(env_config),
                concurrency=_test_episode_concurrency(env_config),
                dryrun=False,
                output_dir=_rollout_output_dir(
                    work_dir,
                    iteration=iteration,
                    phase="post_autopilot",
                    split="test",
                ),
                task_split="test",
                env_config_extra=env_config.env_config or None,
                episode_timeout=_test_episode_timeout(env_config),
                seed=seed,
                unique_tasks=True,
            )
            iter_test_metrics = rollout_test["variant_metrics"]

            await recorder.record_iteration(
                run_id=run_id,
                iteration_number=iteration,
                phase="post_autopilot",
                num_episodes=rollout_train["total"] + rollout_test["total"],
                num_succeeded=rollout_train["succeeded"] + rollout_test["succeeded"],
                num_failed=rollout_train["failed"] + rollout_test["failed"],
                error_counts=_merge_error_counts(rollout_train, rollout_test),
                metrics={"train": iter_train_metrics, "test": iter_test_metrics},
                active_variants=_extract_active_variants(iter_test_metrics),
                autopilot_session_id=session_result.session_id,
                autopilot_turns=session_result.turns,
                autopilot_config_writes=len(session_result.config_writes),
                autopilot_final_status=session_result.final_status,
            )
            iteration_summaries.append(
                {
                    "iteration": iteration,
                    "phase": "post_autopilot",
                    "train": iter_train_metrics,
                    "test": iter_test_metrics,
                    "active_variants": _extract_active_variants(iter_test_metrics),
                    "autopilot_turns": session_result.turns,
                    "autopilot_config_writes": len(session_result.config_writes),
                    "autopilot_status": session_result.final_status,
                }
            )

        # 8. Print run summary and mark as completed
        _log_run_summary(
            run_id=run_id,
            env_name=env_config.effective_llmgym_env,
            metric_name=env_config.metric_name,
            summaries=iteration_summaries,
        )
        await recorder.finish_run(run_id, status="completed")
        logger.info("Evaluation run %s completed", run_id)

    except Exception:
        logger.exception("Evaluation run %s failed", run_id)
        await recorder.finish_run(run_id, status="failed")
        raise

    finally:
        # Clean up
        if gateway is not None:
            await _close_gateway_client(gateway)
        if test_gateway is not None:
            await _close_gateway_client(test_gateway)
        if interlocutor is not None:
            try:
                await interlocutor.close()
            except Exception:
                pass
        if test_gateway_proc is not None and test_gateway_proc.is_running:
            await test_gateway_proc.stop()
        if gateway_proc is not None and gateway_proc.is_running:
            await gateway_proc.stop()

    return run_id


async def run_snapshot(
    env_config: EnvironmentConfig,
    eval_config: EvalConfig,
    work_dir: Path,
    snapshot_dir: Path,
) -> Path:
    """Run the baseline phase only and create a snapshot.

    This runs config generation, database reset, gateway startup, and baseline
    rollouts (train + test), then exports the full state to a snapshot directory.

    Args:
        env_config: Configuration for this environment.
        eval_config: Top-level eval configuration.
        work_dir: Working directory for config files.
        snapshot_dir: Base directory for snapshots.

    Returns:
        Path to the created snapshot directory.
    """
    from datetime import datetime, timezone

    logger.info("=" * 60)
    logger.info("Creating snapshot for environment: %s", env_config.name)
    logger.info("=" * 60)

    gateway = None
    test_gateway = None
    gateway_proc = None
    test_gateway_proc = None

    try:
        # 1. Generate T0 config
        config_dir = write_initial_env_config(
            env_name=env_config.effective_llmgym_env,
            output_path=str(work_dir),
            model_name=env_config.initial_model,
            function_description=env_config.function_description,
            env_config_extra=env_config.env_config or None,
        )
        config_file = str(config_dir / "tensorzero.toml")

        # 2. Reset databases — use a scoped DB for the snapshot too
        snapshot_run_id = uuid7()
        primary_urls = _build_primary_gateway_urls(snapshot_run_id)
        pg_url = primary_urls["postgres_url"]
        gateway_env = _build_gateway_env(eval_config, postgres_url=pg_url)
        await _reset_gateway_postgres(pg_url)

        # 3. Start gateway
        gateway_proc = GatewayProcess(
            binary_path=eval_config.infra.gateway_binary_path,
            config_path=config_file,
            port=eval_config.infra.gateway_port,
            startup_timeout=eval_config.infra.gateway_startup_timeout,
            shutdown_timeout=eval_config.infra.gateway_shutdown_timeout,
            log_dir=work_dir / "gateway",
        )
        await gateway_proc.start(env_overrides=gateway_env)

        # 4. Create gateway client
        gateway = await AsyncTensorZeroGateway.build_http(
            gateway_url=gateway_proc.url,
        )
        test_gateway_proc, test_gateway, _test_gateway_env = await _start_test_gateway(
            run_id=snapshot_run_id,
            primary_pg_url=pg_url,
            eval_config=eval_config,
            config_file=config_file,
            work_dir=work_dir,
        )

        # 5. Run baseline train rollout
        logger.info(
            "Running baseline train rollout (%d episodes)",
            env_config.episodes_per_iteration,
        )
        train_stats = await run_rollout(
            gateway=gateway,
            env_name=env_config.effective_llmgym_env,
            num_episodes=env_config.episodes_per_iteration,
            concurrency=env_config.episode_concurrency,
            dryrun=False,
            output_dir=config_dir.parent / "rollouts" / "baseline" / "train",
            task_split=env_config.task_split,
            env_config_extra=env_config.env_config or None,
            episode_timeout=env_config.episode_timeout,
        )
        train_metrics = await _collect_metrics(gateway_proc.url, env_config.metric_name, env_config.function_name)

        # 6. Run baseline test rollout
        logger.info(
            "Running baseline test rollout (%d episodes)",
            _test_episodes_per_iteration(env_config),
        )
        test_stats = await run_rollout(
            gateway=test_gateway or gateway,
            env_name=env_config.effective_llmgym_env,
            num_episodes=_test_episodes_per_iteration(env_config),
            concurrency=_test_episode_concurrency(env_config),
            dryrun=False,
            output_dir=config_dir.parent / "rollouts" / "baseline" / "test",
            task_split="test",
            env_config_extra=env_config.env_config or None,
            episode_timeout=_test_episode_timeout(env_config),
            unique_tasks=True,
        )
        test_metrics = test_stats["variant_metrics"]

        # 7. Create snapshot
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snap_path = snapshot_dir / env_config.name / timestamp

        # Strip non-serializable episode_results before persisting
        serializable_train_stats = {k: v for k, v in train_stats.items() if k != "episode_results"}
        serializable_test_stats = {k: v for k, v in test_stats.items() if k != "episode_results"}

        result = await create_snapshot(
            snapshot_dir=snap_path,
            config_dir=config_dir,
            postgres_url=pg_url,
            env_config=env_config,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            train_stats=serializable_train_stats,
            test_stats=serializable_test_stats,
        )

        logger.info("Snapshot created at: %s", result)
        return result

    finally:
        if gateway is not None:
            await _close_gateway_client(gateway)
        if test_gateway is not None:
            await _close_gateway_client(test_gateway)
        if test_gateway_proc is not None and test_gateway_proc.is_running:
            await test_gateway_proc.stop()
        if gateway_proc is not None and gateway_proc.is_running:
            await gateway_proc.stop()


def _build_gateway_env(
    eval_config: EvalConfig,
    *,
    postgres_url: str | None = None,
) -> dict[str, str]:
    """Build environment variables for the gateway subprocess."""
    env: dict[str, str] = {}

    # Postgres URL for gateway (not eval results)
    pg_url = postgres_url if postgres_url is not None else os.environ.get("TENSORZERO_POSTGRES_URL", "")
    if pg_url:
        env["TENSORZERO_POSTGRES_URL"] = pg_url

    # Autopilot connection
    try:
        env["TENSORZERO_AUTOPILOT_API_KEY"] = eval_config.autopilot_target.api_key
    except ValueError:
        pass

    if eval_config.autopilot_target.base_url:
        env["TENSORZERO_AUTOPILOT_BASE_URL"] = eval_config.autopilot_target.base_url

    return env


def _log_run_summary(
    run_id: uuid.UUID,
    env_name: str,
    metric_name: str,
    summaries: list[dict[str, Any]],
) -> None:
    """Print a summary table of per-variant performance across all iterations."""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 72)
    lines.append(f"  Run Summary: {env_name}  (run_id={run_id})")
    lines.append(f"  Metric: {metric_name}")
    lines.append("=" * 72)

    for s in summaries:
        iteration = s["iteration"]
        phase = s["phase"]
        variants = s["active_variants"]

        header = f"  Iteration {iteration} ({phase})"
        if phase not in ("baseline", "baseline_restored"):
            ap_turns = s.get("autopilot_turns", "?")
            ap_writes = s.get("autopilot_config_writes", "?")
            ap_status = s.get("autopilot_status", "?")
            header += f"  [autopilot: {ap_turns} turns, {ap_writes} writes, {ap_status}]"
        lines.append("")
        lines.append(header)
        lines.append(f"  Variants: {variants}")
        lines.append("  " + "-" * 68)

        # Print test metrics (the clean evaluation)
        lines.append("  TEST:")
        test = s.get("test")
        if test and isinstance(test, dict):
            metric_data = test.get(metric_name, {})
            variant_stats = metric_data.get("variants", {})
            if variant_stats:
                for vname in sorted(variant_stats):
                    vs = variant_stats[vname]
                    mean = vs.get("mean", 0)
                    n_ep = vs.get("n_episodes", vs.get("n", "?"))
                    ci = vs.get("ci_error")
                    ci_str = f" +/-{ci:.3f}" if ci is not None else ""
                    lines.append(f"    {vname:30s}  {mean:.3f}{ci_str}  ({n_ep} episodes)")
            else:
                lines.append("    (no variant data)")
        else:
            lines.append("    (no test metrics)")

        # Print train metrics
        lines.append("  TRAIN:")
        train = s.get("train")
        if train and isinstance(train, dict):
            metric_data = train.get(metric_name, {})
            variant_stats = metric_data.get("variants", {})
            if variant_stats:
                for vname in sorted(variant_stats):
                    vs = variant_stats[vname]
                    mean = vs.get("mean", 0)
                    n_ep = vs.get("n_episodes", vs.get("n", "?"))
                    ci = vs.get("ci_error")
                    ci_str = f" +/-{ci:.3f}" if ci is not None else ""
                    lines.append(f"    {vname:30s}  {mean:.3f}{ci_str}  ({n_ep} episodes)")
            else:
                lines.append("    (no variant data)")
        else:
            lines.append("    (no train metrics)")

    lines.append("")
    lines.append("=" * 72)

    logger.info("\n".join(lines))


async def _reset_gateway_postgres(postgres_url: str) -> None:
    """Drop and recreate the gateway Postgres database for a clean run."""
    parsed = urlparse(postgres_url)
    db_name = parsed.path.lstrip("/")

    # Connect to maintenance database instead of the target
    maintenance_url = urlunparse(parsed._replace(path="/postgres"))

    logger.info("Resetting gateway Postgres database: %s", db_name)
    conn = await asyncpg.connect(maintenance_url)
    try:
        # Terminate existing connections to the target database
        await conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = $1 AND pid <> pg_backend_pid()",
            db_name,
        )
        await conn.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
        await conn.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await conn.close()
    logger.info("Gateway Postgres database %s reset complete", db_name)


def _extract_active_variants(
    variant_metrics: dict[str, Any],
) -> list[str]:
    """Extract sorted list of variant names from variant_metrics."""
    variants: set[str] = set()
    for metric_data in variant_metrics.values():
        if isinstance(metric_data, dict):
            variants.update(metric_data.get("variants", {}).keys())
    return sorted(variants)


async def _collect_metrics(
    gateway_url: str,
    metric_name: str,
    function_name: str,
) -> Optional[dict[str, Any]]:
    """Collect per-variant metric statistics via the gateway's internal API.

    Calls GET /internal/functions/{function_name}/variant_performances
    which returns performance stats (count, avg, stdev, ci_error) grouped
    by variant, avoiding the cumulative-across-variants bug.
    """
    encoded_fn = quote(function_name, safe="")
    url = (
        f"{gateway_url}/internal/functions/{encoded_fn}"
        f"/variant_performances?metric_name={quote(metric_name, safe='')}"
        f"&time_window=cumulative"
    )

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()

        performances = data.get("performances", [])
        if not performances:
            logger.info("No variant performances found for %s/%s", function_name, metric_name)
            return {metric_name: {"variants": {}, "n": 0}}

        variants: dict[str, Any] = {}
        total_n = 0
        for row in performances:
            vname = row["variant_name"]
            n = int(row["count"])
            total_n += n
            variants[vname] = {
                "mean": row["avg_metric"],
                "std": row.get("stdev"),
                "n": n,
                "ci_error": row.get("ci_error"),
            }

        metrics = {metric_name: {"variants": variants, "n": total_n}}
        logger.info("Collected variant metrics: %s", metrics)
        return metrics

    except Exception:
        logger.exception("Failed to collect metrics from gateway")

    return None
