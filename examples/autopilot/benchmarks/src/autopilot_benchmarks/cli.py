"""CLI entry point for the benchmark harness.

Usage:
    autopilot-benchmark run --config configs/ner.yaml [--env ner_conllpp_v0]
    autopilot-benchmark snapshot --config configs/ner.yaml --env ner_conllpp_v0
"""

import asyncio
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click

from autopilot_benchmarks.config import EnvironmentConfig, EvalConfig
from autopilot_benchmarks.orchestrator import run_environment, run_snapshot
from autopilot_benchmarks.results.recorder import JsonResultRecorder

logger = logging.getLogger("autopilot_benchmarks")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


@click.group()
def main() -> None:
    """Autopilot benchmark harness."""
    pass


@main.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the eval config YAML file.",
)
@click.option(
    "--env",
    "env_name",
    default=None,
    help="Run only this environment (by name). Runs all if omitted.",
)
@click.option(
    "--work-dir",
    default=".",
    type=click.Path(),
    help="Working directory for generated config files.",
)
@click.option(
    "--snapshot",
    "snapshot",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Restore baseline from this snapshot directory instead of running it.",
)
@click.option(
    "--num-iterations",
    "num_iterations",
    default=None,
    type=click.INT,
    help="Override number of autopilot iterations. Defaults to 1 when --snapshot is used.",
)
@click.option(
    "--episodes",
    "episodes",
    default=None,
    type=click.INT,
    help="Override episodes_per_iteration (and test_episodes_per_iteration) for all envs.",
)
@click.option(
    "--seed",
    "seed",
    default=None,
    type=click.INT,
    help="RNG seed for deterministic task selection.",
)
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def run(
    config_path: str,
    env_name: Optional[str],
    work_dir: str,
    snapshot: Optional[str],
    num_iterations: Optional[int],
    episodes: Optional[int],
    seed: Optional[int],
    verbose: bool,
) -> None:
    """Run evaluation loop for configured environments."""
    _setup_logging(verbose)

    config = EvalConfig.from_yaml(config_path)
    logger.info("Loaded config from %s", config_path)

    # Filter environments if requested
    environments = config.environments
    if env_name:
        environments = [e for e in environments if e.name == env_name]
        if not environments:
            logger.error("Environment '%s' not found in config", env_name)
            sys.exit(1)

    # Apply num_iterations override
    effective_iterations = num_iterations
    if effective_iterations is None and snapshot is not None:
        effective_iterations = 1
    if effective_iterations is not None:
        environments = [e.model_copy(update={"num_iterations": effective_iterations}) for e in environments]

    # Apply episodes override
    if episodes is not None:
        environments = [
            e.model_copy(
                update={
                    "episodes_per_iteration": episodes,
                    "test_episodes_per_iteration": episodes,
                }
            )
            for e in environments
        ]

    logger.info(
        "Will evaluate %d environment(s): %s",
        len(environments),
        [e.name for e in environments],
    )

    snapshot_path = Path(snapshot) if snapshot else None
    asyncio.run(
        _run_async(
            config,
            environments,
            Path(work_dir),
            snapshot_path,
            seed=seed,
        )
    )


async def _run_async(
    config: EvalConfig,
    environments: list,
    work_dir: Path,
    snapshot_path: Path | None = None,
    seed: int | None = None,
) -> None:
    """Async entry point for running evaluations."""
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    seed_suffix = f"_seed{seed}" if seed is not None else ""

    env_work_dirs: list[Path] = []
    run_ids: list[uuid.UUID] = []

    for env_config in environments:
        env_work_dir = work_dir / env_config.name / f"{run_timestamp}{seed_suffix}"
        env_work_dir.mkdir(parents=True, exist_ok=True)

        recorder = JsonResultRecorder(env_work_dir)

        run_id = await run_environment(
            env_config=env_config,
            eval_config=config,
            recorder=recorder,
            work_dir=env_work_dir,
            snapshot_path=snapshot_path,
            seed=seed,
        )
        run_ids.append(run_id)
        env_work_dirs.append(env_work_dir)
        logger.info(
            "Environment %s completed. Run ID: %s (seed=%s)",
            env_config.name,
            run_id,
            seed,
        )

    for d in env_work_dirs:
        logger.info("Configs written to: %s", d.resolve())


@main.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the eval config YAML file.",
)
@click.option(
    "--env",
    "env_name",
    default=None,
    help="Environment name to create a snapshot for. Snapshots all if omitted.",
)
@click.option(
    "--work-dir",
    default=".",
    type=click.Path(),
    help="Working directory for generated config files.",
)
@click.option(
    "--snapshot-dir",
    default="snapshots",
    type=click.Path(),
    help="Base directory for storing snapshots.",
)
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def snapshot(
    config_path: str,
    env_name: Optional[str],
    work_dir: str,
    snapshot_dir: str,
    verbose: bool,
) -> None:
    """Create a baseline snapshot (runs baseline only, then exports state)."""
    _setup_logging(verbose)

    config = EvalConfig.from_yaml(config_path)
    logger.info("Loaded config from %s", config_path)

    if env_name:
        environments = [e for e in config.environments if e.name == env_name]
        if not environments:
            logger.error("Environment '%s' not found in config", env_name)
            sys.exit(1)
    else:
        environments = config.environments

    logger.info(
        "Will snapshot %d environment(s): %s",
        len(environments),
        [e.name for e in environments],
    )

    asyncio.run(_snapshot_all_async(environments, config, Path(work_dir), Path(snapshot_dir)))


async def _snapshot_all_async(
    environments: list[EnvironmentConfig],
    eval_config: EvalConfig,
    work_dir: Path,
    snapshot_dir: Path,
) -> None:
    """Async entry point for creating snapshots."""
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for env_config in environments:
        env_work_dir = work_dir / env_config.name / run_timestamp
        env_work_dir.mkdir(parents=True, exist_ok=True)

        result = await run_snapshot(
            env_config=env_config,
            eval_config=eval_config,
            work_dir=env_work_dir,
            snapshot_dir=snapshot_dir,
        )
        logger.info("Snapshot written to: %s", result.resolve())


if __name__ == "__main__":
    main()
