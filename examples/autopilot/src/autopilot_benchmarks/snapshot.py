"""Snapshot creation, validation, and restore for baseline state.

A snapshot captures everything needed to skip the baseline phase:
  - T0 config directory (TOML files, templates, schemas)
  - All Postgres tables as binary COPY files
  - Baseline metrics and rollout stats as JSON
  - Metadata for validation
"""

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from autopilot_benchmarks.config import EnvironmentConfig
from autopilot_benchmarks.infra.postgres_io import export_all_tables

logger = logging.getLogger(__name__)


@dataclass
class SnapshotMetadata:
    """Serializable metadata about a snapshot."""

    env_name: str
    function_name: str
    metric_name: str
    initial_model: str
    episodes_per_iteration: int
    task_split: str
    created_at: str
    env_config: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, data: str) -> "SnapshotMetadata":
        parsed = json.loads(data)
        # Backward compat: old snapshots may lack env_config
        if "env_config" not in parsed:
            parsed["env_config"] = {}
        return cls(**parsed)


async def create_snapshot(
    snapshot_dir: Path,
    config_dir: Path,
    postgres_url: str,
    env_config: EnvironmentConfig,
    train_metrics: Optional[dict[str, Any]],
    test_metrics: Optional[dict[str, Any]],
    train_stats: dict[str, Any],
    test_stats: dict[str, Any],
) -> Path:
    """Create a snapshot of the baseline state.

    Args:
        snapshot_dir: Directory to create the snapshot in.
        config_dir: Path to the generated T0 config directory.
        postgres_url: Postgres URL for exporting tables.
        env_config: Environment configuration.
        train_metrics: Baseline train metrics dict.
        test_metrics: Baseline test metrics dict.
        train_stats: Baseline train rollout stats.
        test_stats: Baseline test rollout stats.

    Returns:
        Path to the created snapshot directory.
    """
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Copy config directory
    dest_config = snapshot_dir / "config"
    if dest_config.exists():
        shutil.rmtree(dest_config)
    shutil.copytree(config_dir, dest_config)
    logger.info("Copied config to %s", dest_config)

    # Export Postgres tables
    pg_dir = snapshot_dir / "postgres"
    pg_dir.mkdir(parents=True, exist_ok=True)
    await export_all_tables(postgres_url, pg_dir)

    # Write baseline metrics
    metrics_path = snapshot_dir / "baseline_metrics.json"
    metrics_path.write_text(
        json.dumps({"train": train_metrics, "test": test_metrics}, indent=2)
    )

    # Write baseline stats
    stats_path = snapshot_dir / "baseline_stats.json"
    stats_path.write_text(
        json.dumps({"train": train_stats, "test": test_stats}, indent=2)
    )

    # Write metadata
    metadata = SnapshotMetadata(
        env_name=env_config.name,
        function_name=env_config.function_name,
        metric_name=env_config.metric_name,
        initial_model=env_config.initial_model,
        episodes_per_iteration=env_config.episodes_per_iteration,
        task_split=env_config.task_split,
        created_at=datetime.now(timezone.utc).isoformat(),
        env_config=env_config.env_config,
    )
    meta_path = snapshot_dir / "snapshot_metadata.json"
    meta_path.write_text(metadata.to_json())

    logger.info("Snapshot created at %s", snapshot_dir)
    return snapshot_dir


def validate_snapshot(
    snapshot_dir: Path,
    env_config: EnvironmentConfig,
) -> None:
    """Validate that a snapshot directory is complete and matches the environment.

    Args:
        snapshot_dir: Path to the snapshot directory.
        env_config: Environment config to validate against.

    Raises:
        FileNotFoundError: If required files/directories are missing.
        ValueError: If snapshot metadata doesn't match the environment config.
    """
    required = [
        snapshot_dir / "config",
        snapshot_dir / "postgres",
        snapshot_dir / "baseline_metrics.json",
        snapshot_dir / "baseline_stats.json",
        snapshot_dir / "snapshot_metadata.json",
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Snapshot missing required path: {path}")

    # Validate metadata matches env config
    meta_text = (snapshot_dir / "snapshot_metadata.json").read_text()
    metadata = SnapshotMetadata.from_json(meta_text)

    mismatches: list[str] = []
    # Accept match on either name or llmgym_env for backward compat with
    # snapshots created before the llmgym_env split (where harbor envs all
    # stored env_name="harbor_v0").
    if metadata.env_name != env_config.name and metadata.env_name != env_config.effective_llmgym_env:
        mismatches.append(
            f"env_name: snapshot={metadata.env_name}, config={env_config.name}"
        )
    if metadata.function_name != env_config.function_name:
        mismatches.append(
            f"function_name: snapshot={metadata.function_name}, config={env_config.function_name}"
        )
    if metadata.metric_name != env_config.metric_name:
        mismatches.append(
            f"metric_name: snapshot={metadata.metric_name}, config={env_config.metric_name}"
        )

    if mismatches:
        raise ValueError(
            f"Snapshot does not match environment config: {'; '.join(mismatches)}"
        )

    logger.info("Snapshot validated: %s", snapshot_dir)


def restore_config(
    snapshot_dir: Path,
    work_dir: Path,
) -> Path:
    """Restore the T0 config directory from a snapshot.

    Args:
        snapshot_dir: Path to the snapshot directory.
        work_dir: Working directory to copy config into.

    Returns:
        Path to the restored config directory.
    """
    src_config = snapshot_dir / "config"
    dest_config = work_dir / "config"

    if dest_config.exists():
        shutil.rmtree(dest_config)
    shutil.copytree(src_config, dest_config)

    logger.info("Restored config from snapshot to %s", dest_config)
    return dest_config


def load_baseline_data(
    snapshot_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], SnapshotMetadata]:
    """Load baseline metrics, stats, and metadata from a snapshot.

    Args:
        snapshot_dir: Path to the snapshot directory.

    Returns:
        Tuple of (metrics_dict, stats_dict, metadata).
        metrics_dict has keys "train" and "test".
        stats_dict has keys "train" and "test".
    """
    metrics = json.loads((snapshot_dir / "baseline_metrics.json").read_text())
    stats = json.loads((snapshot_dir / "baseline_stats.json").read_text())
    metadata = SnapshotMetadata.from_json(
        (snapshot_dir / "snapshot_metadata.json").read_text()
    )

    logger.info(
        "Loaded baseline data from snapshot: env=%s, created=%s",
        metadata.env_name,
        metadata.created_at,
    )
    return metrics, stats, metadata
