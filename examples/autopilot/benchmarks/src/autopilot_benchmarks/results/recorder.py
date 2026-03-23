"""Record eval run and iteration results as JSON files on disk."""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _uuid7_hex() -> str:
    """Generate a UUIDv7-like string (time-sortable). Falls back to uuid4 if tensorzero is unavailable."""
    try:
        from tensorzero.util import uuid7

        return str(uuid7())
    except ImportError:
        return str(uuid.uuid4())


def _json_default(obj: Any) -> Any:
    """Custom JSON serializer for types not handled by default."""
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _dump_json(data: Any) -> str:
    return json.dumps(data, indent=2, default=_json_default)


class JsonResultRecorder:
    """Writes eval run and iteration data as JSON files to a work directory.

    Drop-in replacement for the Postgres-backed ResultRecorder.
    All methods are async to match the original interface (the orchestrator awaits them),
    but they perform synchronous file I/O only.

    Files written:
        <work_dir>/results/run.json          -- run metadata
        <work_dir>/results/iterations.json   -- array of iteration records
    """

    def __init__(self, work_dir: str | Path):
        self._results_dir = Path(work_dir) / "results"
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._run_path = self._results_dir / "run.json"
        self._iterations_path = self._results_dir / "iterations.json"

    # ------------------------------------------------------------------
    # Public API — matches ResultRecorder method signatures
    # ------------------------------------------------------------------

    async def create_run(
        self,
        environment_name: str,
        autopilot_target: str,
        autopilot_base_url: str,
        eval_config: dict[str, Any],
        gateway_version: Optional[str] = None,
        seed: Optional[int] = None,
        group_id: Optional[uuid.UUID] = None,
        group_label: Optional[str] = None,
        version_metadata: Optional[dict[str, Any]] = None,
    ) -> uuid.UUID:
        """Create a new eval run record. Returns the run ID."""
        run_id_str = _uuid7_hex()
        run_id = uuid.UUID(run_id_str)

        run_data = {
            "id": run_id_str,
            "environment_name": environment_name,
            "autopilot_target": autopilot_target,
            "autopilot_base_url": autopilot_base_url,
            "gateway_version": gateway_version,
            "eval_config": eval_config,
            "seed": seed,
            "group_id": str(group_id) if group_id else None,
            "group_label": group_label,
            "version_metadata": version_metadata,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
        }

        self._run_path.write_text(_dump_json(run_data))

        # Initialize empty iterations file
        self._iterations_path.write_text(_dump_json([]))

        logger.info(
            "Created eval run %s for %s (seed=%s, group_id=%s)",
            run_id,
            environment_name,
            seed,
            group_id,
        )
        return run_id

    async def finish_run(self, run_id: uuid.UUID, status: str = "completed") -> None:
        """Mark a run as finished by updating run.json."""
        run_data = json.loads(self._run_path.read_text())
        run_data["status"] = status
        run_data["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._run_path.write_text(_dump_json(run_data))
        logger.info("Finished eval run %s with status=%s", run_id, status)

    async def record_iteration(
        self,
        run_id: uuid.UUID,
        iteration_number: int,
        phase: str,
        num_episodes: int,
        num_succeeded: int,
        num_failed: int,
        error_counts: Optional[dict[str, int]] = None,
        metrics: Optional[dict[str, Any]] = None,
        active_variants: Optional[dict[str, Any]] = None,
        autopilot_session_id: Optional[str] = None,
        autopilot_turns: Optional[int] = None,
        autopilot_config_writes: Optional[int] = None,
        autopilot_final_status: Optional[str] = None,
    ) -> uuid.UUID:
        """Append an iteration result to iterations.json. Returns the iteration ID."""
        iteration_id_str = _uuid7_hex()
        iteration_id = uuid.UUID(iteration_id_str)

        record = {
            "id": iteration_id_str,
            "run_id": str(run_id),
            "iteration_number": iteration_number,
            "phase": phase,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "num_episodes": num_episodes,
            "num_succeeded": num_succeeded,
            "num_failed": num_failed,
            "error_counts": error_counts,
            "metrics": metrics,
            "active_variants": active_variants,
            "autopilot_session_id": autopilot_session_id,
            "autopilot_turns": autopilot_turns,
            "autopilot_config_writes": autopilot_config_writes,
            "autopilot_final_status": autopilot_final_status,
        }

        # Read existing iterations, append, write back
        iterations = json.loads(self._iterations_path.read_text())
        iterations.append(record)
        self._iterations_path.write_text(_dump_json(iterations))

        logger.info(
            "Recorded iteration %d (%s) for run %s",
            iteration_number,
            phase,
            run_id,
        )
        return iteration_id
