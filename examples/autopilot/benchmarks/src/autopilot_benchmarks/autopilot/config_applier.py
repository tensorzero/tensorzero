"""Python wrapper around the Rust config-applier-cli binary.

Sends EditPayload JSON to the CLI via stdin and reads written file paths from stdout.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_BINARY = "/usr/local/bin/config-applier-cli"


async def apply_config_writes(
    config_writes: list[dict[str, Any]],
    config_glob: str,
    binary_path: str = DEFAULT_BINARY,
) -> list[str]:
    """Apply config edits via the Rust CLI.

    Args:
        config_writes: List of EditPayload dicts from autopilot session.
        config_glob: Glob pattern for T0 config files (e.g., "config/**/*.toml").
        binary_path: Path to the config-applier-cli binary.

    Returns:
        List of file paths that were written.

    Raises:
        RuntimeError: If the CLI exits with a non-zero code.
        FileNotFoundError: If the binary is not found.
    """
    if not config_writes:
        logger.info("No config writes to apply")
        return []

    if not Path(binary_path).exists():
        raise FileNotFoundError(f"config-applier-cli binary not found at {binary_path}")

    input_json = json.dumps(config_writes)
    logger.info("Applying %d config writes via %s", len(config_writes), binary_path)
    logger.info("Config write payloads:\n%s", json.dumps(config_writes, indent=2))

    proc = await asyncio.create_subprocess_exec(
        binary_path,
        "--config-glob",
        config_glob,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await proc.communicate(input=input_json.encode())

    if proc.returncode != 0:
        error_msg = stderr.decode(errors="replace").strip()
        raise RuntimeError(f"config-applier-cli exited with code {proc.returncode}: {error_msg}")

    written_paths: list[str] = json.loads(stdout.decode())
    logger.info("Config applier wrote %d files: %s", len(written_paths), written_paths)
    return written_paths
