"""Manage the TensorZero gateway as a subprocess."""

import asyncio
import logging
import os
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import httpx

logger = logging.getLogger(__name__)


class GatewayProcess:
    """Manages the TensorZero gateway binary as a subprocess.

    The gateway binary is expected to be present in the container image
    (e.g., /usr/local/bin/gateway from tensorzero/gateway:latest).
    """

    def __init__(
        self,
        binary_path: str = "/usr/local/bin/gateway",
        config_path: str = "config/tensorzero.toml",
        port: int = 3000,
        startup_timeout: float = 30.0,
        shutdown_timeout: float = 10.0,
        log_dir: str | Path | None = None,
    ):
        self.binary_path = binary_path
        self.config_path = config_path
        self.port = port
        self.startup_timeout = startup_timeout
        self.shutdown_timeout = shutdown_timeout
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self._process: asyncio.subprocess.Process | None = None
        self._stdout_handle: BinaryIO | None = None
        self._stderr_handle: BinaryIO | None = None

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    @property
    def stdout_log_path(self) -> Path | None:
        if self.log_dir is None:
            return None
        return self.log_dir / "gateway.stdout.log"

    @property
    def stderr_log_path(self) -> Path | None:
        if self.log_dir is None:
            return None
        return self.log_dir / "gateway.stderr.log"

    async def start(self, env_overrides: dict[str, str] | None = None) -> None:
        """Start the gateway binary and wait for it to become healthy.

        Args:
            env_overrides: Additional environment variables to set for the gateway process.
        """
        if self.is_running:
            raise RuntimeError("Gateway is already running")

        env = {**os.environ}
        if env_overrides:
            env.update(env_overrides)

        # Ensure the config file exists
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Gateway config not found: {self.config_path}")

        logger.info(
            "Starting gateway: %s (config=%s, port=%d)",
            self.binary_path,
            self.config_path,
            self.port,
        )

        env["TENSORZERO_GATEWAY_BIND_ADDRESS"] = f"0.0.0.0:{self.port}"
        self._open_log_handles()

        try:
            # Run migrations first (one-shot processes, exit when done)
            for migration_flag in ("--run-postgres-migrations",):
                logger.info("Running gateway %s...", migration_flag)
                self._write_log_banner(migration_flag)
                migrate_proc = await asyncio.create_subprocess_exec(
                    self.binary_path,
                    "--config-file",
                    self.config_path,
                    migration_flag,
                    env=env,
                    stdout=self._stdout_handle,
                    stderr=self._stderr_handle,
                )
                await migrate_proc.wait()
                if migrate_proc.returncode != 0:
                    raise RuntimeError(f"Gateway {migration_flag} failed (code {migrate_proc.returncode})")
                logger.info("Gateway %s complete", migration_flag)

            # Now start the gateway for real
            self._write_log_banner("gateway")
            self._process = await asyncio.create_subprocess_exec(
                self.binary_path,
                "--config-file",
                self.config_path,
                env=env,
                stdout=self._stdout_handle,
                stderr=self._stderr_handle,
            )

            await self._wait_for_health()
        except Exception:
            await self._cleanup_start_failure()
            raise

        if self.stdout_log_path is not None and self.stderr_log_path is not None:
            logger.info(
                "Gateway is healthy at %s (logs: %s, %s)",
                self.url,
                self.stdout_log_path,
                self.stderr_log_path,
            )
        else:
            logger.info("Gateway is healthy at %s", self.url)

    async def stop(self) -> None:
        """Stop the gateway process gracefully."""
        if self._process is None:
            self._close_log_handles()
            return

        if self._process.returncode is not None:
            logger.info(
                "Gateway process already exited with code %d",
                self._process.returncode,
            )
            self._process = None
            self._close_log_handles()
            return

        logger.info("Stopping gateway (PID %d)", self._process.pid)

        # Send SIGTERM for graceful shutdown
        self._process.send_signal(signal.SIGTERM)
        try:
            await asyncio.wait_for(self._process.wait(), timeout=self.shutdown_timeout)
            logger.info("Gateway stopped gracefully")
        except asyncio.TimeoutError:
            logger.warning("Gateway did not stop gracefully, sending SIGKILL")
            self._process.kill()
            await self._process.wait()

        self._process = None
        self._close_log_handles()

    async def restart(self, env_overrides: dict[str, str] | None = None) -> None:
        """Restart the gateway process."""
        await self.stop()
        await self.start(env_overrides)

    async def _wait_for_health(self) -> None:
        """Poll the /health endpoint until the gateway is ready."""
        health_url = f"{self.url}/health"
        deadline = asyncio.get_event_loop().time() + self.startup_timeout

        async with httpx.AsyncClient() as client:
            while asyncio.get_event_loop().time() < deadline:
                # Check if process died
                if self._process is not None and self._process.returncode is not None:
                    raise RuntimeError(f"Gateway process exited with code {self._process.returncode}")

                try:
                    resp = await client.get(health_url, timeout=2.0)
                    if resp.status_code == 200:
                        return
                except httpx.ConnectError:
                    pass
                except httpx.TimeoutException:
                    pass

                await asyncio.sleep(0.5)

        raise TimeoutError(f"Gateway did not become healthy within {self.startup_timeout}s")

    def _open_log_handles(self) -> None:
        if self.log_dir is None or self._stdout_handle is not None:
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._stdout_handle = self.stdout_log_path.open("ab")
        self._stderr_handle = self.stderr_log_path.open("ab")

    def _close_log_handles(self) -> None:
        for handle in (self._stdout_handle, self._stderr_handle):
            if handle is not None:
                handle.close()
        self._stdout_handle = None
        self._stderr_handle = None

    async def _cleanup_start_failure(self) -> None:
        if self._process is not None and self._process.returncode is None:
            self._process.kill()
            await self._process.wait()
        self._process = None
        self._close_log_handles()

    def _write_log_banner(self, stage: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        banner = (f"\n=== [{timestamp}] {stage} (config={self.config_path}, port={self.port}) ===\n").encode()

        if self._stdout_handle is not None:
            self._stdout_handle.write(banner)
            self._stdout_handle.flush()
        if self._stderr_handle is not None:
            self._stderr_handle.write(banner)
            self._stderr_handle.flush()
