#!/usr/bin/env python3
"""
Local development helper for TensorZero.

Usage:
    ./scripts/dev.py up [--proxy]     # Start docker + gateway (logs stream here)
    ./scripts/dev.py down             # Stop docker containers

Options:
    --proxy   Also start the provider-proxy in read-write mode, and set
              TENSORZERO_E2E_PROXY so the gateway routes through it.
              Recorded responses land in ci/provider-proxy-cache/.

`up` starts ClickHouse + Postgres, loads .env, and runs the gateway in the
foreground. Ctrl+C stops the gateway. Docker keeps running (use `down`).

Then in another terminal:
    ./scripts/test-cache-tokens.sh --gemini --clickhouse
    cd crates && cargo test-e2e -E 'test(cache_input_tokens)'
"""

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CRATES = ROOT / "crates"
DOCKER_COMPOSE = CRATES / "tensorzero-core" / "tests" / "e2e" / "docker-compose.yml"
ENV_FILE = ROOT / ".env"

DOCKER_SERVICES = ["clickhouse", "postgres"]


def load_env():
    """Load .env file into os.environ."""
    if not ENV_FILE.exists():
        print(f"Warning: {ENV_FILE} not found — API keys won't be set.")
        return
    count = 0
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value:
                os.environ[key] = value
                count += 1
    print(f"Loaded {count} vars from {ENV_FILE}")


def url_ok(url):
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def wait_for(url, name, timeout=60):
    print(f"Waiting for {name}...", end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if url_ok(url):
            print(f" ready ({time.time() - t0:.0f}s)")
            return True
        print(".", end="", flush=True)
        time.sleep(2)
    print(f" TIMEOUT ({timeout}s)")
    return False


def proxy_container_healthy():
    """Check if the provider-proxy container is running and healthy via docker."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Health.Status}}", "e2e-provider-proxy-1"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and "healthy" in result.stdout
    except Exception:
        return False


def cmd_up(use_proxy=False):
    services = list(DOCKER_SERVICES)

    # 1. Start docker (--wait ensures containers are healthy before returning)
    if url_ok("http://localhost:8123/ping"):
        print("Docker already running.")
        if use_proxy and not proxy_container_healthy():
            print("Starting provider-proxy...")
            env = {**os.environ, "PROVIDER_PROXY_CACHE_MODE": "read-write"}
            subprocess.run(
                [
                    "docker",
                    "compose",
                    "-f",
                    str(DOCKER_COMPOSE),
                    "--profile",
                    "provider-proxy",
                    "up",
                    "-d",
                    "--wait",
                    "provider-proxy",
                ],
                cwd=ROOT,
                check=True,
                env=env,
            )
    else:
        print("Starting docker...")
        cmd = ["docker", "compose", "-f", str(DOCKER_COMPOSE)]
        env = dict(os.environ)
        if use_proxy:
            cmd += ["--profile", "provider-proxy"]
            env["PROVIDER_PROXY_CACHE_MODE"] = "read-write"
        cmd += ["up", "-d", "--wait"] + services
        if use_proxy:
            cmd.append("provider-proxy")
        subprocess.run(cmd, cwd=ROOT, check=True, env=env)

        if not wait_for("http://localhost:8123/ping", "ClickHouse"):
            sys.exit(1)

    if use_proxy:
        # Docker --wait already verified the container is healthy internally.
        # Port 3004 (health) isn't exposed to the host, only port 3003 (proxy).
        os.environ["TENSORZERO_E2E_PROXY"] = "http://localhost:3003"
        print("Provider-proxy enabled — responses will be recorded to ci/provider-proxy-cache/")

    # 2. Load env
    load_env()

    # 3. Run gateway in foreground
    print("\nStarting gateway (Ctrl+C to stop)...")
    if use_proxy:
        print(f"  TENSORZERO_E2E_PROXY={os.environ.get('TENSORZERO_E2E_PROXY', '')}")
    print("=" * 60)
    try:
        subprocess.run(
            [
                "cargo",
                "run",
                "--bin",
                "gateway",
                "--features",
                "e2e_tests",
                "--",
                "--config-file",
                "tensorzero-core/tests/e2e/config/tensorzero.*.toml",
            ],
            cwd=CRATES,
            check=True,
        )
    except KeyboardInterrupt:
        print("\nGateway stopped.")


def cmd_down():
    print("Stopping docker...")
    subprocess.run(
        ["docker", "compose", "-f", str(DOCKER_COMPOSE), "--profile", "provider-proxy", "down"],
        cwd=ROOT,
        check=True,
    )


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]
    extra = sys.argv[2:]

    if cmd == "up":
        cmd_up(use_proxy="--proxy" in extra)
    elif cmd == "down":
        cmd_down()
    else:
        print(f"Unknown command: {cmd}\n")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
