#!/usr/bin/env python3
"""
Local development helper for TensorZero.

Usage:
    ./scripts/dev.py up [--proxy] [--seed NAME]  # Start docker + gateway
    ./scripts/dev.py down                        # Stop docker containers

Options:
    --proxy       Also start the provider-proxy in read-write mode, and set
                  TENSORZERO_E2E_PROXY so the gateway routes through it.
                  Recorded responses land in ci/provider-proxy-cache/.
    --seed NAME   Run gateway with separate DB names, one per worktree/clone.
                  The port is derived from the name (3001 + hash % 999).
                  Auto-detected from directory name if not specified:
                    tensorzero-black/ -> seed "black"
                    tensorzero-red/   -> seed "red"
                  Allows multiple instances side by side:
                    cd tensorzero-black && ./scripts/dev.py up  # auto: seed black
                    cd tensorzero-red && ./scripts/dev.py up    # auto: seed red
                  Plain `tensorzero/` gets no seed (port 3000, default DBs).
                  Docker containers are shared; only the gateway and DB names differ.

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


def seed_to_port(name):
    """Derive a deterministic port from a seed name (3001-3999)."""
    # Use a stable hash (sum of char codes) since Python's hash() is randomized per process
    h = sum(ord(c) for c in name) % 999
    return 3001 + h


def auto_detect_seed():
    """Detect seed from repo directory name (e.g. tensorzero-black -> black)."""
    repo_name = ROOT.name  # e.g. "tensorzero-black"
    prefix = "tensorzero-"
    if repo_name.startswith(prefix) and len(repo_name) > len(prefix):
        return repo_name[len(prefix) :]
    return None


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
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Health.Status}}",
                "e2e-provider-proxy-1",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and "healthy" in result.stdout
    except Exception:
        return False


def cmd_up(use_proxy=False, seed=None):
    services = list(DOCKER_SERVICES)

    # Compute seed-dependent settings
    if seed:
        gateway_port = seed_to_port(seed)
        ch_db = f"tensorzero_e2e_tests_{seed}"
        pg_db = f"tensorzero-e2e-tests-{seed}"
    else:
        gateway_port = 3000
        ch_db = "tensorzero_e2e_tests"
        pg_db = "tensorzero-e2e-tests"

    ch_url = f"http://chuser:chpassword@localhost:8123/{ch_db}"
    pg_url = f"postgres://postgres:postgres@localhost:5432/{pg_db}"

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

    # 2. Create seed-specific databases if needed
    if seed:
        _ensure_clickhouse_db(ch_db)
        _ensure_postgres_db(pg_db)

    # 3. Set database URLs for the gateway
    os.environ["TENSORZERO_CLICKHOUSE_URL"] = ch_url
    os.environ["TENSORZERO_POSTGRES_URL"] = pg_url

    # 4. Load env (API keys etc.)
    load_env()

    # 5. Run Postgres migrations first (separate step — this flag makes the gateway exit after migrating)
    if seed:
        print("Running Postgres migrations...")
        subprocess.run(
            [
                "cargo",
                "run",
                "--bin",
                "gateway",
                "--features",
                "e2e_tests",
                "--",
                "--run-postgres-migrations",
                "--config-file",
                "tensorzero-core/tests/e2e/config/tensorzero.*.toml",
            ],
            cwd=CRATES,
            check=True,
        )

    # 6. Run gateway in foreground
    bind_addr = f"0.0.0.0:{gateway_port}"
    os.environ["TENSORZERO_GATEWAY_BIND_ADDRESS"] = bind_addr

    label = f" [{seed}]" if seed else ""
    print(f"\nStarting gateway{label} on port {gateway_port} (Ctrl+C to stop)...")
    print(f"  TENSORZERO_GATEWAY_BIND_ADDRESS={bind_addr}")
    print(f"  TENSORZERO_CLICKHOUSE_URL={ch_url}")
    print(f"  TENSORZERO_POSTGRES_URL={pg_url}")
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


def _ensure_clickhouse_db(db_name):
    """Create a ClickHouse database if it doesn't exist."""
    try:
        subprocess.run(
            [
                "curl",
                "-s",
                "http://chuser:chpassword@localhost:8123/",
                "--data-binary",
                f"CREATE DATABASE IF NOT EXISTS {db_name}",
            ],
            capture_output=True,
            timeout=5,
        )
        print(f"ClickHouse database `{db_name}` ready.")
    except Exception as e:
        print(f"Warning: could not create ClickHouse DB `{db_name}`: {e}")


def _ensure_postgres_db(db_name):
    """Create a Postgres database if it doesn't exist (via docker exec)."""
    try:
        result = subprocess.run(
            [
                "docker",
                "exec",
                "e2e-postgres-1",
                "psql",
                "-U",
                "postgres",
                "-tc",
                f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if "1" not in (result.stdout or ""):
            subprocess.run(
                [
                    "docker",
                    "exec",
                    "e2e-postgres-1",
                    "psql",
                    "-U",
                    "postgres",
                    "-c",
                    f'CREATE DATABASE "{db_name}"',
                ],
                capture_output=True,
                timeout=10,
            )
            print(f"Postgres database `{db_name}` created.")
        else:
            print(f"Postgres database `{db_name}` ready.")
    except Exception as e:
        print(f"Warning: could not create Postgres DB `{db_name}`: {e}")


def cmd_down():
    print("Stopping docker...")
    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(DOCKER_COMPOSE),
            "--profile",
            "provider-proxy",
            "down",
        ],
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
        use_proxy = "--proxy" in extra
        if "--seed" in extra:
            idx = extra.index("--seed")
            if idx + 1 < len(extra):
                seed = extra[idx + 1]
            else:
                print("Error: --seed requires a name (e.g. black, red)")
                sys.exit(1)
        else:
            seed = auto_detect_seed()
            if seed:
                print(f"Auto-detected seed from directory: {seed}")
        cmd_up(use_proxy=use_proxy, seed=seed)
    elif cmd == "down":
        cmd_down()
    else:
        print(f"Unknown command: {cmd}\n")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
