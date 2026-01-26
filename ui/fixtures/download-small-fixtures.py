# /// script
# dependencies = [
#   "requests",
# ]
# ///
# For local development without R2 credentials, use download-small-fixtures-http.py instead.

import hashlib
import os
import subprocess
import time
from pathlib import Path

import requests

# cd to directory of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# Constants
# =============================================================================

PART_SIZE = 8388608

# Map of remote filename (in R2) -> local filename
# When a file needs updating, add version suffix to remote name (e.g., "file_v2.jsonl")
# and keep the local name unchanged (e.g., "file.jsonl")
FIXTURES = {
    "model_inference_examples.jsonl": "model_inference_examples.jsonl",
    "chat_inference_examples_20260123.jsonl": "chat_inference_examples.jsonl",
    "json_inference_examples.jsonl": "json_inference_examples.jsonl",
    "boolean_metric_feedback_examples.jsonl": "boolean_metric_feedback_examples.jsonl",
    "float_metric_feedback_examples.jsonl": "float_metric_feedback_examples.jsonl",
    "demonstration_feedback_examples.jsonl": "demonstration_feedback_examples.jsonl",
    "model_inference_cache_e2e_20260122_183412.jsonl": "model_inference_cache_e2e.jsonl",
    "json_inference_datapoint_examples.jsonl": "json_inference_datapoint_examples.jsonl",
    "chat_inference_datapoint_examples.jsonl": "chat_inference_datapoint_examples.jsonl",
    "dynamic_evaluation_run_episode_examples.jsonl": "dynamic_evaluation_run_episode_examples.jsonl",
    "jaro_winkler_similarity_feedback.jsonl": "jaro_winkler_similarity_feedback.jsonl",
    "comment_feedback_examples.jsonl": "comment_feedback_examples.jsonl",
    "dynamic_evaluation_run_examples.jsonl": "dynamic_evaluation_run_examples.jsonl",
}

R2_PUBLIC_BUCKET_URL = "https://pub-147e9850a60643208c411e70b636e956.r2.dev"
R2_S3_ENDPOINT_URL = "https://19918a216783f0ac9e052233569aef60.r2.cloudflarestorage.com/"
FIXTURES_DIR = Path("./small-fixtures")


# =============================================================================
# Shared utilities
# =============================================================================


def calculate_etag(file_path):
    """Calculate S3/R2 style ETag for a file."""
    file_size = os.path.getsize(file_path)
    num_parts = (file_size + PART_SIZE - 1) // PART_SIZE

    if num_parts == 1:
        # Single part upload - just MD5 of the file
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    else:
        # Multipart upload - concatenate MD5s of each part
        md5s = []
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(PART_SIZE)
                if not chunk:
                    break
                md5s.append(hashlib.md5(chunk).digest())

        # Calculate MD5 of concatenated part MD5s
        combined_md5 = hashlib.md5(b"".join(md5s)).hexdigest()
        return f"{combined_md5}-{num_parts}"


def get_remote_etag(remote_filename, retries=3):
    """Get ETag from R2 bucket via public URL."""
    for i in range(retries):
        try:
            response = requests.head(f"{R2_PUBLIC_BUCKET_URL}/{remote_filename}", timeout=30)
            response.raise_for_status()
            return response.headers.get("ETag", "").strip('"')
        except Exception as e:
            if i < retries - 1:
                print(
                    f"Error getting ETag for `{remote_filename}` (attempt {i + 1} of {retries}): {e}",
                    flush=True,
                )
                time.sleep(1)
            else:
                raise


def verify_etags():
    """Verify ETags of all downloaded fixtures match remote."""
    mismatches = []
    for remote_filename in FIXTURES.keys():
        local_file = FIXTURES_DIR / remote_filename
        if not local_file.exists():
            raise Exception(f"Fixture {remote_filename} not found after sync")

        local_etag = calculate_etag(local_file)
        remote_etag = get_remote_etag(remote_filename)

        if local_etag != remote_etag:
            mismatches.append(f"{remote_filename}: local={local_etag}, remote={remote_etag}")
        else:
            print(f"ETag OK: {remote_filename}", flush=True)

    if mismatches:
        raise Exception("ETag mismatches after sync:\n" + "\n".join(mismatches))


def rename_fixtures():
    """Rename downloaded fixtures from remote names to local names."""
    for remote_filename, local_filename in FIXTURES.items():
        if remote_filename != local_filename:
            src = FIXTURES_DIR / remote_filename
            dst = FIXTURES_DIR / local_filename
            if src.exists():
                src.rename(dst)
                print(f"Renamed {remote_filename} -> {local_filename}", flush=True)


# =============================================================================
# Authenticated path: S3 sync (used in CI with R2 credentials)
# =============================================================================


def sync_fixtures_from_r2(retries: int = 3) -> None:
    """Sync fixtures from R2 using aws s3 sync with retry logic."""
    cmd = [
        "aws",
        "s3",
        "--region",
        "auto",
        "--endpoint-url",
        R2_S3_ENDPOINT_URL,
        "--no-progress",
        "--cli-connect-timeout",
        "30",
        "--cli-read-timeout",
        "180",
        "sync",
        "s3://tensorzero-fixtures/",
        str(FIXTURES_DIR),
        # Only download the files in `FIXTURES`
        "--exclude",
        "*",
        *[arg for f in FIXTURES.keys() for arg in ("--include", f)],
    ]

    env = {
        **os.environ,  # Preserve PATH and other environment variables
        "AWS_ACCESS_KEY_ID": os.environ["R2_ACCESS_KEY_ID"],
        "AWS_SECRET_ACCESS_KEY": os.environ["R2_SECRET_ACCESS_KEY"],
        "AWS_MAX_ATTEMPTS": "15",
        "AWS_RETRY_MODE": "adaptive",
    }

    for attempt in range(retries):
        print(f"Running aws s3 sync (attempt {attempt + 1} of {retries})...", flush=True)
        result = subprocess.run(cmd, env=env)

        if result.returncode == 0:
            print("Sync completed successfully. Verifying ETags...", flush=True)
            try:
                verify_etags()
                rename_fixtures()
                return
            except Exception as e:
                print(
                    f"Verification failed (attempt {attempt + 1} of {retries}): {e}",
                    flush=True,
                )
                if attempt >= retries - 1:
                    raise
        else:
            print(
                f"aws s3 sync failed with exit code {result.returncode} (attempt {attempt + 1} of {retries})",
                flush=True,
            )

        if attempt < retries - 1:
            sleep_time = 3**attempt  # Exponential backoff: 1, 3, 9 seconds
            print(f"Retrying in {sleep_time} seconds...", flush=True)
            time.sleep(sleep_time)

    raise Exception(f"aws s3 sync failed after {retries} attempts")


# =============================================================================
# Main
# =============================================================================


def main():
    FIXTURES_DIR.mkdir(exist_ok=True)

    if not os.environ.get("R2_ACCESS_KEY_ID") or not os.environ.get("R2_SECRET_ACCESS_KEY"):
        raise Exception(
            "R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY must be set. "
            "For local development without R2 credentials, use download-small-fixtures-http.py instead."
        )

    print("R2 credentials found, downloading fixtures using `aws s3 sync`", flush=True)
    sync_fixtures_from_r2()


if __name__ == "__main__":
    main()
