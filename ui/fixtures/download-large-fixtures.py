# /// script
# dependencies = [
#   "requests",
#   "parquet-tools",
# ]
# ///

import concurrent.futures
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
FIXTURES = [
    "large_chat_inference_v2.parquet",
    "large_chat_model_inference_v2.parquet",
    "large_json_inference_v2.parquet",
    "large_json_model_inference_v2.parquet",
    "large_chat_boolean_feedback.parquet",
    "large_chat_float_feedback.parquet",
    "large_chat_comment_feedback.parquet",
    "large_chat_demonstration_feedback.parquet",
    "large_json_boolean_feedback.parquet",
    "large_json_float_feedback.parquet",
    "large_json_comment_feedback.parquet",
    "large_json_demonstration_feedback.parquet",
]
R2_PUBLIC_BUCKET_URL = "https://pub-147e9850a60643208c411e70b636e956.r2.dev"
R2_S3_ENDPOINT_URL = "https://19918a216783f0ac9e052233569aef60.r2.cloudflarestorage.com/"
S3_FIXTURES_DIR = Path("./s3-fixtures")


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


def get_remote_etag(filename):
    """Get ETag from R2 bucket via public URL."""
    response = requests.head(f"{R2_PUBLIC_BUCKET_URL}/{filename}", timeout=30)
    return response.headers.get("ETag", "").strip('"')


def verify_etags():
    """Verify ETags of all downloaded fixtures match remote."""
    mismatches = []
    for fixture in FIXTURES:
        local_file = S3_FIXTURES_DIR / fixture
        if not local_file.exists():
            raise Exception(f"Fixture {fixture} not found after sync")

        local_etag = calculate_etag(local_file)
        remote_etag = get_remote_etag(fixture)

        if local_etag != remote_etag:
            mismatches.append(f"{fixture}: local={local_etag}, remote={remote_etag}")
        else:
            print(f"ETag OK: {fixture}", flush=True)

    if mismatches:
        raise Exception("ETag mismatches after sync:\n" + "\n".join(mismatches))


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
        str(S3_FIXTURES_DIR),
        # Only download the files in `FIXTURES`
        "--exclude",
        "*",
        *[arg for f in FIXTURES for arg in ("--include", f)],
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
            verify_etags()
            return

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
# Public HTTP fallback (used locally without R2 credentials)
# =============================================================================


def download_file_http(filename, remote_etag):
    """Download a single file from R2 via public HTTP URL."""
    RETRIES = 3
    for i in range(RETRIES):
        try:
            url = f"{R2_PUBLIC_BUCKET_URL}/{filename}"
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            local_file = S3_FIXTURES_DIR / filename

            with open(local_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            local_etag = calculate_etag(local_file)
            if local_etag != remote_etag:
                raise Exception(f"ETag mismatch after downloading: {local_etag} != {remote_etag}")
            return
        except Exception as e:
            print(
                f"Error downloading `{filename}` (attempt {i + 1} of {RETRIES}): {e}",
                flush=True,
            )
            time.sleep(1)
    raise Exception(f"Failed to download `{filename}` after {RETRIES} attempts")


def download_fixtures_http():
    """Download all fixtures via public HTTP (fallback when no R2 credentials)."""

    def process_fixture(fixture):
        local_file = S3_FIXTURES_DIR / fixture
        remote_etag = get_remote_etag(fixture)

        if not local_file.exists():
            print(f"Downloading {fixture} (file doesn't exist locally)", flush=True)
            download_file_http(fixture, remote_etag)
            return

        local_etag = calculate_etag(local_file)

        if local_etag != remote_etag:
            print(f"Downloading {fixture} (ETag mismatch)", flush=True)
            print(f"Local ETag: {local_etag}", flush=True)
            print(f"Remote ETag: {remote_etag}", flush=True)
            download_file_http(fixture, remote_etag)
        else:
            print(f"Skipping {fixture} (up to date)", flush=True)

    # Use ThreadPoolExecutor to download files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Loop over the results to propagate exceptions
        for result in executor.map(process_fixture, FIXTURES):
            assert result is None


# =============================================================================
# Main
# =============================================================================


def main():
    S3_FIXTURES_DIR.mkdir(exist_ok=True)

    if os.environ.get("R2_ACCESS_KEY_ID") and os.environ.get("R2_SECRET_ACCESS_KEY"):
        print("R2 credentials found, downloading fixtures using `aws s3 sync`", flush=True)
        sync_fixtures_from_r2()
    else:
        print(
            "WARNING: `R2_ACCESS_KEY_ID` or `R2_SECRET_ACCESS_KEY` not set. Falling back to slow public HTTP downloads.",
            flush=True,
        )
        download_fixtures_http()

    for fixture in FIXTURES:
        print(f"Fixture {fixture}:", flush=True)
        subprocess.run(
            ["parquet-tools", "inspect", S3_FIXTURES_DIR / fixture],
            check=True,
            stderr=subprocess.STDOUT,
        )


if __name__ == "__main__":
    main()
