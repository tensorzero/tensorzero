# /// script
# dependencies = [
#   "requests",
# ]
# ///
# HTTP-only version for local development without R2 credentials.
# For CI with R2 credentials, use download-small-fixtures.py instead.

import concurrent.futures
import hashlib
import os
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
    "chat_inference_datapoint_examples_20260129.jsonl": "chat_inference_datapoint_examples.jsonl",
    "dynamic_evaluation_run_episode_examples.jsonl": "dynamic_evaluation_run_episode_examples.jsonl",
    "jaro_winkler_similarity_feedback.jsonl": "jaro_winkler_similarity_feedback.jsonl",
    "comment_feedback_examples.jsonl": "comment_feedback_examples.jsonl",
    "dynamic_evaluation_run_examples.jsonl": "dynamic_evaluation_run_examples.jsonl",
}

R2_PUBLIC_BUCKET_URL = "https://pub-147e9850a60643208c411e70b636e956.r2.dev"
FIXTURES_DIR = Path("./small-fixtures")


# =============================================================================
# Utilities
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


# =============================================================================
# HTTP download
# =============================================================================


def download_file_http(remote_filename, local_filename, remote_etag):
    """Download a single file from R2 via public HTTP URL."""
    RETRIES = 3
    for i in range(RETRIES):
        try:
            url = f"{R2_PUBLIC_BUCKET_URL}/{remote_filename}"
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            local_file = FIXTURES_DIR / local_filename

            with open(local_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            local_etag = calculate_etag(local_file)
            if local_etag != remote_etag:
                raise Exception(f"ETag mismatch after downloading: {local_etag} != {remote_etag}")
            return
        except Exception as e:
            print(
                f"Error downloading `{remote_filename}` (attempt {i + 1} of {RETRIES}): {e}",
                flush=True,
            )
            time.sleep(1)
    raise Exception(f"Failed to download `{remote_filename}` after {RETRIES} attempts")


def download_fixtures_http():
    """Download all fixtures via public HTTP."""

    def process_fixture(item):
        remote_filename, local_filename = item
        local_file = FIXTURES_DIR / local_filename
        remote_etag = get_remote_etag(remote_filename)

        if not local_file.exists():
            print(f"Downloading {remote_filename} (file doesn't exist locally)", flush=True)
            download_file_http(remote_filename, local_filename, remote_etag)
            return

        local_etag = calculate_etag(local_file)

        if local_etag != remote_etag:
            print(f"Downloading {remote_filename} (ETag mismatch)", flush=True)
            print(f"Local ETag: {local_etag}", flush=True)
            print(f"Remote ETag: {remote_etag}", flush=True)
            download_file_http(remote_filename, local_filename, remote_etag)
        else:
            print(f"Skipping {remote_filename} (up to date)", flush=True)

    # Use ThreadPoolExecutor to download files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Loop over the results to propagate exceptions
        for result in executor.map(process_fixture, FIXTURES.items()):
            assert result is None


# =============================================================================
# Main
# =============================================================================


def main():
    FIXTURES_DIR.mkdir(exist_ok=True)
    print("Downloading fixtures via public HTTP...", flush=True)
    download_fixtures_http()


if __name__ == "__main__":
    main()
