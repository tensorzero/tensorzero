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

# Constants
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
R2_BUCKET = "https://pub-147e9850a60643208c411e70b636e956.r2.dev"
S3_FIXTURES_DIR = Path("./s3-fixtures")


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
    """Get ETag from R2 bucket."""
    response = requests.head(f"{R2_BUCKET}/{filename}")
    return response.headers.get("ETag", "").strip('"')


def download_file(filename, remote_etag):
    """Download file from R2 bucket."""
    RETRIES = 3
    for i in range(RETRIES):
        try:
            url = f"{R2_BUCKET}/{filename}"
            response = requests.get(url, stream=True)
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


def main():
    # Create s3-fixtures directory if it doesn't exist
    S3_FIXTURES_DIR.mkdir(exist_ok=True)

    if os.environ.get("R2_ACCESS_KEY_ID") is not None and os.environ.get("R2_SECRET_ACCESS_KEY") != "":
        print("R2_ACCESS_KEY_ID set, downloading fixtures using 'aws s3 sync'")
        subprocess.check_call(
            f"aws s3 --region auto --endpoint-url https://19918a216783f0ac9e052233569aef60.r2.cloudflarestorage.com/ sync s3://tensorzero-fixtures/ {S3_FIXTURES_DIR}",
            env={
                "AWS_ACCESS_KEY_ID": os.environ["R2_ACCESS_KEY_ID"],
                "AWS_SECRET_ACCESS_KEY": os.environ["R2_SECRET_ACCESS_KEY"],
            },
            shell=True,
        )
        return

    def process_fixture(fixture):
        local_file = S3_FIXTURES_DIR / fixture
        remote_etag = get_remote_etag(fixture)

        if not local_file.exists():
            print(f"Downloading {fixture} (file doesn't exist locally)", flush=True)
            download_file(fixture, remote_etag)
            return

        local_etag = calculate_etag(local_file)

        if local_etag != remote_etag:
            print(f"Downloading {fixture} (ETag mismatch)", flush=True)
            print(f"Local ETag: {local_etag}", flush=True)
            print(f"Remote ETag: {remote_etag}", flush=True)
            download_file(fixture, remote_etag)
        else:
            print(f"Skipping {fixture} (up to date)", flush=True)

    # Use ThreadPoolExecutor to download files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Loop over the results to propagate exceptions
        for result in executor.map(process_fixture, FIXTURES):
            assert result is None

    for fixture in FIXTURES:
        print(f"Fixture {fixture}:", flush=True)
        subprocess.run(
            ["parquet-tools", "inspect", S3_FIXTURES_DIR / fixture],
            check=True,
            stderr=subprocess.STDOUT,
        )


if __name__ == "__main__":
    main()
