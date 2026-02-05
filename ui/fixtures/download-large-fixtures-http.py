# /// script
# dependencies = [
#   "requests",
#   "parquet-tools",
# ]
# ///
# HTTP-only version for local development without R2 credentials.
# For CI with R2 credentials, use download-large-fixtures.py instead.

import concurrent.futures
import hashlib
import os
import subprocess
import time

import requests
from download_fixtures_consts import LARGE_FIXTURES as FIXTURES
from download_fixtures_consts import LARGE_FIXTURES_DIR, PART_SIZE, R2_PUBLIC_BUCKET_URL

# cd to directory of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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


def get_remote_etag(filename, retries: int = 3):
    """Get ETag from R2 bucket via public URL with retry logic."""
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.head(f"{R2_PUBLIC_BUCKET_URL}/{filename}", timeout=30)
            response.raise_for_status()
            etag = response.headers.get("ETag")
            if not etag:
                raise Exception(f"Missing ETag header for {filename}")
            return etag.strip('"')
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                sleep_time = 3**attempt
                print(
                    f"Error fetching ETag for {filename} (attempt {attempt + 1} of {retries}): {exc}",
                    flush=True,
                )
                print(f"Retrying in {sleep_time} seconds...", flush=True)
                time.sleep(sleep_time)

    raise Exception(f"Failed to fetch ETag for {filename} after {retries} attempts") from last_error


# =============================================================================
# HTTP download
# =============================================================================


def download_file_http(filename, remote_etag):
    """Download a single file from R2 via public HTTP URL."""
    RETRIES = 3
    for i in range(RETRIES):
        try:
            url = f"{R2_PUBLIC_BUCKET_URL}/{filename}"
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            local_file = LARGE_FIXTURES_DIR / filename

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
    """Download all fixtures via public HTTP."""

    def process_fixture(fixture):
        local_file = LARGE_FIXTURES_DIR / fixture
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
    LARGE_FIXTURES_DIR.mkdir(exist_ok=True)
    print("Downloading fixtures via public HTTP...", flush=True)
    download_fixtures_http()

    for fixture in FIXTURES:
        print(f"Fixture {fixture}:", flush=True)
        subprocess.run(
            ["parquet-tools", "inspect", LARGE_FIXTURES_DIR / fixture],
            check=True,
            stderr=subprocess.STDOUT,
        )


if __name__ == "__main__":
    main()
