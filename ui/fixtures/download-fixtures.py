# /// script
# dependencies = [
#   "requests",
# ]
# ///


import os
import hashlib
import requests
import sys
from pathlib import Path

# Constants
PART_SIZE = 8388608
FIXTURES = ["large_chat_inference.parquet", "large_model_inference.parquet"]
R2_BUCKET = "https://pub-147e9850a60643208c411e70b636e956.r2.dev"
S3_FIXTURES_DIR = Path("./s3-fixtures")

def calculate_etag(file_path):
    """Calculate S3/R2 style ETag for a file."""
    file_size = os.path.getsize(file_path)
    num_parts = (file_size + PART_SIZE - 1) // PART_SIZE

    if num_parts == 1:
        # Single part upload - just MD5 of the file
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    else:
        # Multipart upload - concatenate MD5s of each part
        md5s = []
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(PART_SIZE)
                if not chunk:
                    break
                md5s.append(hashlib.md5(chunk).digest())
        
        # Calculate MD5 of concatenated part MD5s
        combined_md5 = hashlib.md5(b''.join(md5s)).hexdigest()
        return f"{combined_md5}-{num_parts}"

def get_remote_etag(filename):
    """Get ETag from R2 bucket."""
    response = requests.head(f"{R2_BUCKET}/{filename}")
    return response.headers.get('ETag', '').strip('"')

def download_file(filename):
    """Download file from R2 bucket."""
    url = f"{R2_BUCKET}/{filename}"
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(S3_FIXTURES_DIR / filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def main():
    # Create s3-fixtures directory if it doesn't exist
    S3_FIXTURES_DIR.mkdir(exist_ok=True)

    for fixture in FIXTURES:
        local_file = S3_FIXTURES_DIR / fixture
        
        if not local_file.exists():
            print(f"Downloading {fixture} (file doesn't exist locally)")
            download_file(fixture)
            continue

        remote_etag = get_remote_etag(fixture)
        local_etag = calculate_etag(local_file)

        if local_etag != remote_etag:
            print(f"Downloading {fixture} (ETag mismatch)")
            print(f"Local ETag: {local_etag}")
            print(f"Remote ETag: {remote_etag}")
            download_file(fixture)
        else:
            print(f"Skipping {fixture} (up to date)")

if __name__ == "__main__":
    main()