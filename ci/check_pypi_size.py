#!/usr/bin/env python3
"""
Check that the total size of local artifacts plus all existing PyPI releases
of `tensorzero` does not exceed the PyPI project size quota (currently 10 GB).

Usage:
    python3 ci/check_pypi_size.py './wheels-*/*'
"""

import argparse
import glob
import json
import os
import sys
import urllib.request

PYPI_PROJECT_URL = "https://pypi.org/pypi/tensorzero/json"


def format_size(size_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def fetch_pypi_release_sizes() -> tuple[int, int]:
    """Fetch the total size of all existing releases from PyPI.

    Returns:
        A tuple of (total_size_bytes, file_count).
    """
    req = urllib.request.Request(
        PYPI_PROJECT_URL,
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    total_size = 0
    file_count = 0
    for _version, files in data.get("releases", {}).items():
        for file_info in files:
            total_size += file_info.get("size", 0)
            file_count += 1

    return total_size, file_count


def get_local_artifact_sizes(pattern: str) -> tuple[int, int, list[tuple[str, int]]]:
    """Sum the sizes of local files matching the given glob pattern.

    Returns:
        A tuple of (total_size_bytes, file_count, list of (filename, size) pairs).
    """
    paths = glob.glob(pattern)
    if not paths:
        print(f"ERROR: No files matched pattern `{pattern}`", file=sys.stderr)
        sys.exit(1)

    total_size = 0
    file_count = 0
    file_details: list[tuple[str, int]] = []
    for path in sorted(paths):
        size = os.path.getsize(path)
        total_size += size
        file_count += 1
        file_details.append((path, size))

    return total_size, file_count, file_details


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check that local artifacts + existing PyPI releases don't exceed the size quota"
    )
    parser.add_argument(
        "pattern",
        help="Glob pattern for local wheel/sdist files (e.g. './wheels-*/*')",
    )
    parser.add_argument(
        "--max-size-gb",
        type=float,
        default=10.0,
        help="Maximum total size in GB (default: 10.0)",
    )
    args = parser.parse_args()

    max_size_bytes = int(args.max_size_gb * 1024 * 1024 * 1024)

    print("Fetching existing release sizes from PyPI...")
    pypi_size, pypi_file_count = fetch_pypi_release_sizes()
    print(f"  Existing PyPI releases: {pypi_file_count} files, {format_size(pypi_size)}")

    print(f"\nScanning local artifacts matching `{args.pattern}`...")
    local_size, local_file_count, local_files = get_local_artifact_sizes(args.pattern)
    for path, size in local_files:
        print(f"  {path}: {format_size(size)}")
    print(f"  Local artifacts total: {local_file_count} files, {format_size(local_size)}")

    combined_size = pypi_size + local_size
    print(f"\nCombined total: {format_size(combined_size)} / {format_size(max_size_bytes)}")

    if combined_size > max_size_bytes:
        print(
            f"\nERROR: Combined size ({format_size(combined_size)}) exceeds "
            f"limit ({format_size(max_size_bytes)})",
            file=sys.stderr,
        )
        sys.exit(1)

    remaining = max_size_bytes - combined_size
    print(f"Remaining capacity: {format_size(remaining)}")
    print("\nSize check passed.")


if __name__ == "__main__":
    main()
