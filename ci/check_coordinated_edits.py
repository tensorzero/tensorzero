#!/usr/bin/env python3
"""
Check that files marked with Lint.IfEdited/ThenEdit markers have coordinated edits.

This script validates that when a line within a Lint.IfEdited() block is modified,
all files listed in the associated Lint.ThenEdit() marker are also modified in the same PR.

The markers can appear anywhere in a line (in any comment style or file format).

To skip this check, add the 'skip-if-edited-check' label to your PR.

Usage:
    python check_coordinated_edits.py [--base-ref BASE] [--head-ref HEAD]

Testing:
    python tests/test_check_coordinated_edits.py

Environment Variables:
    GITHUB_EVENT_NAME: The GitHub event type (pull_request, merge_group, etc.)
    GITHUB_BASE_REF: Base branch for PRs
    GITHUB_REF: Current ref for merge groups
    GITHUB_EVENT_PATH: Path to GitHub event payload (for label checking)
"""

import argparse
import os
import re
import subprocess
import sys
from typing import List, Set, Tuple


def run_command(cmd: List[str]) -> str:
    """Run a command and return its output."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def get_changed_files(base_ref: str, head_ref: str) -> Set[str]:
    """Get the list of files changed between base and head."""
    try:
        output = run_command(["git", "diff", "--name-only", f"{base_ref}...{head_ref}"])
        return set(line.strip() for line in output.split("\n") if line.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error getting changed files: {e}", file=sys.stderr)
        sys.exit(1)


def get_changed_line_ranges(file_path: str, base_ref: str, head_ref: str) -> List[Tuple[int, int]]:
    """Get the line ranges that were modified in a file."""
    try:
        # Use git diff with unified format to get changed line numbers
        output = run_command(["git", "diff", "-U0", f"{base_ref}...{head_ref}", "--", file_path])

        ranges = []
        # Parse diff output for line ranges: @@ -start,count +start,count @@
        for line in output.split("\n"):
            # Match both old and new sides: @@ -old_start,old_count +new_start,new_count @@
            match = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if match:
                _old_start = int(match.group(1))
                old_count = int(match.group(2)) if match.group(2) else 1
                new_start = int(match.group(3))
                new_count = int(match.group(4)) if match.group(4) else 1

                # For additions or modifications, use the new-side range
                if new_count > 0:
                    ranges.append((new_start, new_start + new_count - 1))
                # For pure deletions (new_count == 0), record the deletion point
                # The deletion happens "at" the line position where content was removed
                elif old_count > 0:
                    # For a deletion at position N, we record it as modifying line N
                    # This ensures Lint.IfEdited blocks that contain the deletion are detected
                    ranges.append((new_start, new_start))

        return ranges
    except subprocess.CalledProcessError:
        return []


def is_line_in_ranges(line_num: int, ranges: List[Tuple[int, int]]) -> bool:
    """Check if a line number falls within any of the given ranges."""
    return any(start <= line_num <= end for start, end in ranges)


def parse_lint_blocks(file_path: str, changed_ranges: List[Tuple[int, int]]) -> List[Set[str]]:
    """
    Parse Lint.IfEdited/ThenEdit blocks and return required files for modified blocks.

    Format:
        // Lint.IfEdited()
        [code that requires coordinated edits]
        // Lint.ThenEdit(file1, file2, ...)

    The markers can appear anywhere in a line (e.g., in comments for any language). These file paths are relative to
    the root of the repository.

    Returns a list of sets, where each set contains file paths that must be edited together.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except (IOError, UnicodeDecodeError):
        return []

    required_edits = []
    in_lint_block = False
    lint_block_start = 0

    for i, line in enumerate(lines, start=1):
        # Check for Lint.IfEdited() marker - starts a new block
        # Just look for the token anywhere in the line
        if "Lint.IfEdited()" in line:
            in_lint_block = True
            lint_block_start = i
            continue

        # Check for Lint.ThenEdit() marker - ends the block
        if in_lint_block:
            # Look for single-line format: Lint.ThenEdit(file1, file2)
            match = re.search(r"Lint\.ThenEdit\(([^)]*)\)", line)
            if match:
                lint_block_end = i

                # Extract file paths from the parentheses
                files_str = match.group(1)
                # Split by comma and clean up whitespace
                files = [f.strip() for f in files_str.split(",") if f.strip()]

                # Check if any line in this block (between IfEdited and ThenEdit) was modified
                # Exclude the marker lines themselves (lint_block_start and lint_block_end)
                block_modified = any(
                    is_line_in_ranges(line_num, changed_ranges)
                    for line_num in range(lint_block_start + 1, lint_block_end)
                )

                if block_modified and files:
                    required_edits.append(set(files))

                in_lint_block = False
                continue

            # Check if we're in a multiline ThenEdit continuation
            # Look for Lint.ThenEdit( without closing paren
            multiline_match = re.search(r"Lint\.ThenEdit\((.*)", line)
            if multiline_match:
                # Start of multiline ThenEdit
                files_str = multiline_match.group(1).rstrip(",").rstrip()
                current_files = [f.strip() for f in files_str.split(",") if f.strip() and f.strip() != ")"]

                # Look ahead for continuation lines
                j = i + 1
                while j <= len(lines):
                    next_line = lines[j - 1]
                    # Check if this line ends the ThenEdit
                    if ")" in next_line:
                        # Extract any remaining files before the closing paren
                        remaining = next_line.split(")")[0]
                        # Parse: strip leading spaces/tabs, then strip until alphanumeric
                        remaining = remaining.lstrip(" \t")
                        remaining = re.sub(r"^[^a-zA-Z0-9]+", "", remaining)
                        if remaining:
                            for f in remaining.split(","):
                                f = f.strip()
                                if f:
                                    current_files.append(f)

                        lint_block_end = j

                        # Check if any line in this block was modified
                        # Exclude the marker lines themselves (lint_block_start and lint_block_end)
                        block_modified = any(
                            is_line_in_ranges(line_num, changed_ranges)
                            for line_num in range(lint_block_start + 1, lint_block_end)
                        )

                        if block_modified and current_files:
                            required_edits.append(set(current_files))

                        in_lint_block = False
                        break
                    else:
                        # Continue collecting files
                        # Parse: strip leading spaces/tabs, then strip until alphanumeric
                        clean_line = next_line.lstrip(" \t")
                        clean_line = re.sub(r"^[^a-zA-Z0-9]+", "", clean_line)
                        if clean_line.strip():
                            for f in clean_line.split(","):
                                f = f.strip()
                                if f:
                                    current_files.append(f)
                        j += 1

    return required_edits


def check_coordinated_edits(base_ref: str, head_ref: str) -> List[dict]:
    """
    Check for coordinated edit violations.

    Returns a list of violations, where each violation is a dict with:
    - file: the file that was modified
    - required: list of files that should have been edited
    - missing: list of files that were not edited
    """
    # Get changed files
    changed_files = get_changed_files(base_ref, head_ref)
    if not changed_files:
        return []

    # Check each changed file for Lint.IfEdited blocks
    violations = []

    for file_path in sorted(changed_files):
        # Skip checking the check_coordinated_edits.py script itself
        if file_path == "ci/check_coordinated_edits.py":
            continue

        # Skip checking the test script
        if file_path == "ci/tests/test_check_coordinated_edits.py":
            continue

        if not os.path.exists(file_path):
            continue

        # Get the line ranges that were modified
        changed_ranges = get_changed_line_ranges(file_path, base_ref, head_ref)
        if not changed_ranges:
            continue

        # Parse lint blocks and check if required files were edited
        required_edits_list = parse_lint_blocks(file_path, changed_ranges)

        for required_files in required_edits_list:
            missing_files = required_files - changed_files
            if missing_files:
                violations.append(
                    {"file": file_path, "required": sorted(required_files), "missing": sorted(missing_files)}
                )

    return violations


def main():
    parser = argparse.ArgumentParser(description="Check coordinated file edits based on Lint.IfEdited/ThenEdit markers")
    parser.add_argument("--base-ref", help="Base reference for comparison")
    parser.add_argument("--head-ref", help="Head reference for comparison", default="HEAD")
    args = parser.parse_args()

    # Determine base ref
    base_ref = args.base_ref
    if not base_ref:
        event_name = os.environ.get("GITHUB_EVENT_NAME", "")
        if event_name == "pull_request":
            base_ref = f"origin/{os.environ.get('GITHUB_BASE_REF', 'main')}"
        elif event_name == "merge_group":
            # For merge groups, compare against the base branch
            github_ref = os.environ.get("GITHUB_REF", "")
            match = re.search(r"gh-readonly-queue/[^/]+/(.+?)/", github_ref)
            if match:
                base_ref = f"origin/{match.group(1)}"
            else:
                base_ref = "origin/main"
        else:
            base_ref = "origin/main"

    print(f"Checking coordinated edits between {base_ref} and {args.head_ref}")

    violations = check_coordinated_edits(base_ref, args.head_ref)

    if not violations:
        # Check if there were any changed files
        changed_files = get_changed_files(base_ref, args.head_ref)
        if not changed_files:
            print("✓ No files changed, skipping check")
        else:
            print(f"Found {len(changed_files)} changed file(s)")
            print("\n✓ All coordinated edits are present")
        sys.exit(0)

    # Report violations
    print("\n❌ Coordinated edit violations found:\n")
    for violation in violations:
        print(f"File: {violation['file']}")
        print(f"  Required edits: {', '.join(violation['required'])}")
        print(f"  Missing edits: {', '.join(violation['missing'])}")
        print()

    print("To skip this check, add the 'skip-if-edited-check' label to the PR, but be very careful!")
    sys.exit(1)


if __name__ == "__main__":
    main()
