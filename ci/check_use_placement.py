#!/usr/bin/env python3
"""
Check that private `use` statements are placed correctly:
1. Not interleaved after function definitions at the same scope level.
2. Not inside function bodies (should be at module level instead).

This catches common AI coding assistant mistakes where `use` imports are
appended after existing code or placed inside function bodies rather than
grouped at the top of the enclosing module/scope.

Conservative: false negatives are acceptable, false positives are not.

Strategy: track brace depth to identify scope boundaries. Within each scope,
flag private `use` statements that appear after `fn` definitions at the same
brace depth. Also track function body scopes and flag `use` inside them.
`pub use` re-exports are excluded since they are commonly placed alongside
the items they re-export.

Usage:
    python ci/check_use_placement.py [--warn-fn-body] [paths...]

    If no paths given, checks all .rs files under crates/.

    --warn-fn-body  Also check for `use` inside function bodies (warning only,
                    does not cause a non-zero exit code).
"""

import re
import sys
from pathlib import Path

# Matches a private `use` statement (NOT `pub use` re-exports)
PRIVATE_USE_RE = re.compile(r"^\s*use\s+")

# Matches a `pub use` re-export (excluded from checks)
PUB_USE_RE = re.compile(r"^\s*pub(\s*\(.*?\))?\s+use\s+")

# Matches a `fn` definition (with optional visibility/async/unsafe/const modifiers)
FN_RE = re.compile(r"^\s*(pub(\s*\(.*?\))?\s+)?(async\s+)?(unsafe\s+)?(const\s+)?fn\s+")

# Files/dirs to skip (generated code, macro expansions)
SKIP_PATTERNS = [
    "/target/",
    "\\target\\",
    ".expanded.rs",
    "/bindings/",
]


def count_braces(line: str) -> int:
    """Count net brace change in a line, skipping strings and comments."""
    # Strip line comments
    comment_pos = line.find("//")
    if comment_pos >= 0:
        line = line[:comment_pos]
    depth = 0
    in_string = False
    escape = False
    for ch in line:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if in_string:
            if ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
    return depth


def check_source(source: str) -> list[tuple[int, str]]:
    """Check source code string. Returns list of (line_number, line_text) violations."""
    lines = source.splitlines()
    return _check_lines(lines)


def check_file(path: Path) -> list[tuple[int, str]]:
    """Returns list of (line_number, line_text) for misplaced use statements."""
    try:
        source = path.read_text()
    except (OSError, UnicodeDecodeError):
        return []
    return check_source(source)


def _check_lines(lines: list[str]) -> list[tuple[int, str]]:
    violations = []
    brace_depth = 0
    # Per brace depth: have we seen a fn definition?
    # Reset when leaving a scope.
    fn_seen: dict[int, bool] = {}
    # Set of brace depths that are function body scopes.
    # When a fn opens a brace, the depth inside is a fn body.
    fn_body_depths: set[int] = set()

    for i, line in enumerate(lines, 1):
        stripped = line.lstrip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            continue

        net = count_braces(stripped)

        # Process closing braces: leaving scope(s), clear state for exited scopes.
        # When going from depth 2 to 1, clear depth 2+ (the fn body we're leaving),
        # but preserve depth 1 (the scope we're returning to — fn decl still counts).
        if net < 0:
            new_depth = brace_depth + net
            fn_seen = {k: v for k, v in fn_seen.items() if k <= new_depth}
            fn_body_depths = {d for d in fn_body_depths if d <= new_depth}

        # Determine the depth for this line's content.
        # For `}` lines, the content belongs to the outer scope (after closing).
        # For `fn foo() {` lines, `fn` belongs to current depth before the `{`.
        # We handle this by checking content at the "minimum" depth:
        # if net < 0, content is at the new (lower) depth.
        # if net >= 0, content is at the current depth before opening.
        if net < 0:
            content_depth = brace_depth + net
        else:
            content_depth = brace_depth

        # Check if we're inside any function body
        in_fn_body = any(d <= content_depth for d in fn_body_depths)

        # Check content
        if PUB_USE_RE.match(line):
            pass  # Skip pub use re-exports
        elif PRIVATE_USE_RE.match(line):
            if in_fn_body:
                violations.append((i, line.rstrip(), "inside_fn"))
            elif fn_seen.get(content_depth, False):
                violations.append((i, line.rstrip(), "after_fn"))
        elif FN_RE.match(line):
            fn_seen[content_depth] = True
            if net > 0:
                # fn opens a brace — the inside is a fn body
                fn_body_depths.add(brace_depth + net)
            elif net == 0 and "{" not in stripped:
                # fn signature without body on this line (brace on next line)
                fn_body_depths.add(brace_depth + 1)

        # Update depth
        brace_depth += net

    return violations


def should_skip(path: Path) -> bool:
    s = str(path)
    return any(pat in s for pat in SKIP_PATTERNS)


def main() -> int:
    warn_fn_body = "--warn-fn-body" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if args:
        paths = [Path(p) for p in args]
        rs_files = []
        for p in paths:
            if p.is_file() and p.suffix == ".rs":
                rs_files.append(p)
            elif p.is_dir():
                rs_files.extend(p.rglob("*.rs"))
    else:
        rs_files = list(Path("crates").rglob("*.rs"))

    rs_files = [f for f in rs_files if not should_skip(f)]

    error_count = 0
    warning_count = 0
    for path in sorted(rs_files):
        violations = check_file(path)
        for violation in violations:
            line_no, line_text = violation[0], violation[1]
            reason = violation[2] if len(violation) > 2 else "after_fn"
            if reason == "inside_fn":
                if not warn_fn_body:
                    continue
                msg = "warning: `use` inside function body (move to module level)"
                warning_count += 1
            else:
                msg = "error: `use` after `fn` at same scope level"
                error_count += 1
            print(f"{path}:{line_no}: {msg}")
            print(f"  {line_text}")

    if error_count > 0:
        print(f"\nFound {error_count} misplaced `use` statement(s). Move them to the top of their enclosing scope.")
    if warning_count > 0:
        print(
            f"\nFound {warning_count} `use`-inside-function-body warning(s). "
            "Consider moving them to the top of the enclosing module."
        )
    if error_count == 0 and warning_count == 0:
        print("No misplaced `use` statements found.")

    # Only fail on errors, not warnings
    return 1 if error_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
