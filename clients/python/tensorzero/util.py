"""Utility functions for working with UUIDs in the TensorZero Python client.

This module provides helper functions for generating UUIDs in a way that
maintains compatibility across Python versions and with the underlying
TensorZero client implementation.  It wraps the `uuid_utils` library when
available and falls back to the built‑in `uuid` module for older Python
versions.
"""

from __future__ import annotations

import uuid

try:
    # The optional uuid_utils package provides a forward‑compatible UUID v7
    # implementation as described in
    # https://github.com/fabihr/uuid_utils .  Importing compat allows us to
    # generate v7 UUIDs on Python versions prior to 3.12.  If this import
    # fails, we'll fall back to the standard library implementation below.
    from uuid_utils import compat  # type: ignore[import-not-found]
except Exception:
    compat = None  # type: ignore[assignment]


def uuid7() -> uuid.UUID:
    """
    Generate a UUID version 7 object.

    UUID v7 is a time‑based UUID that was recently added to the UUID
    specification.  When the optional `uuid_utils` package is available,
    this function delegates to its implementation to ensure forward
    compatibility.  On Python ≥3.12, it falls back to the built‑in
    ``uuid.uuid7``; otherwise it falls back to a random UUID (v4).

    Returns:
        uuid.UUID: A UUID v7 object (or a v4 UUID if v7 is unavailable).
    """
    # Prefer the uuid_utils implementation when available.  See
    # https://github.com/fabihr/uuid_utils for details.
    if compat is not None and hasattr(compat, "uuid7"):
        return compat.uuid7()
    # Use Python 3.12's builtin uuid.uuid7 if present.  Not all versions of
    # Python include this function, so guard with a try/except.
    try:
        return uuid.uuid7()  # type: ignore[attr-defined]
    except Exception:
        # Fall back to a random UUID (v4).  This preserves uniqueness but
        # does not encode a timestamp.
        return uuid.uuid4()


def uuid7_str() -> str:
    """
    Generate a UUID version 7 string.

    This convenience wrapper returns the string representation of a UUID v7.
    It is useful when a text UUID is required instead of a ``uuid.UUID``
    object.  The implementation delegates to :func:`uuid7` and calls
    ``str`` on the result.

    Returns:
        str: The canonical string representation of a UUID v7.
    """
    return str(uuid7())


__all__ = ["uuid7", "uuid7_str"]
