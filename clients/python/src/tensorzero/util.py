"""Utility functions for UUID generation.

This module re-exports the uuid7 function from uuid_utils for generating UUIDs.
"""

from uuid_utils import (
    uuid7 as uuid7,  # type: ignore # uuid_utils doesn't have type hints
)

__all__: list[str] = ["uuid7"]
