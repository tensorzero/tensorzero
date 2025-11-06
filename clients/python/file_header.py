

class _UnsetType:
    """Sentinel value to distinguish between omitted fields and null values."""

    def __repr__(self):
        return "UNSET"


UNSET = _UnsetType()
"""
Sentinel value to distinguish between omitted and null in API requests.

Usage:
- UNSET: Field is omitted (don't change existing value)
- None: Field is explicitly set to null
- value: Field is set to the provided value
"""
