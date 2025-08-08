from uuid import UUID

from uuid_utils import compat


def uuid7() -> UUID:
    """
    Generate a UUIDv7 using the uuid_utils compatibility layer.
    This ensures type compatibility with the rest of the TensorZero client.
    """
    return compat.uuid7()


__all__ = [
    "UUID",
    "uuid7",
]
