import uuid

from uuid_utils import compat


def uuid7() -> uuid.UUID:
    """
    Generate a UUID v7 using uuid_utils compatibility layer.
    This ensures type compatibility with the rest of the TensorZero client.
    """
    return compat.uuid7()


__all__ = ["uuid7"]
