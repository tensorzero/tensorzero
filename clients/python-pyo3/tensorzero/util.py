from uuid import UUID
from uuid_utils import uuid7 as _uuid7

def uuid7() -> UUID:
    """
    Generate a UUID v7 using uuid_utils and convert it to a standard uuid.UUID.
    This ensures type compatibility with the rest of the TensorZero client.
    """
    return UUID(str(_uuid7()))

__all__ = ["uuid7"]
