from uuid_utils import compat as uuid

def uuid7() -> uuid.UUID:
    """
    Generate a UUID v7 using uuid_utils compatibility layer.
    This ensures type compatibility with the rest of the TensorZero client.
    """
    return uuid.uuid7()

__all__ = ["uuid7"]
