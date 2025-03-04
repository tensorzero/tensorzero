from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseConfigs(BaseModel, Generic[T]):
    """
    Generic container for configuration objects that acts like a dictionary.
    """

    class Config:
        extra = "allow"

    def __getitem__(self, key: str) -> T:
        """Get the configuration associated with the given key."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: T) -> None:
        """Set the configuration for a given key."""
        setattr(self, key, value)

    def items(self):
        """Return all configuration items."""
        return self.model_dump().items()

    def keys(self):
        """Return all configuration keys."""
        return self.model_dump().keys()
