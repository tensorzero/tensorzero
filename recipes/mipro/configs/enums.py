from enum import Enum


class FunctionConfigType(str, Enum):
    """
    Enumeration of function configuration types.
    """

    CHAT = "chat"
    JSON = "json"
