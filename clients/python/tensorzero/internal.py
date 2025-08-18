from dataclasses import dataclass
from typing import List, Literal, Optional

from tensorzero.types import ContentBlock, Tool

# Note: the auxiliary types below before the clients are never actually constructed by the client and should not be constructed directly.
# They are part of the type checking interface only.

# In the future, we may convert these to constructible types for optional construction and passing into other TensorZero methods.
# In that case, we should add a constructor in Rust to the types and test against raw Python types as well.


# NOTE: there are two different message types. This one is to hint that
# the RenderedStoredInference.input.messages field is a list of this type.
# The other one is for inputs and uses Any on the way in.
# This must change as the Python codebase evolves.
@dataclass
class OutputMessage:
    role: Literal["user", "assistant"]
    content: List[ContentBlock]


@dataclass
class ModelInput:
    system: Optional[str]
    messages: List[OutputMessage]


@dataclass
class ToolCallConfigDatabaseInsert:
    tools_available: List[Tool]
    # tool_choice: ToolChoice
    # The Rust codebase doesn't expose this yet
    parallel_tool_calls: Optional[bool]
