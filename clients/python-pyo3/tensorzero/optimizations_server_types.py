from typing import Any, Dict, List

from typing_extensions import NotRequired, Required, TypedDict

# TODO: we should combine some of these types with the existing types in the client, but we need to make sure they don't trample the existing types

# TODO: we should try to specialize the types of `Any` as much as possible

System = str | Dict[str, Any]

Message = Dict[str, Any]


class Input(TypedDict):
    system: NotRequired[System]
    messages: NotRequired[List[Message]]


class Sample(TypedDict):
    input: Required[Input]
    output: Required[Any]
