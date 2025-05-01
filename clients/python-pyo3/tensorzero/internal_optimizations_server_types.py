from typing import Any, List

from typing_extensions import NotRequired, Required, TypedDict

from .types import InferenceInput, Message, System

# TODO: we should combine some of these types with the existing types in the client, but we need to make sure they don't trample the existing types

# TODO: we should try to specialize the types of `Any` as much as possible


class Input(TypedDict):
    system: NotRequired[System]
    messages: NotRequired[List[Message]]


class Sample(TypedDict):
    input: Required[InferenceInput]
    output: Required[Any]
