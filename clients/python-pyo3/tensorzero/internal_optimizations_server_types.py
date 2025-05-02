# IMPORTANT:
#
# This file is not part of the public API of the tensorzero package.
# The types in this file are unstable and subject to change at any time.


from typing import Any

from typing_extensions import Required, TypedDict

from .types import InferenceInput

# TODO: we should combine some of these types with the existing types in the client, but we need to make sure they don't trample the existing types

# TODO: we should try to specialize the types of `Any` as much as possible


# This is an inference row in the database
class Sample(TypedDict):
    input: Required[InferenceInput]
    output: Required[Any]
    # non-exhaustive
