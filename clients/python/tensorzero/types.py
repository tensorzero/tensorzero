import warnings
from abc import ABC
from dataclasses import dataclass, fields, is_dataclass
from json import JSONEncoder
from typing import Any, Dict, List, Literal, Optional, Protocol, Union, cast
from uuid import UUID

import httpx
import uuid_utils
from typing_extensions import NotRequired, TypedDict, deprecated

from tensorzero.generated_types import (
    ChatInferenceResponse,
    ContentBlockChatOutput,
    ContentBlockChatOutputText,
    ContentBlockChatOutputThought,
    ContentBlockChatOutputToolCall,
    ContentBlockChatOutputUnknown,
    FinishReason,
    InferenceFilter,
    InferenceFilterAnd,
    InferenceFilterBooleanMetric,
    InferenceFilterFloatMetric,
    InferenceFilterNot,
    InferenceFilterOr,
    InferenceFilterTag,
    InferenceFilterTime,
    JsonInferenceOutput,
    System,
    UnsetType,
    Usage,
)
from tensorzero.generated_types import (
    ThoughtSummaryBlock as ThoughtSummaryBlockGenerated,
)


# For type checking purposes only
class HasTypeField(Protocol):
    type: str


# Input ContentBlock types (used for constructing messages)
@dataclass
class ContentBlock(ABC, HasTypeField):
    pass


@dataclass
class Template(ContentBlock):
    name: str
    arguments: Any
    type: str = "template"


@dataclass
class Text(ContentBlock):
    """Input text content block with deprecation support for arguments field."""

    text: Optional[str] = None
    arguments: Optional[Any] = None
    type: str = "text"

    def __post_init__(self):
        if self.text is None and self.arguments is None:
            raise ValueError("Either `text` or `arguments` must be provided.")

        if self.text is not None and self.arguments is not None:
            raise ValueError("Only one of `text` or `arguments` must be provided.")

        # Warn about on going deprecation: https://github.com/tensorzero/tensorzero/issues/1170
        if self.text is not None and not isinstance(self.text, str):  # pyright: ignore [reportUnnecessaryIsInstance]
            warnings.warn(
                'Please use `ContentBlock(type="text", arguments=...)` when providing arguments for a prompt template/schema. In a future release, `Text(type="text", text=...)` will require a string literal.',
                DeprecationWarning,
                stacklevel=2,
            )

    def to_dict(self) -> Dict[str, Any]:
        if self.text is not None:
            # Handle ongoing deprecation: https://github.com/tensorzero/tensorzero/issues/1170
            # The first branch will be removed in a future release.
            if isinstance(self.text, dict):
                return dict(type="text", arguments=self.text)
            else:
                return dict(type="text", text=self.text)
        elif self.arguments is not None:
            return dict(type="text", arguments=self.arguments)
        else:
            raise ValueError("Either `text` or `arguments` must be provided.")


@dataclass
class RawText(ContentBlock):
    value: str
    type: str = "raw_text"


@dataclass
class ImageBase64(ContentBlock):
    data: Optional[str]
    mime_type: str
    type: str = "image"


@dataclass
class FileBase64(ContentBlock):
    data: Optional[str]
    mime_type: str
    type: str = "file"


Detail = Literal["low", "high", "auto"]


@dataclass
class ImageUrl(ContentBlock):
    url: str
    mime_type: Optional[str] = None
    detail: Optional[Detail] = None
    type: str = "image"


@dataclass
class FileUrl(ContentBlock):
    url: str
    type: str = "file"


@dataclass
class ToolCall(ContentBlock):
    """Input tool call content block for constructing messages."""

    id: str
    raw_arguments: str
    raw_name: str
    arguments: Optional[Dict[str, Any]] = None
    name: Optional[str] = None
    type: str = "tool_call"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "raw_arguments": self.raw_arguments,
            "raw_name": self.raw_name,
            "type": "tool_call",
        }
        if self.arguments is not None:
            d["arguments"] = self.arguments
        if self.name is not None:
            d["name"] = self.name
        return d


@dataclass
class ThoughtSummaryBlock:
    text: str
    type: str = "summary_text"


@dataclass
class Thought(ContentBlock):
    """Input thought content block for constructing messages."""

    text: Optional[str] = None
    type: str = "thought"
    signature: Optional[str] = None
    summary: Optional[List["ThoughtSummaryBlock"]] = None
    _internal_provider_type: Optional[str] = None


@dataclass
class ToolResult(ContentBlock):
    name: str
    result: str
    id: str
    type: str = "tool_result"


@dataclass
class JsonInferenceResponse:
    inference_id: UUID
    episode_id: UUID
    variant_name: str
    output: "JsonInferenceOutput"  # Imported from generated_types
    usage: Usage
    finish_reason: Optional[FinishReason] = None
    original_response: Optional[str] = None


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: Any


class InferenceInput(TypedDict):
    messages: NotRequired[List[Message]]
    system: NotRequired[System]


class EvaluatorStatsDict(TypedDict):
    """Statistics computed about a particular evaluator."""

    mean: float
    stderr: float
    count: int


InferenceResponse = Union[ChatInferenceResponse, JsonInferenceResponse]

# Note: parse_inference_response, parse_content_block_output, and parse_content_block
# have been removed. These are now handled by dacite in Rust via convert_response_to_python.


# Types for streaming inference responses


@dataclass
class ContentBlockChunk(ABC, HasTypeField):
    pass


@dataclass
class TextChunk(ContentBlockChunk):
    # In the possibility that multiple text messages are sent in a single streaming response,
    # this `id` will be used to disambiguate them
    id: str
    text: str
    type: str = "text"


@dataclass
class ToolCallChunk(ContentBlockChunk):
    # This is the tool call ID that many LLM APIs use to associate tool calls with tool responses
    id: str
    # `raw_arguments` will come as partial JSON
    raw_arguments: str
    raw_name: str
    type: str = "tool_call"


@dataclass
class ThoughtChunk(ContentBlockChunk):
    id: str
    text: Optional[str]
    type: str = "thought"
    signature: Optional[str] = None
    summary_id: Optional[str] = None
    summary_text: Optional[str] = None
    _internal_provider_type: Optional[str] = None


@dataclass
class UnknownChunk(ContentBlockChunk):
    id: str
    data: Any
    type: str = "unknown"


@dataclass
class ChatChunk:
    inference_id: UUID
    episode_id: UUID
    variant_name: str
    content: List[ContentBlockChunk]
    usage: Optional[Usage] = None
    finish_reason: Optional[FinishReason] = None


@dataclass
class JsonChunk:
    inference_id: UUID
    episode_id: UUID
    variant_name: str
    raw: str
    usage: Optional[Usage] = None
    finish_reason: Optional[FinishReason] = None


InferenceChunk = Union[ChatChunk, JsonChunk]


class VariantExtraBody(TypedDict):
    variant_name: str
    pointer: str
    value: NotRequired[Any]
    delete: NotRequired[bool]


class ProviderExtraBody(TypedDict):
    model_provider_name: str
    pointer: str
    value: NotRequired[Any]
    delete: NotRequired[bool]


ExtraBody = Union[VariantExtraBody, ProviderExtraBody]

# Note: parse_inference_chunk and parse_content_block_chunk have been removed.
# These are now handled by dacite in Rust via convert_response_to_python.


# Types for feedback
@dataclass
class FeedbackResponse:
    feedback_id: UUID


class BaseTensorZeroError(Exception):
    def __init__(self):
        pass


class TensorZeroInternalError(BaseTensorZeroError):
    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self) -> str:
        return self.msg


class TensorZeroError(BaseTensorZeroError):
    def __init__(self, status_code: int, text: Optional[str] = None):
        self.text = text
        self.status_code = status_code
        self._response = httpx.Response(status_code=status_code, text=text)

    @property
    def response(self) -> httpx.Response:
        warnings.warn(
            "TensorZeroError.response is deprecated - use '.text' and '.status_code' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._response

    def __str__(self) -> str:
        return f"TensorZeroError (status code {self.status_code}): {self.text}"


@dataclass
class WorkflowEvaluationRunResponse:
    run_id: UUID


@dataclass
class WorkflowEvaluationRunEpisodeResponse:
    episode_id: UUID


# DEPRECATED: Use WorkflowEvaluationRunResponse instead
DynamicEvaluationRunResponse = WorkflowEvaluationRunResponse


# DEPRECATED: Use WorkflowEvaluationRunEpisodeResponse instead
DynamicEvaluationRunEpisodeResponse = WorkflowEvaluationRunEpisodeResponse

# Note: parse_workflow_evaluation_run_response, parse_workflow_evaluation_run_episode_response,
# and their deprecated aliases have been removed. These are now handled by dacite in Rust
# via convert_response_to_python.


@dataclass
class ChatDatapointInsert:
    function_name: str
    input: InferenceInput
    output: Optional[Any] = None
    allowed_tools: Optional[List[str]] = None
    additional_tools: Optional[List[Any]] = None
    tool_choice: Optional[str] = None
    parallel_tool_calls: Optional[bool] = None
    tags: Optional[Dict[str, str]] = None
    name: Optional[str] = None


@dataclass
class JsonDatapointInsert:
    function_name: str
    input: InferenceInput
    output: Optional[Any] = None
    output_schema: Optional[Any] = None
    tags: Optional[Dict[str, str]] = None
    name: Optional[str] = None


@dataclass
class Tool:
    description: str
    parameters: Any
    name: str
    strict: bool


@dataclass
class ToolParams:
    """Legacy ToolParams class for backward compatibility.

    Use the flattened DynamicToolParams fields directly when constructing StoredInference:
    - allowed_tools: Optional[List[str]]
    - additional_tools: Optional[List[Tool]]
    - tool_choice: Optional[str]
    - parallel_tool_calls: Optional[bool]
    - provider_tools: Optional[List[ProviderTool]]
    """

    tools_available: List[Tool]
    tool_choice: str
    parallel_tool_calls: Optional[bool] = None


class TensorZeroTypeEncoder(JSONEncoder):
    """
    Helper used to serialize Python objects to JSON, which may contain dataclasses like `Text`
    Used by the Rust native module
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, UUID) or isinstance(o, uuid_utils.UUID):
            return str(o)
        elif hasattr(o, "to_dict"):
            return o.to_dict()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]
        elif is_dataclass(o) and not isinstance(o, type):
            # Convert dataclass to dict, but filter out UNSET fields
            result = {}
            for field in fields(o):
                value = getattr(o, field.name)
                # Skip UNSET fields entirely (they won't be in the JSON)
                if not isinstance(value, UnsetType):
                    # Recursively handle nested dataclasses/lists/dicts
                    result[field.name] = self._convert_value(value)
            return result  # pyright: ignore[reportUnknownVariableType]
        else:
            super().default(o)

    def _convert_value(self, value: Any) -> Any:
        """Recursively convert values, filtering out UNSET."""
        if isinstance(value, UnsetType):
            # This shouldn't happen at top level, but handle it just in case
            return None
        elif hasattr(value, "to_dict"):
            return value.to_dict()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]
        elif is_dataclass(value) and not isinstance(value, type):
            # Recursively convert nested dataclasses
            # Note: pyright can't infer types through dict operations when value is Any
            result: dict[str, Any] = {}
            for field in fields(value):
                field_value = getattr(value, field.name)
                if not isinstance(field_value, UnsetType):
                    result[field.name] = self._convert_value(field_value)
            return result  # pyright: ignore[reportUnknownVariableType]
        elif isinstance(value, (list, tuple)):
            # Handle lists/tuples
            return [self._convert_value(item) for item in value]  # pyright: ignore[reportUnknownVariableType]
        elif isinstance(value, dict):
            # Handle dicts
            return {k: self._convert_value(v) for k, v in value.items()}  # pyright: ignore[reportUnknownVariableType]
        else:
            # Return as-is for primitive types
            return value


ToolChoice = Union[Literal["auto", "required", "off"], Dict[Literal["specific"], str]]


# Types for the experimental list inferences API


InferenceFilterTreeNode = InferenceFilter
"""Deprecated: Use InferenceFilter instead."""


@deprecated("Deprecated; use InferenceFilterFloatMetric instead. This alias will be removed in a future version.")
class FloatMetricFilter(InferenceFilterFloatMetric):
    """Deprecated: Use InferenceFilterFloatMetric instead."""

    pass


@deprecated("Deprecated; use InferenceFilterBooleanMetric instead. This alias will be removed in a future version.")
class BooleanMetricFilter(InferenceFilterBooleanMetric):
    """Deprecated: Use InferenceFilterBooleanMetric instead."""

    pass


@deprecated("Deprecated; use InferenceFilterTag instead. This alias will be removed in a future version.")
class TagFilter(InferenceFilterTag):
    """Deprecated: Use InferenceFilterTag instead."""

    pass


@deprecated("Deprecated; use InferenceFilterTime instead. This alias will be removed in a future version.")
class TimeFilter(InferenceFilterTime):
    """Deprecated: Use InferenceFilterTime instead."""

    pass


@deprecated("Deprecated; use InferenceFilterAnd instead. This alias will be removed in a future version.")
class AndFilter(InferenceFilterAnd):
    """Deprecated: Use InferenceFilterAnd instead."""

    pass


@deprecated("Deprecated; use InferenceFilterOr instead. This alias will be removed in a future version.")
class OrFilter(InferenceFilterOr):
    """Deprecated: Use InferenceFilterOr instead."""

    pass


@deprecated("Deprecated; use InferenceFilterNot instead. This alias will be removed in a future version.")
class NotFilter(InferenceFilterNot):
    """Deprecated: Use InferenceFilterNot instead."""

    pass


@dataclass
class OrderBy:
    by: Literal["timestamp", "metric"]
    name: Optional[str] = None
    direction: Literal["ascending", "descending"] = "descending"
