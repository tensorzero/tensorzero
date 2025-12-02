import warnings
from abc import ABC
from dataclasses import dataclass, fields, is_dataclass
from json import JSONEncoder
from typing import Any, Dict, List, Literal, Optional, Protocol, Union
from uuid import UUID

import httpx
import uuid_utils
from typing_extensions import NotRequired, TypedDict, deprecated

from .generated_types import (
    ContentBlockRawText,
    ContentBlockTemplate,
    ContentBlockText,
    ContentBlockThought,
    ContentBlockToolResult,
    ContentBlockUnknown,
    ContentBlockValidatedToolCall,
    FinishReason,
    InferenceFilter,
    InferenceFilterAnd,
    InferenceFilterBooleanMetric,
    InferenceFilterFloatMetric,
    InferenceFilterNot,
    InferenceFilterOr,
    InferenceFilterTag,
    InferenceFilterTime,
    MessageContentObjectStorageFile,
    MessageContentStoredFile,
    MessageContentToolCall,
    OmitType,
    OrderByMetric,
    OrderByTimestamp,
    RenderedSample,
    ResolvedInputMessageContent,
    System,
    Usage,
)


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


# For type checking purposes only
class HasTypeField(Protocol):
    type: str


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
    provider_type: Optional[str] = None


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


def parse_inference_chunk(chunk: Dict[str, Any]) -> InferenceChunk:
    if "content" in chunk:
        return ChatChunk(
            inference_id=UUID(chunk["inference_id"]),
            episode_id=UUID(chunk["episode_id"]),
            variant_name=chunk["variant_name"],
            content=[parse_content_block_chunk(block) for block in chunk["content"]],
            usage=Usage(**chunk["usage"]) if "usage" in chunk else None,
            finish_reason=chunk.get("finish_reason"),
        )
    elif "raw" in chunk:
        return JsonChunk(
            inference_id=UUID(chunk["inference_id"]),
            episode_id=UUID(chunk["episode_id"]),
            variant_name=chunk["variant_name"],
            raw=chunk["raw"],
            usage=Usage(**chunk["usage"]) if "usage" in chunk else None,
            finish_reason=chunk.get("finish_reason"),
        )
    else:
        raise ValueError(f"Unable to determine response type: {chunk}")


def parse_content_block_chunk(block: Dict[str, Any]) -> ContentBlockChunk:
    block_type = block["type"]
    if block_type == "text":
        return TextChunk(id=block["id"], text=block["text"])
    elif block_type == "tool_call":
        return ToolCallChunk(
            id=block["id"],
            raw_arguments=block["raw_arguments"],
            raw_name=block["raw_name"],
        )
    elif block_type == "thought":
        return ThoughtChunk(
            id=block["id"],
            text=block.get("text"),
            signature=block.get("signature"),
            summary_id=block.get("summary_id"),
            summary_text=block.get("summary_text"),
        )
    elif block_type == "unknown":
        return UnknownChunk(
            id=block["id"],
            data=block["data"],
        )

    else:
        raise ValueError(f"Unknown content block type: {block}")


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


def parse_workflow_evaluation_run_response(
    data: Dict[str, Any],
) -> WorkflowEvaluationRunResponse:
    return WorkflowEvaluationRunResponse(run_id=UUID(data["run_id"]))


@dataclass
class WorkflowEvaluationRunEpisodeResponse:
    episode_id: UUID


def parse_workflow_evaluation_run_episode_response(
    data: Dict[str, Any],
) -> WorkflowEvaluationRunEpisodeResponse:
    return WorkflowEvaluationRunEpisodeResponse(episode_id=UUID(data["episode_id"]))


# DEPRECATED: Use WorkflowEvaluationRunResponse instead
DynamicEvaluationRunResponse = WorkflowEvaluationRunResponse


# DEPRECATED: Use parse_workflow_evaluation_run_response instead
def parse_dynamic_evaluation_run_response(
    data: Dict[str, Any],
) -> WorkflowEvaluationRunResponse:
    return parse_workflow_evaluation_run_response(data)


# DEPRECATED: Use WorkflowEvaluationRunEpisodeResponse instead
DynamicEvaluationRunEpisodeResponse = WorkflowEvaluationRunEpisodeResponse


# DEPRECATED: Use parse_workflow_evaluation_run_episode_response instead
def parse_dynamic_evaluation_run_episode_response(
    data: Dict[str, Any],
) -> WorkflowEvaluationRunEpisodeResponse:
    return parse_workflow_evaluation_run_episode_response(data)


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
            # Convert dataclass to dict, but filter out OMIT fields
            result = {}
            for field in fields(o):
                value = getattr(o, field.name)
                # Skip OMIT fields entirely (they won't be in the JSON)
                if not isinstance(value, OmitType):
                    # Recursively handle nested dataclasses/lists/dicts
                    result[field.name] = self._convert_value(value)
            return result  # pyright: ignore[reportUnknownVariableType]
        else:
            super().default(o)

    def _convert_value(self, value: Any) -> Any:
        """Recursively convert values, filtering out OMIT."""
        if isinstance(value, OmitType):
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
                if not isinstance(field_value, OmitType):
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


# The following are all deprecated types that are aliased to the new types.
RenderedStoredInference = RenderedSample
"""Deprecated; use RenderedSample instead. This alias will be removed after 2026.2+."""


# ContentBlock is only used in the context of ResolvedInputMessage, so these types are aliased to the resolved versions. Some of them were incorrect previously.
ContentBlock = ResolvedInputMessageContent
"""Deprecated; use ResolvedInputMessageContent instead. This alias will be removed after 2026.2+."""


UnknownContentBlock = ContentBlockUnknown
"""Deprecated; use ContentBlockUnknown instead. This alias will be removed after 2026.2+."""


Text = ContentBlockText
"""Deprecated; use ContentBlockText instead. This alias will be removed after 2026.2+."""


Template = ContentBlockTemplate
"""Deprecated; use ContentBlockTemplate instead. This alias will be removed after 2026.2+."""


ToolCall = ContentBlockValidatedToolCall
"""Deprecated; use ContentBlockValidatedToolCall instead. This alias will be removed after 2026.2+."""


ToolResult = ContentBlockToolResult
"""Deprecated; use ContentBlockToolResult instead. This alias will be removed after 2026.2+."""


Unknown = ContentBlockUnknown
"""Deprecated; use ContentBlockUnknown instead. This alias will be removed after 2026.2+."""


Thought = ContentBlockThought
"""Deprecated; use ContentBlockThought instead. This alias will be removed after 2026.2+."""


File = MessageContentObjectStorageFile
"""Deprecated; use MessageContentObjectStorageFile instead. This alias will be removed after 2026.2+."""


RawText = ContentBlockRawText
"""Deprecated; use ContentBlockRawText instead. This alias will be removed after 2026.2+."""


@deprecated("Deprecated; use OrderByTimestamp or OrderByMetric instead. This alias will be removed after 2026.2+.")
def OrderBy(
    by: Literal["timestamp", "metric"],
    name: Optional[str] = None,
    direction: Literal["ascending", "descending"] = "descending",
) -> OrderByTimestamp | OrderByMetric:
    """
    DEPRECATED: Do not create OrderBy directly. Use OrderByTimestamp or OrderByMetric instead.

    This function maintains backward compatibility with the old OrderBy class constructor.
    """
    warnings.warn(
        "Do not create OrderBy directly. Use OrderByTimestamp or OrderByMetric instead. "
        "In a future release, this function will be removed.",
        DeprecationWarning,
        stacklevel=2,
    )

    if by == "timestamp":
        return OrderByTimestamp(direction=direction)
    elif by == "metric":
        if name is None:
            raise ValueError("name is required when by='metric'")
        return OrderByMetric(name=name, direction=direction)
    else:
        raise ValueError(f"Invalid value for 'by': {by}. Must be 'timestamp' or 'metric'.")


# Types for the experimental list inferences API
InferenceFilterTreeNode = InferenceFilter
"""Deprecated: Use InferenceFilter instead. This alias will be removed after 2026.2+."""


@deprecated("Deprecated; use InferenceFilterFloatMetric instead. This alias will be removed after 2026.2+.")
class FloatMetricFilter(InferenceFilterFloatMetric):
    """Deprecated: Use InferenceFilterFloatMetric instead. This alias will be removed after 2026.2+."""


@deprecated("Deprecated; use InferenceFilterBooleanMetric instead. This alias will be removed after 2026.2+.")
class BooleanMetricFilter(InferenceFilterBooleanMetric):
    """Deprecated: Use InferenceFilterBooleanMetric instead. This alias will be removed after 2026.2+."""


@deprecated("Deprecated; use InferenceFilterTag instead. This alias will be removed after 2026.2+.")
class TagFilter(InferenceFilterTag):
    """Deprecated: Use InferenceFilterTag instead. This alias will be removed after 2026.2+."""


@deprecated("Deprecated; use InferenceFilterTime instead. This alias will be removed after 2026.2+.")
class TimeFilter(InferenceFilterTime):
    """Deprecated: Use InferenceFilterTime instead. This alias will be removed after 2026.2+."""


@deprecated("Deprecated; use InferenceFilterAnd instead. This alias will be removed after 2026.2+.")
class AndFilter(InferenceFilterAnd):
    """Deprecated: Use InferenceFilterAnd instead. This alias will be removed after 2026.2+."""


@deprecated("Deprecated; use InferenceFilterOr instead. This alias will be removed after 2026.2+.")
class OrFilter(InferenceFilterOr):
    """Deprecated: Use InferenceFilterOr instead. This alias will be removed after 2026.2+."""


@deprecated("Deprecated; use InferenceFilterNot instead. This alias will be removed after 2026.2+.")
class NotFilter(InferenceFilterNot):
    """Deprecated: Use InferenceFilterNot instead. This alias will be removed after 2026.2+."""


# Old generated types
@deprecated("Deprecated; use DICLConfig instead. This alias will be removed after 2026.2+.")
class ContentBlockChatOutputText(ContentBlockText):
    """Deprecated: Use ContentBlockText instead. This alias will be removed after 2026.2+."""


@deprecated("Deprecated; use DICLConfig instead. This alias will be removed after 2026.2+.")
class ContentBlockChatOutputToolCall(ContentBlockValidatedToolCall):
    """Deprecated: Use ContentBlockValidatedToolCall instead. This alias will be removed after 2026.2+."""


@deprecated("Deprecated; use ContentBlockText instead. This alias will be removed after 2026.2+.")
class InputContentBlockText(ContentBlockText):
    """Deprecated: Use ContentBlockText instead. This alias will be removed after 2026.2+."""


@deprecated("Deprecated; use ContentBlockTemplate instead. This alias will be removed after 2026.2+.")
class InputContentBlockTemplate(ContentBlockTemplate):
    """Deprecated: Use ContentBlockTemplate instead. This alias will be removed after 2026.2+."""


@deprecated("Deprecated; use MessageContentStoredFile instead. This alias will be removed after 2026.2+.")
class StoredInputMessageContentFile(MessageContentStoredFile):
    """Deprecated: Use MessageContentStoredFile instead. This alias will be removed after 2026.2+."""


@deprecated("Deprecated; use ContentBlockTemplate instead. This alias will be removed after 2026.2+.")
class StoredInputContentBlockTemplate(ContentBlockTemplate):
    """Deprecated: Use ContentBlockTemplate instead. This alias will be removed after 2026.2+."""


@deprecated("Deprecated; use ContentBlockText instead. This alias will be removed after 2026.2+.")
class StoredInputContentBlockText(ContentBlockText):
    """Deprecated: Use ContentBlockText instead. This alias will be removed after 2026.2+."""


@deprecated("Deprecated; use ContentBlockThought instead. This alias will be removed after 2026.2+.")
class StoredInputContentBlockThought(ContentBlockThought):
    """Deprecated: Use ContentBlockThought instead. This alias will be removed after 2026.2+."""


@deprecated("Deprecated; use MessageContentToolCall instead. This alias will be removed after 2026.2+.")
class StoredInputMessageContentToolCall(MessageContentToolCall):
    """Deprecated: Use MessageContentToolCall instead. This alias will be removed after 2026.2+."""


@deprecated("Deprecated; use ContentBlockToolResult instead. This alias will be removed after 2026.2+.")
class StoredInputContentBlockToolResult(ContentBlockToolResult):
    """Deprecated: Use ContentBlockToolResult instead. This alias will be removed after 2026.2+."""


@deprecated("Deprecated; use ContentBlockUnknown instead. This alias will be removed after 2026.2+.")
class StoredInputContentBlockUnknown(ContentBlockUnknown):
    """Deprecated: Use ContentBlockUnknown instead. This alias will be removed after 2026.2+."""
