import warnings
from abc import ABC
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from json import JSONEncoder
from typing import Any, Dict, List, Literal, Optional, Protocol, Union, cast
from uuid import UUID

import httpx
import uuid_utils
from typing_extensions import NotRequired, TypedDict


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


# For type checking purposes only
class HasTypeField(Protocol):
    type: str


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
class UnknownContentBlock(ContentBlock):
    data: Any
    model_provider_name: Optional[str] = None
    type: str = "unknown"


class FinishReason(str, Enum):
    STOP = "stop"
    STOP_SEQUENCE = "stop_sequence"
    LENGTH = "length"
    TOOL_CALL = "tool_call"
    CONTENT_FILTER = "content_filter"
    UNKNOWN = "unknown"


@dataclass
class JsonInferenceOutput:
    raw: Optional[str] = None
    parsed: Optional[Dict[str, Any]] = None


@dataclass
class ChatInferenceResponse:
    inference_id: UUID
    episode_id: UUID
    variant_name: str
    content: List[ContentBlock]
    usage: Usage
    finish_reason: Optional[FinishReason] = None
    original_response: Optional[str] = None


@dataclass
class JsonInferenceResponse:
    inference_id: UUID
    episode_id: UUID
    variant_name: str
    output: JsonInferenceOutput
    usage: Usage
    finish_reason: Optional[FinishReason] = None
    original_response: Optional[str] = None


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: Any


System = Union[str, Dict[str, Any]]


class InferenceInput(TypedDict):
    messages: NotRequired[List[Message]]
    system: NotRequired[System]


class EvaluatorStatsDict(TypedDict):
    """Statistics computed about a particular evaluator."""

    mean: float
    stderr: float
    count: int


InferenceResponse = Union[ChatInferenceResponse, JsonInferenceResponse]


def parse_inference_response(data: Dict[str, Any]) -> InferenceResponse:
    if "content" in data and isinstance(data["content"], list):
        finish_reason = data.get("finish_reason")
        finish_reason_enum = FinishReason(finish_reason) if finish_reason else None

        return ChatInferenceResponse(
            inference_id=UUID(data["inference_id"]),
            episode_id=UUID(data["episode_id"]),
            variant_name=data["variant_name"],
            content=[parse_content_block(block) for block in data["content"]],  # type: ignore
            usage=Usage(**data["usage"]),
            finish_reason=finish_reason_enum,
            original_response=data.get("original_response"),
        )
    elif "output" in data and isinstance(data["output"], dict):
        output = cast(Dict[str, Any], data["output"])
        finish_reason = data.get("finish_reason")
        finish_reason_enum = FinishReason(finish_reason) if finish_reason else None

        return JsonInferenceResponse(
            inference_id=UUID(data["inference_id"]),
            episode_id=UUID(data["episode_id"]),
            variant_name=data["variant_name"],
            output=JsonInferenceOutput(**output),
            usage=Usage(**data["usage"]),
            finish_reason=finish_reason_enum,
            original_response=data.get("original_response"),
        )
    else:
        raise ValueError("Unable to determine response type")


def parse_content_block(block: Dict[str, Any]) -> ContentBlock:
    block_type = block["type"]
    if block_type == "text":
        return Text(text=block["text"], type=block_type)
    elif block_type == "tool_call":
        return ToolCall(
            arguments=block.get("arguments"),
            id=block["id"],
            name=block.get("name"),
            raw_arguments=block["raw_arguments"],
            raw_name=block["raw_name"],
            type=block_type,
        )
    elif block_type == "thought":
        summary_data = block.get("summary")
        summary = None
        if summary_data:
            summary = [ThoughtSummaryBlock(text=s["text"]) for s in summary_data]
        return Thought(
            text=block.get("text"),
            signature=block.get("signature"),
            summary=summary,
            type=block_type,
        )
    elif block_type == "unknown":
        return UnknownContentBlock(
            data=block["data"],
            model_provider_name=block.get("model_provider_name"),
        )
    else:
        raise ValueError(f"Unknown content block type: {block}")


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


def parse_inference_chunk(chunk: Dict[str, Any]) -> InferenceChunk:
    finish_reason = chunk.get("finish_reason")
    finish_reason_enum = FinishReason(finish_reason) if finish_reason else None

    if "content" in chunk:
        return ChatChunk(
            inference_id=UUID(chunk["inference_id"]),
            episode_id=UUID(chunk["episode_id"]),
            variant_name=chunk["variant_name"],
            content=[parse_content_block_chunk(block) for block in chunk["content"]],
            usage=Usage(**chunk["usage"]) if "usage" in chunk else None,
            finish_reason=finish_reason_enum,
        )
    elif "raw" in chunk:
        return JsonChunk(
            inference_id=UUID(chunk["inference_id"]),
            episode_id=UUID(chunk["episode_id"]),
            variant_name=chunk["variant_name"],
            raw=chunk["raw"],
            usage=Usage(**chunk["usage"]) if "usage" in chunk else None,
            finish_reason=finish_reason_enum,
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


# Helper used to serialize Python objects to JSON, which may contain dataclasses like `Text`
# Used by the Rust native module
class TensorZeroTypeEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, UUID) or isinstance(o, uuid_utils.UUID):
            return str(o)
        elif hasattr(o, "to_dict"):
            return o.to_dict()
        elif is_dataclass(o) and not isinstance(o, type):
            return asdict(o)
        else:
            super().default(o)


ToolChoice = Union[Literal["auto", "required", "off"], Dict[Literal["specific"], str]]


# Types for the experimental list inferences API
# These are serialized across the PyO3 boundary


@dataclass
class InferenceFilter(ABC, HasTypeField):
    pass


# DEPRECATED: Use InferenceFilter instead
InferenceFilterTreeNode = InferenceFilter


@dataclass
class FloatMetricFilter(InferenceFilter):
    metric_name: str
    value: float
    comparison_operator: Literal["<", "<=", "=", ">", ">=", "!="]
    type: str = "float_metric"


@dataclass
class BooleanMetricFilter(InferenceFilter):
    metric_name: str
    value: bool
    type: str = "boolean_metric"


@dataclass
class TagFilter(InferenceFilter):
    key: str
    value: str
    comparison_operator: Literal["=", "!="]
    type: str = "tag"


@dataclass
class TimeFilter(InferenceFilter):
    time: str  # RFC 3339 timestamp
    comparison_operator: Literal["<", "<=", "=", ">", ">=", "!="]
    type: str = "time"


@dataclass
class AndFilter(InferenceFilter):
    children: List[InferenceFilter]
    type: str = "and"


@dataclass
class OrFilter(InferenceFilter):
    children: List[InferenceFilter]
    type: str = "or"


@dataclass
class NotFilter(InferenceFilter):
    child: InferenceFilter
    type: str = "not"


@dataclass
class OrderBy:
    by: Literal["timestamp", "metric"]
    name: Optional[str] = None
    direction: Literal["ascending", "descending"] = "descending"
