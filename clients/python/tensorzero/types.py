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
    data: str
    mime_type: str
    type: str = "image"


@dataclass
class FileBase64(ContentBlock):
    data: str
    mime_type: str
    type: str = "file"


@dataclass
class ImageUrl(ContentBlock):
    url: str
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
class Thought(ContentBlock):
    text: str
    type: str = "thought"


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
        return Thought(text=block["text"], type=block_type)
    elif block_type == "unknown":
        return UnknownContentBlock(
            data=block["data"], model_provider_name=block.get("model_provider_name")
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
    text: str
    type: str = "thought"


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
    value: Any


class ProviderExtraBody(TypedDict):
    model_provider_name: str
    pointer: str
    value: Any


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
        return ThoughtChunk(id=block["id"], text=block["text"])
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
class DynamicEvaluationRunResponse:
    run_id: UUID


def parse_dynamic_evaluation_run_response(
    data: Dict[str, Any],
) -> DynamicEvaluationRunResponse:
    return DynamicEvaluationRunResponse(run_id=UUID(data["run_id"]))


@dataclass
class DynamicEvaluationRunEpisodeResponse:
    episode_id: UUID


def parse_dynamic_evaluation_run_episode_response(
    data: Dict[str, Any],
) -> DynamicEvaluationRunEpisodeResponse:
    return DynamicEvaluationRunEpisodeResponse(episode_id=UUID(data["episode_id"]))


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


# CAREFUL: deprecated
class ChatInferenceDatapointInput(ChatDatapointInsert):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "Please use `ChatDatapointInsert` instead of `ChatInferenceDatapointInput`. In a future release, `ChatInferenceDatapointInput` will be removed.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


@dataclass
class JsonDatapointInsert:
    function_name: str
    input: InferenceInput
    output: Optional[Any] = None
    output_schema: Optional[Any] = None
    tags: Optional[Dict[str, str]] = None


# CAREFUL: deprecated
class JsonInferenceDatapointInput(JsonDatapointInsert):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "Please use `JsonDatapointInsert` instead of `JsonInferenceDatapointInput`. In a future release, `JsonInferenceDatapointInput` will be removed.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


@dataclass
class Tool:
    description: str
    parameters: Any
    name: str
    strict: bool


@dataclass
class ToolParams:
    tools_available: List[Tool]
    tool_choice: str
    parallel_tool_calls: Optional[bool] = None


@dataclass
class ChatDatapoint:
    dataset_name: str
    function_name: str
    id: UUID
    input: InferenceInput
    episode_id: Optional[UUID] = None
    output: Optional[List[ContentBlock]] = None
    tool_params: Optional[ToolParams] = None
    tags: Optional[Dict[str, str]] = None
    # `auxiliary` is not serialized yet
    source_inference_id: Optional[UUID] = None
    staled_at: Optional[str] = None
    is_deleted: bool = False


@dataclass
class JsonDatapoint:
    dataset_name: str
    function_name: str
    id: UUID
    input: InferenceInput
    episode_id: Optional[UUID] = None
    output: Optional[JsonInferenceOutput] = None
    output_schema: Optional[Any] = None
    tags: Optional[Dict[str, str]] = None
    # `auxiliary` is not serialized yet
    source_inference_id: Optional[UUID] = None
    staled_at: Optional[str] = None
    is_deleted: bool = False


Datapoint = Union[ChatDatapoint, JsonDatapoint]


def parse_datapoint(data: Dict[str, Any]) -> Datapoint:
    datapoint_type = data.pop("type")
    if datapoint_type == "json":
        return JsonDatapoint(**data)
    elif datapoint_type == "chat":
        return ChatDatapoint(**data)
    else:
        raise ValueError(f"Unknown datapoint type: {datapoint_type}")


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


@dataclass
class StoredChatInference:
    function_name: str
    variant_name: str
    input: InferenceInput
    output: List[ContentBlock]
    episode_id: UUID
    inference_id: UUID
    tool_params: ToolParams
    type: Literal["chat"] = "chat"


@dataclass
class StoredJsonInference:
    function_name: str
    variant_name: str
    input: InferenceInput
    output: JsonInferenceOutput
    episode_id: UUID
    inference_id: UUID
    output_schema: Any
    type: Literal["json"] = "json"


StoredInference = Union[StoredChatInference, StoredJsonInference]


ToolChoice = Union[Literal["auto", "required", "off"], Dict[Literal["specific"], str]]
