import json
import typing as t
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from json import JSONEncoder
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union
from uuid import UUID

import httpx


# Helper used to serialize Python objects to JSON, which may contain dataclasses like `Text`
# Used by the Rust native module
class ToDictEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        return o.to_dict()


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class ContentBlock(ABC):
    type: str

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass


@dataclass
class Text(ContentBlock):
    text: Optional[str] = None
    arguments: Optional[Any] = None

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
class RawText:
    # This class does not subclass ContentBlock since it cannot be output by the API.
    value: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(type="raw_text", value=self.value)


@dataclass
class ImageBase64:
    # This class does not subclass ContentBlock since it cannot be output by the API.
    data: str
    mime_type: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(type="image", data=self.data, mime_type=self.mime_type)


@dataclass
class ImageUrl:
    # This class does not subclass ContentBlock since it cannot be output by the API.
    url: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(type="image", url=self.url)


@dataclass
class ToolCall(ContentBlock):
    arguments: Optional[Dict[str, Any]]
    id: str
    name: Optional[str]
    raw_arguments: Dict[str, Any]
    raw_name: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            type="tool_call",
            arguments=json.dumps(self.raw_arguments),
            id=self.id,
            name=self.raw_name,
        )


@dataclass
class Thought(ContentBlock):
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(type="thought", value=self.text)


@dataclass
class ToolResult:
    # This class does not subclass ContentBlock since it cannot be output by the API.
    name: str
    result: str
    id: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(type="tool_result", name=self.name, result=self.result, id=self.id)


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALL = "tool_call"
    CONTENT_FILTER = "content_filter"
    UNKNOWN = "unknown"


@dataclass
class JsonInferenceOutput:
    raw: str
    parsed: Optional[Dict[str, Any]]


@dataclass
class ChatInferenceResponse:
    inference_id: UUID
    episode_id: UUID
    variant_name: str
    content: List[ContentBlock]
    usage: Usage
    finish_reason: Optional[FinishReason]


@dataclass
class JsonInferenceResponse:
    inference_id: UUID
    episode_id: UUID
    variant_name: str
    output: JsonInferenceOutput
    usage: Usage
    finish_reason: Optional[FinishReason]


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: Any


class InferenceInput(TypedDict):
    messages: Union[List[Message], Dict[str, Any]]
    system: Optional[Union[str, Dict[str, Any]]]


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
        )
    elif "output" in data and isinstance(data["output"], dict):
        output: Dict[str, Any] = data["output"]
        finish_reason = data.get("finish_reason")
        finish_reason_enum = FinishReason(finish_reason) if finish_reason else None

        return JsonInferenceResponse(
            inference_id=UUID(data["inference_id"]),
            episode_id=UUID(data["episode_id"]),
            variant_name=data["variant_name"],
            output=JsonInferenceOutput(**output),
            usage=Usage(**data["usage"]),
            finish_reason=finish_reason_enum,
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
    else:
        raise ValueError(f"Unknown content block type: {block}")


# Types for streaming inference responses


@dataclass
class ContentBlockChunk:
    type: str


@dataclass
class TextChunk(ContentBlockChunk):
    # In the possibility that multiple text messages are sent in a single streaming response,
    # this `id` will be used to disambiguate them
    id: str
    text: str


@dataclass
class ToolCallChunk(ContentBlockChunk):
    # This is the tool call ID that many LLM APIs use to associate tool calls with tool responses
    id: str
    # `raw_arguments` will come as partial JSON
    raw_arguments: str
    raw_name: str


@dataclass
class ThoughtChunk(ContentBlockChunk):
    text: str
    id: str


@dataclass
class ChatChunk:
    inference_id: UUID
    episode_id: UUID
    variant_name: str
    content: List[ContentBlockChunk]
    usage: Optional[Usage]
    finish_reason: Optional[FinishReason] = None


@dataclass
class JsonChunk:
    inference_id: UUID
    episode_id: UUID
    variant_name: str
    raw: str
    usage: Optional[Usage]
    finish_reason: Optional[FinishReason] = None


InferenceChunk = Union[ChatChunk, JsonChunk]


class VariantExtraBody(TypedDict):
    variant_name: str
    pointer: str
    value: t.Any


class ProviderExtraBody(TypedDict):
    model_provider_name: str
    pointer: str
    value: t.Any


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
        return TextChunk(id=block["id"], text=block["text"], type=block_type)
    elif block_type == "tool_call":
        return ToolCallChunk(
            id=block["id"],
            raw_arguments=block["raw_arguments"],
            raw_name=block["raw_name"],
            type=block_type,
        )
    elif block_type == "thought":
        return ThoughtChunk(id=block["id"], text=block["text"], type=block_type)
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
        return f"TensorZeroInternalError: {self.msg}"


class TensorZeroError(BaseTensorZeroError):
    def __init__(self, status_code: int, text: t.Optional[str]):
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
