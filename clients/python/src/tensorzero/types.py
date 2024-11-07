import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import httpx

# Types for non-streaming inference responses


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
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(type="text", value=self.text)


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
class ToolResult:
    name: str
    result: str
    id: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(type="tool_result", name=self.name, result=self.result, id=self.id)


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


@dataclass
class JsonInferenceResponse:
    inference_id: UUID
    episode_id: UUID
    variant_name: str
    output: JsonInferenceOutput
    usage: Usage


InferenceResponse = Union[ChatInferenceResponse, JsonInferenceResponse]


def parse_inference_response(data: Dict[str, Any]) -> InferenceResponse:
    if "content" in data and isinstance(data["content"], list):
        return ChatInferenceResponse(
            inference_id=UUID(data["inference_id"]),
            episode_id=UUID(data["episode_id"]),
            variant_name=data["variant_name"],
            content=[parse_content_block(block) for block in data["content"]],
            usage=Usage(**data["usage"]),
        )
    elif "output" in data and isinstance(data["output"], dict):
        return JsonInferenceResponse(
            inference_id=UUID(data["inference_id"]),
            episode_id=UUID(data["episode_id"]),
            variant_name=data["variant_name"],
            output=JsonInferenceOutput(**data["output"]),
            usage=Usage(**data["usage"]),
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
class ChatChunk:
    inference_id: UUID
    episode_id: UUID
    variant_name: str
    content: List[ContentBlockChunk]
    usage: Optional[Usage]


@dataclass
class JsonChunk:
    inference_id: UUID
    episode_id: UUID
    variant_name: str
    raw: str
    usage: Optional[Usage]


InferenceChunk = Union[ChatChunk, JsonChunk]


def parse_inference_chunk(chunk: Dict[str, Any]) -> InferenceChunk:
    if "content" in chunk:
        return ChatChunk(
            inference_id=UUID(chunk["inference_id"]),
            episode_id=UUID(chunk["episode_id"]),
            variant_name=chunk["variant_name"],
            content=[parse_content_block_chunk(block) for block in chunk["content"]],
            usage=Usage(**chunk["usage"]) if "usage" in chunk else None,
        )
    elif "raw" in chunk:
        return JsonChunk(
            inference_id=UUID(chunk["inference_id"]),
            episode_id=UUID(chunk["episode_id"]),
            variant_name=chunk["variant_name"],
            raw=chunk["raw"],
            usage=Usage(**chunk["usage"]) if "usage" in chunk else None,
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
    else:
        raise ValueError(f"Unknown content block type: {block}")


# Types for feedback
@dataclass
class FeedbackResponse:
    feedback_id: UUID


# Custom TensorZero error type
class TensorZeroError(Exception):
    def __init__(self, response: httpx.Response):
        self.response = response
        self.status_code = response.status_code
        try:
            self.text = response.text
        except Exception:
            self.text = "(response body unavailable)"

    def __str__(self) -> str:
        return f"TensorZeroError (status code {self.status_code}): {self.text}"
