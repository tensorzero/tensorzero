from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

# Types for non-streaming inference responses


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class Text:
    text: str


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    id: str
    parsed_name: Optional[str]
    parsed_arguments: Optional[Dict[str, Any]]


ContentBlock = Union[Text, ToolCall]


@dataclass
class JsonInferenceOutput:
    raw: str
    parsed: Optional[Dict[str, Any]]


@dataclass
class ChatInferenceResponse:
    inference_id: UUID
    episode_id: UUID
    variant_name: str
    output: List[ContentBlock]
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
    if "output" in data and isinstance(data["output"], list):
        return ChatInferenceResponse(
            inference_id=UUID(data["inference_id"]),
            episode_id=UUID(data["episode_id"]),
            variant_name=data["variant_name"],
            output=[parse_content_block(block) for block in data["output"]],
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
        return Text(text=block["text"])
    elif block_type == "tool_call":
        return ToolCall(
            name=block["name"],
            arguments=block["arguments"],
            id=block["id"],
            parsed_name=block.get("parsed_name"),
            parsed_arguments=block.get("parsed_arguments"),
        )
    else:
        raise ValueError(f"Unknown content block type: {block}")


# Types for streaming inference responses


@dataclass
class TextChunk:
    # In the possibility that multiple text messages are sent in a single streaming response,
    # this `id` will be used to disambiguate them
    id: str
    text: str


@dataclass
class ToolCallChunk:
    name: str
    # This is the tool call ID that many LLM APIs use to associate tool calls with tool responses
    id: str
    # `arguments` will come as partial JSON
    arguments: str


ContentBlockChunk = Union[TextChunk, ToolCallChunk]


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
        return TextChunk(id=block["id"], text=block["text"])
    elif block_type == "tool_call":
        return ToolCallChunk(
            name=block["name"], id=block["id"], arguments=block["arguments"]
        )
    else:
        raise ValueError(f"Unknown content block type: {block}")


# Types for feedback
@dataclass
class FeedbackResponse:
    feedback_id: UUID
