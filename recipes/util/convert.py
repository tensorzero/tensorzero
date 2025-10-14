import warnings
from typing import Any, Dict, List, Optional

from tensorzero import ContentBlock, RenderedSample, Text, Thought, ToolCall, ToolResult
from tensorzero.internal import OutputMessage


def warning_message(role: str) -> str:
    return (
        f"Provider may not support multiple content blocks per message. "
        f"We have chosen to concatenate the text across all content blocks for the message with role '{role}'. "
        f"You may want to manually review this behavior."
    )


def tensorzero_to_openai_tools(tools: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """Convert TensorZero tools to OpenAI format."""
    chatml_tools: List[Dict[str, Any]] = []
    if tools:
        for tool in tools:
            chatml_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
            )
    return chatml_tools


def tensorzero_message_to_openai(
    message: OutputMessage,
    join_text_blocks: bool = True,
) -> Optional[List[Dict[str, Any]]]:
    chatml_messages: List[Dict[str, Any]] = []
    assert message.role in ["user", "assistant"], f"Invalid role: {message.role}"
    content: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for content_block in message.content:
        if isinstance(content_block, Text):
            assert content_block.arguments is None, "Arguments should be None"
            if content_block.text:
                content.append(content_block.text)
        elif isinstance(content_block, Thought):
            content.append(f"<think>{content_block.text}</think>")
        elif isinstance(content_block, ToolCall):
            tool_calls.append(
                {
                    "function": {
                        "arguments": content_block.raw_arguments,
                        "name": content_block.name,
                    },
                    "id": content_block.id,
                    "type": "function",
                }
            )
        elif isinstance(content_block, ToolResult):
            # Tool results get priority so that they follow the tool call in the conversation.
            # Any other "user" content will be appended in another message below.
            chatml_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": content_block.id,
                    "content": content_block.result,
                }
            )
        else:
            warnings.warn(
                f"We do not support content block type: {type(content_block)}, dropping example.",
                UserWarning,
            )
            return None
    if content or tool_calls:
        chatml_message: Dict[str, Any] = {"role": message.role}
        if content:
            if join_text_blocks:
                warnings.warn(warning_message(message.role), UserWarning)
                chatml_message["content"] = "\n".join(content)
            else:
                chatml_message["content"] = [{"type": "text", "text": c} for c in content]
        if tool_calls:
            chatml_message["tool_calls"] = tool_calls
        chatml_messages.append(chatml_message)

    return chatml_messages


def tensorzero_output_to_openai(output: List[ContentBlock], join_text_blocks: bool = True) -> Optional[Dict[str, Any]]:
    content: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for content_block in output:
        if isinstance(content_block, Text):
            assert content_block.arguments is None, "Arguments should be None"
            if content_block.text:
                content.append(content_block.text)
        elif isinstance(content_block, Thought):
            content.append(f"<think>{content_block.text}</think>")
        elif isinstance(content_block, ToolCall):
            tool_calls.append(
                {
                    "function": {
                        "arguments": content_block.raw_arguments,
                        "name": content_block.name,
                    },
                    "id": content_block.id,
                    "type": "function",
                }
            )
        else:
            warnings.warn(
                f"We do not support content block type: {type(content_block)}, dropping example.",
                UserWarning,
            )
            return None

    # Once we finish collecting all blocks, create one assistant message.
    output_message: Dict[str, Any] = {"role": "assistant"}
    if content:
        if join_text_blocks:
            warnings.warn(warning_message("assistant"), UserWarning)
            output_message["content"] = "\n".join(content)
        else:
            output_message["content"] = [{"type": "text", "text": c} for c in content]
    if tool_calls:
        output_message["tool_calls"] = tool_calls

    return output_message


def tensorzero_rendered_samples_to_conversations(
    rendered_inferences: List[RenderedSample],
    conversation_key: str = "conversation",
    join_text_blocks: bool = True,
) -> List[Dict[str, Any]]:
    conversations: List[Dict[str, Any]] = []
    for rendered_inference in rendered_inferences:
        messages: List[Dict[str, Any]] = []
        model_output = rendered_inference.output
        if model_output is None:
            warnings.warn(
                "Model output is not defined, dropping example.",
                UserWarning,
            )
            continue
        output_message = tensorzero_output_to_openai(model_output, join_text_blocks=join_text_blocks)
        if output_message is None:
            continue
        model_input = rendered_inference.input
        if model_input.system is not None:
            messages.append({"role": "system", "content": model_input.system})
        for message in model_input.messages:
            chatml_message = tensorzero_message_to_openai(message, join_text_blocks=join_text_blocks)
            if chatml_message:
                messages.extend(chatml_message)
        messages.append(output_message)
        # Add tools if available
        payload = {
            conversation_key: messages,
        }
        if rendered_inference.tool_params:
            tools = tensorzero_to_openai_tools(rendered_inference.tool_params.tools_available)
            payload["tools"] = tools
        conversations.append(payload)

    return conversations
