# type: ignore
"""
Tests for multi-turn parallel tool use with AWS Bedrock using the TensorZero Python SDK.

This test verifies that:
1. The model can make parallel tool calls (get_temperature + get_humidity)
2. Tool results can be sent back in a follow-up message
3. The model responds correctly with the tool results
"""

import typing as t

import pytest
from tensorzero import AsyncTensorZeroGateway, ChatInferenceResponse, Text, ToolCall

from .shared import USER_MESSAGE, create_config_file


@pytest.mark.asyncio
async def test_multi_turn_parallel_tool_use_aws_bedrock():
    # Set up the SDK
    config_path = create_config_file()
    t0 = await AsyncTensorZeroGateway.build_embedded(config_file=config_path)

    # First turn: trigger parallel tool calls
    messages: t.List[t.Dict[str, t.Any]] = [{"role": "user", "content": [{"type": "text", "text": USER_MESSAGE}]}]

    first_response = await t0.inference(
        function_name="multi_turn_parallel_tool_test",
        input={"messages": messages},
    )

    assert isinstance(first_response, ChatInferenceResponse)
    assert len(first_response.content) == 2, f"Expected exactly 2 tool calls, got {len(first_response.content)}"

    # Add assistant message with tool calls
    messages.append({"role": "assistant", "content": first_response.content})

    # Build tool results for the second turn
    tool_results: t.List[t.Dict[str, t.Any]] = []
    for content_block in first_response.content:
        assert isinstance(content_block, ToolCall), f"Expected ToolCall, got {type(content_block)}"
        if content_block.name == "get_temperature":
            result = "70"
        elif content_block.name == "get_humidity":
            result = "30"
        else:
            raise AssertionError(f"Unknown tool: {content_block.name}")

        tool_results.append(
            {
                "type": "tool_result",
                "id": content_block.id,
                "name": content_block.name,
                "result": result,
            }
        )

    messages.append({"role": "user", "content": tool_results})

    # Second turn: send tool results
    second_response = await t0.inference(
        function_name="multi_turn_parallel_tool_test",
        input={"messages": messages},
    )

    assert isinstance(second_response, ChatInferenceResponse)
    assert len(second_response.content) > 0, "Expected content in second response"
    assert isinstance(second_response.content[0], Text), "Expected text content in second response"

    text = second_response.content[0].text
    assert "70" in text and "30" in text, f"Expected response to contain '70' and '30', got: {text}"
