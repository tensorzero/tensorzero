# type: ignore
"""
Tests for multi-turn parallel tool use with AWS Bedrock using the OpenAI Python SDK.

This test verifies that:
1. The model can make parallel tool calls (get_temperature + get_humidity)
2. Tool results can be sent back in a follow-up message
3. The model responds correctly with the tool results
"""

import pytest
import tensorzero
from openai import AsyncOpenAI

from .shared import USER_MESSAGE, create_config_file


@pytest.mark.asyncio
async def test_multi_turn_parallel_tool_use_aws_bedrock():
    # Set up the SDK
    config_path = create_config_file()
    oai = await tensorzero.patch_openai_client(
        AsyncOpenAI(),
        config_file=config_path,
    )

    try:
        # First turn: trigger parallel tool calls
        messages = [{"role": "user", "content": USER_MESSAGE}]

        first_response = await oai.chat.completions.create(
            model="tensorzero::function_name::multi_turn_parallel_tool_test",
            messages=messages,
            stream=False,
        )

        # Verify we got tool calls
        message = first_response.choices[0].message
        assert message.tool_calls is not None, "Expected tool calls in response"
        assert len(message.tool_calls) == 2, f"Expected exactly 2 tool calls, got {len(message.tool_calls)}"

        messages.append(message)

        # Build tool results for the second turn
        for tool_call in message.tool_calls:
            if tool_call.function.name == "get_temperature":
                result = "70"
            elif tool_call.function.name == "get_humidity":
                result = "30"
            else:
                raise AssertionError(f"Unknown tool: {tool_call.function.name}")

            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

        # Second turn: send tool results
        second_response = await oai.chat.completions.create(
            model="tensorzero::function_name::multi_turn_parallel_tool_test",
            messages=messages,
            stream=False,
        )

        # Verify the response contains the expected values
        second_message = second_response.choices[0].message
        assert second_message.content is not None, "Expected text content in second response"
        assert "70" in second_message.content and "30" in second_message.content, (
            f"Expected response to contain '70' and '30', got: {second_message.content}"
        )

    finally:
        tensorzero.close_patched_openai_client_gateway(oai)
