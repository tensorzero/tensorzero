# type: ignore
"""
Tests for multi-turn parallel tool use with AWS Bedrock using the OpenAI Python SDK.

This test verifies that:
1. The model can make parallel tool calls (get_temperature + get_humidity)
2. Tool results can be sent back in a follow-up message
3. The model responds correctly with the tool results
"""

import json
import tempfile

import pytest
import tensorzero
from openai import AsyncOpenAI

GET_TEMPERATURE_PARAMS = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": 'The location to get the temperature for (e.g. "New York")',
        },
        "units": {
            "type": "string",
            "description": 'The units to get the temperature in (must be "fahrenheit" or "celsius")',
            "enum": ["fahrenheit", "celsius"],
        },
    },
    "required": ["location"],
    "additionalProperties": False,
}

GET_HUMIDITY_PARAMS = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": 'The location to get the humidity for (e.g. "New York")',
        }
    },
    "required": ["location"],
    "additionalProperties": False,
}

SYSTEM_TEMPLATE = """You are a helpful and friendly assistant named Dr. Mehta.
People will ask you questions about the weather.
If asked about the weather, just respond with two tool calls. Use BOTH the "get_temperature" and "get_humidity" tools.
If provided with a tool result, use it to respond to the user (e.g. "The weather in New York is 55 degrees Fahrenheit with 50% humidity.").
"""


async def setup_openai_sdk():
    # Create temp files for tool parameters
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_params_file:
        json.dump(GET_TEMPERATURE_PARAMS, temp_params_file)
        temp_params_path = temp_params_file.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as humidity_params_file:
        json.dump(GET_HUMIDITY_PARAMS, humidity_params_file)
        humidity_params_path = humidity_params_file.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".minijinja", delete=False) as system_template_file:
        system_template_file.write(SYSTEM_TEMPLATE)
        system_template_path = system_template_file.name

    # Create config
    config = f"""
[functions.multi_turn_parallel_tool_test]
type = "chat"
tools = ["get_temperature", "get_humidity"]
tool_choice = "auto"
parallel_tool_calls = true

[functions.multi_turn_parallel_tool_test.variants.aws_bedrock]
type = "chat_completion"
model = "claude-haiku-4-5-aws-bedrock"
system_template = "{system_template_path}"

[tools.get_temperature]
description = "Get the current temperature in a given location"
parameters = "{temp_params_path}"

[tools.get_humidity]
description = "Get the current humidity in a given location"
parameters = "{humidity_params_path}"

[models.claude-haiku-4-5-aws-bedrock]
routing = ["aws_bedrock"]

[models.claude-haiku-4-5-aws-bedrock.providers.aws_bedrock]
type = "aws_bedrock"
model_id = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
region = "us-east-1"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as config_file:
        config_file.write(config)
        config_path = config_file.name

    # Patch OpenAI client with embedded gateway
    return await tensorzero.patch_openai_client(
        AsyncOpenAI(),
        config_file=config_path,
    )


@pytest.mark.asyncio
async def test_multi_turn_parallel_tool_use_aws_bedrock():
    oai = await setup_openai_sdk()

    messages = [
        {
            "role": "user",
            "content": "What is the weather like in Tokyo (in Fahrenheit)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions.",
        }
    ]

    try:
        # First turn: trigger parallel tool calls
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
