"""
Shared constants and utilities for multi-turn parallel tool use tests with AWS Bedrock.
"""

import json
import tempfile

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

USER_MESSAGE = "What is the weather like in Tokyo (in Fahrenheit)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."


def create_config_file() -> str:
    """Create temp files and config, return the config file path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_params_file:
        json.dump(GET_TEMPERATURE_PARAMS, temp_params_file)
        temp_params_path = temp_params_file.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as humidity_params_file:
        json.dump(GET_HUMIDITY_PARAMS, humidity_params_file)
        humidity_params_path = humidity_params_file.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".minijinja", delete=False) as system_template_file:
        system_template_file.write(SYSTEM_TEMPLATE)
        system_template_path = system_template_file.name

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
        return config_file.name
