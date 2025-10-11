# TODO - remove this 'type: ignore'
# type: ignore
import json
from uuid import UUID

import pytest
import tensorzero
from openai import AsyncOpenAI
from tensorzero.util import uuid7


@pytest.mark.asyncio
async def test_dynamic_json_mode_inference_openai():
    async_client = AsyncOpenAI(api_key="donotuse")
    # Patch the client
    async_client = await tensorzero.patch_openai_client(
        async_client,
        clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero_e2e_tests",
        config_file="../../tensorzero-core/tests/e2e/tensorzero.toml",
        async_setup=True,
    )

    episode_id = str(uuid7())
    output_schema = {
        "type": "object",
        "properties": {"response": {"type": "string"}},
        "required": ["response"],
        "additionalProperties": False,
    }
    serialized_output_schema = json.dumps(output_schema)
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "test",
            "description": "test",
            "schema": output_schema,
        },
    }
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "assistant_name": "Dr. Mehta",
                    "schema": serialized_output_schema,
                }
            ],
        },
        {"role": "user", "content": [{"country": "Japan"}]},
    ]
    result = await async_client.chat.completions.create(
        extra_body={
            "tensorzero::episode_id": episode_id,
            "tensorzero::variant_name": "openai",
        },
        messages=messages,
        model="tensorzero::function_name::dynamic_json",
        response_format=response_format,
    )
    assert result.model == "tensorzero::function_name::dynamic_json::variant_name::openai"
    assert result.episode_id == episode_id
    json_content = json.loads(result.choices[0].message.content)
    assert "tokyo" in json_content["response"].lower()
    assert result.choices[0].message.tool_calls is None
    assert result.usage.prompt_tokens > 50
    assert result.usage.completion_tokens > 0


@pytest.mark.asyncio
async def test_patch_openai_client_with_async_client_async_setup_true():
    """Tests that tensorzero.patch_openai_client works with AsyncOpenAI client."""
    client = AsyncOpenAI(api_key="donotuse")

    # Patch the client
    patched_client = await tensorzero.patch_openai_client(
        client,
        clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero_e2e_tests",
        config_file="../../examples/quickstart/config/tensorzero.toml",
        async_setup=True,
    )

    messages = [
        {"role": "user", "content": "What is the capital of Japan?"},
    ]

    result = await patched_client.chat.completions.create(
        messages=messages,
        model="tensorzero::function_name::generate_haiku",
        temperature=0.4,
        extra_body={"tensorzero::episode_id": str(uuid7())},
    )

    # Verify IDs are valid UUIDs
    UUID(result.id)  # Will raise ValueError if invalid
    UUID(result.episode_id)  # Will raise ValueError if invalid
    assert "Tokyo" in result.choices[0].message.content
    assert result.usage.prompt_tokens > 0
    assert result.usage.completion_tokens > 0
    assert result.usage.total_tokens > 0
    assert result.choices[0].finish_reason == "stop"
    assert result.model == "tensorzero::function_name::generate_haiku::variant_name::gpt_4o_mini"

    tensorzero.close_patched_openai_client_gateway(patched_client)


@pytest.mark.asyncio
async def test_patch_openai_client_with_async_client_async_setup_false():
    """Tests that tensorzero.patch_openai_client works with AsyncOpenAI client using sync setup."""
    client = AsyncOpenAI(api_key="donotuse")

    # Patch the client with sync setup
    patched_client = tensorzero.patch_openai_client(
        client,
        clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero_e2e_tests",
        config_file="../../examples/quickstart/config/tensorzero.toml",
        async_setup=False,
    )

    messages = [
        {"role": "user", "content": "What is the capital of Japan?"},
    ]

    result = await patched_client.chat.completions.create(
        messages=messages,
        model="tensorzero::function_name::generate_haiku",
        temperature=0.4,
        extra_body={"tensorzero::episode_id": str(uuid7())},
    )

    # Verify IDs are valid UUIDs
    UUID(result.id)  # Will raise ValueError if invalid
    UUID(result.episode_id)  # Will raise ValueError if invalid
    assert "Tokyo" in result.choices[0].message.content
    assert result.usage.prompt_tokens > 0
    assert result.usage.completion_tokens > 0
    assert result.usage.total_tokens > 0
    assert result.choices[0].finish_reason == "stop"
    assert result.model == "tensorzero::function_name::generate_haiku::variant_name::gpt_4o_mini"

    tensorzero.close_patched_openai_client_gateway(patched_client)
