# type: ignore
"""
Tests for the OpenAI compatibility interface using the OpenAI Python client

We use pytest to run the tests.

These tests should cover the major functionality of the translation
layer between the OpenAI interface and TensorZero. They do not
attempt to comprehensively cover all of TensorZero's functionality.
See the tests across the Rust codebase for more comprehensive tests.

To run:
```
pytest
```
or
```
uv run pytest
```
"""

import json
from time import time
from uuid import UUID

import pytest
import pytest_asyncio
from openai import AsyncOpenAI
from tensorzero.util import uuid7


@pytest_asyncio.fixture
async def async_client():
    async with AsyncOpenAI(
        api_key="donotuse", base_url="http://localhost:3000/openai/v1"
    ) as client:
        yield client


@pytest.mark.asyncio
async def test_async_basic_inference(async_client):
    messages = [
        {"role": "system", "content": [{"assistant_name": "Alfred Pennyworth"}]},
        {"role": "user", "content": "Hello"},
    ]

    result = await async_client.chat.completions.create(
        extra_headers={"episode_id": str(uuid7())},
        messages=messages,
        model="tensorzero::basic_test",
        temperature=0.4,
    )
    # Verify IDs are valid UUIDs
    UUID(result.id)  # Will raise ValueError if invalid
    UUID(result.episode_id)  # Will raise ValueError if invalid
    assert (
        result.choices[0].message.content
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    usage = result.usage
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 10
    assert usage.total_tokens == 20


@pytest.mark.asyncio
async def test_async_inference_streaming(async_client):
    start_time = time()
    messages = [
        {"role": "system", "content": [{"assistant_name": "Alfred Pennyworth"}]},
        {"role": "user", "content": "Hello"},
    ]
    stream = await async_client.chat.completions.create(
        extra_headers={"episode_id": str(uuid7())},
        messages=messages,
        model="tensorzero::basic_test",
        stream=True,
        max_tokens=300,
        seed=69,
    )
    first_chunk_duration = None
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
        if first_chunk_duration is None:
            first_chunk_duration = time() - start_time
    last_chunk_duration = time() - start_time - first_chunk_duration
    assert last_chunk_duration > first_chunk_duration + 0.1
    expected_text = [
        "Wally,",
        " the",
        " golden",
        " retriever,",
        " wagged",
        " his",
        " tail",
        " excitedly",
        " as",
        " he",
        " devoured",
        " a",
        " slice",
        " of",
        " cheese",
        " pizza.",
    ]
    previous_inference_id = None
    previous_episode_id = None
    for i, chunk in enumerate(chunks):
        if previous_inference_id is not None:
            assert chunk.id == previous_inference_id
        if previous_episode_id is not None:
            assert chunk.episode_id == previous_episode_id
        previous_inference_id = chunk.id
        previous_episode_id = chunk.episode_id
        variant_name = chunk.model
        assert variant_name == "test"
        if i + 1 < len(chunks):
            assert len(chunk.choices) == 1
            assert chunk.choices[0].delta.content == expected_text[i]
        else:
            assert chunk.choices[0].delta.content is None
            assert chunk.usage.prompt_tokens == 10
            assert chunk.usage.completion_tokens == 16
            assert chunk.usage.total_tokens == 26


@pytest.mark.asyncio
async def test_async_inference_streaming_nonexistent_function(async_client):
    with pytest.raises(Exception) as exc_info:
        messages = [
            {"role": "system", "content": [{"assistant_name": "Alfred Pennyworth"}]},
            {"role": "user", "content": "Hello"},
        ]

        await async_client.chat.completions.create(
            extra_headers={
                "episode_id": str(uuid7()),
            },
            messages=messages,
            model="tensorzero::does_not_exist",
        )
    assert exc_info.value.status_code == 404
    assert (
        str(exc_info.value)
        == "Error code: 404 - {'error': 'Unknown function: does_not_exist'}"
    )


@pytest.mark.asyncio
async def test_async_inference_streaming_missing_function(async_client):
    with pytest.raises(Exception) as exc_info:
        messages = [
            {"role": "system", "content": [{"assistant_name": "Alfred Pennyworth"}]},
            {"role": "user", "content": "Hello"},
        ]

        await async_client.chat.completions.create(
            extra_headers={
                "episode_id": str(uuid7()),
            },
            messages=messages,
            model="tensorzero::",
        )
    assert exc_info.value.status_code == 400
    assert (
        str(exc_info.value)
        == "Error code: 400 - {'error': 'Invalid request to OpenAI-compatible endpoint: function_name (passed in model field after \"tensorzero::\") cannot be empty'}"
    )


@pytest.mark.asyncio
async def test_async_inference_streaming_malformed_function(async_client):
    with pytest.raises(Exception) as exc_info:
        messages = [
            {"role": "system", "content": [{"assistant_name": "Alfred Pennyworth"}]},
            {"role": "user", "content": "Hello"},
        ]

        await async_client.chat.completions.create(
            extra_headers={
                "episode_id": str(uuid7()),
            },
            messages=messages,
            model="chatgpt",
        )
    assert exc_info.value.status_code == 400
    assert (
        str(exc_info.value)
        == "Error code: 400 - {'error': \"Invalid request to OpenAI-compatible endpoint: model name must start with 'tensorzero::'\"}"
    )


@pytest.mark.asyncio
async def test_async_inference_streaming_missing_model(async_client):
    with pytest.raises(Exception) as exc_info:
        messages = [
            {"role": "system", "content": [{"assistant_name": "Alfred Pennyworth"}]},
            {"role": "user", "content": "Hello"},
        ]

        await async_client.chat.completions.create(
            messages=messages,
        )
    assert (
        str(exc_info.value)
        == "Missing required arguments; Expected either ('messages' and 'model') or ('messages', 'model' and 'stream') arguments to be given"
    )


@pytest.mark.asyncio
async def test_async_inference_streaming_malformed_input(async_client):
    with pytest.raises(Exception) as exc_info:
        messages = [
            {"role": "system", "content": [{"name_of_assistant": "Alfred Pennyworth"}]},
            {"role": "user", "content": "Hello"},
        ]
        await async_client.chat.completions.create(
            extra_headers={"episode_id": str(uuid7())},
            messages=messages,
            model="tensorzero::basic_test",
            stream=True,
        )
    assert exc_info.value.status_code == 400
    assert (
        str(exc_info.value)
        == """Error code: 400 - {'error': 'JSON Schema validation failed for Function:\\n\\n"assistant_name" is a required property\\nData: {"name_of_assistant":"Alfred Pennyworth"}Schema: {"type":"object","properties":{"assistant_name":{"type":"string"}},"required":["assistant_name"]}'}"""
    )


@pytest.mark.asyncio
async def test_async_tool_call_inference(async_client):
    messages = [
        {"role": "system", "content": [{"assistant_name": "Alfred Pennyworth"}]},
        {
            "role": "user",
            "content": "Hi I'm visiting Brooklyn from Brazil. What's the weather?",
        },
    ]
    result = await async_client.chat.completions.create(
        extra_headers={"episode_id": str(uuid7())},
        messages=messages,
        model="tensorzero::weather_helper",
        top_p=0.5,
    )
    assert result.model == "variant"
    assert result.choices[0].message.content is None
    assert result.choices[0].message.tool_calls is not None
    tool_calls = result.choices[0].message.tool_calls
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call.type == "function"
    assert tool_call.function.name == "get_temperature"
    assert tool_call.function.arguments == '{"location":"Brooklyn","units":"celsius"}'
    usage = result.usage
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 10


@pytest.mark.asyncio
async def test_async_malformed_tool_call_inference(async_client):
    messages = [
        {"role": "system", "content": [{"assistant_name": "Alfred Pennyworth"}]},
        {
            "role": "user",
            "content": "Hi I'm visiting Brooklyn from Brazil. What's the weather?",
        },
    ]
    result = await async_client.chat.completions.create(
        extra_headers={
            "episode_id": str(uuid7()),
            "variant_name": "bad_tool",
        },
        messages=messages,
        model="tensorzero::weather_helper",
        presence_penalty=0.5,
    )
    assert result.model == "bad_tool"
    assert result.choices[0].message.content is None
    assert result.choices[0].message.tool_calls is not None
    tool_calls = result.choices[0].message.tool_calls
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call.type == "function"
    assert tool_call.function.name == "get_temperature"
    assert tool_call.function.arguments == '{"location":"Brooklyn","units":"Celsius"}'
    usage = result.usage
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 10


@pytest.mark.asyncio
async def test_async_tool_call_streaming(async_client):
    messages = [
        {"role": "system", "content": [{"assistant_name": "Alfred Pennyworth"}]},
        {
            "role": "user",
            "content": "Hi I'm visiting Brooklyn from Brazil. What's the weather?",
        },
    ]
    stream = await async_client.chat.completions.create(
        extra_headers={"episode_id": str(uuid7())},
        messages=messages,
        model="tensorzero::weather_helper",
        stream=True,
    )
    chunks = [chunk async for chunk in stream]
    expected_text = [
        '{"location"',
        ':"Brooklyn"',
        ',"units"',
        ':"celsius',
        '"}',
    ]
    previous_inference_id = None
    previous_episode_id = None
    for i, chunk in enumerate(chunks):
        if previous_inference_id is not None:
            assert chunk.id == previous_inference_id
        if previous_episode_id is not None:
            assert chunk.episode_id == previous_episode_id
        previous_inference_id = chunk.id
        previous_episode_id = chunk.episode_id
        variant_name = chunk.model
        assert variant_name == "variant"
        if i + 1 < len(chunks):
            assert len(chunk.choices) == 1
            assert chunk.choices[0].delta.content is None
            assert len(chunk.choices[0].delta.tool_calls) == 1
            tool_call = chunk.choices[0].delta.tool_calls[0]
            assert tool_call.type == "function"
            assert tool_call.function.name == "get_temperature"
            assert tool_call.function.arguments == expected_text[i]
        else:
            assert chunk.choices[0].delta.content is None
            assert len(chunk.choices[0].delta.tool_calls) == 0
            assert chunk.usage.prompt_tokens == 10
            assert chunk.usage.completion_tokens == 5


@pytest.mark.asyncio
async def test_async_json_streaming(async_client):
    # We don't actually have a streaming JSON function implemented in `dummy.rs` but it doesn't matter for this test since
    # TensorZero doesn't parse the JSON output of the function for streaming calls.
    messages = [
        {"role": "system", "content": [{"assistant_name": "Alfred Pennyworth"}]},
        {"role": "user", "content": [{"country": "Japan"}]},
    ]
    stream = await async_client.chat.completions.create(
        extra_headers={"episode_id": str(uuid7())},
        messages=messages,
        model="tensorzero::json_success",
        stream=True,
    )
    chunks = [chunk async for chunk in stream]
    expected_text = [
        "Wally,",
        " the",
        " golden",
        " retriever,",
        " wagged",
        " his",
        " tail",
        " excitedly",
        " as",
        " he",
        " devoured",
        " a",
        " slice",
        " of",
        " cheese",
        " pizza.",
    ]
    previous_inference_id = None
    previous_episode_id = None
    for i, chunk in enumerate(chunks):
        if previous_inference_id is not None:
            assert chunk.id == previous_inference_id
        if previous_episode_id is not None:
            assert chunk.episode_id == previous_episode_id
        previous_inference_id = chunk.id
        previous_episode_id = chunk.episode_id
        variant_name = chunk.model
        assert variant_name == "test"
        if i + 1 < len(chunks):
            assert chunk.choices[0].delta.content == expected_text[i]
        else:
            assert len(chunk.choices[0].delta.content) == 0
            assert chunk.usage.prompt_tokens == 10
            assert chunk.usage.completion_tokens == 16


@pytest.mark.asyncio
async def test_async_json_success(async_client):
    messages = [
        {"role": "system", "content": [{"assistant_name": "Alfred Pennyworth"}]},
        {"role": "user", "content": [{"country": "Japan"}]},
    ]
    episode_id = str(uuid7())
    result = await async_client.chat.completions.create(
        extra_headers={"episode_id": episode_id},
        messages=messages,
        model="tensorzero::json_success",
    )
    assert result.model == "test"
    assert result.episode_id == episode_id
    assert result.choices[0].message.content == '{"answer":"Hello"}'
    assert result.choices[0].message.tool_calls is None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 10


@pytest.mark.asyncio
async def test_async_json_failure(async_client):
    messages = [
        {"role": "system", "content": [{"assistant_name": "Alfred Pennyworth"}]},
        {"role": "user", "content": "Hello, world!"},
    ]
    result = await async_client.chat.completions.create(
        extra_headers={"episode_id": str(uuid7())},
        messages=messages,
        model="tensorzero::json_fail",
    )
    assert result.model == "test"
    assert (
        result.choices[0].message.content
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    assert result.choices[0].message.tool_calls is None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 10


@pytest.mark.asyncio
async def test_dynamic_tool_use_inference_openai(async_client):
    episode_id = str(uuid7())
    messages = [
        {"role": "system", "content": [{"assistant_name": "Dr. Mehta"}]},
        {
            "role": "user",
            "content": "What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function.",
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_temperature",
                "description": "Get the current temperature in a given location",
                "parameters": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
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
                },
            },
        }
    ]
    result = await async_client.chat.completions.create(
        extra_headers={
            "episode_id": episode_id,
            "variant_name": "openai",
        },
        messages=messages,
        model="tensorzero::basic_test",
        tools=tools,
    )
    assert result.model == "openai"
    assert result.episode_id == episode_id
    assert result.choices[0].message.content is None
    assert len(result.choices[0].message.tool_calls) == 1
    tool_call = result.choices[0].message.tool_calls[0]
    assert tool_call.type == "function"
    assert tool_call.function.name == "get_temperature"
    assert tool_call.function.arguments == '{"location":"Tokyo","units":"celsius"}'
    assert result.usage.prompt_tokens > 100
    assert result.usage.completion_tokens > 10


@pytest.mark.asyncio
async def test_dynamic_json_mode_inference_openai(async_client):
    episode_id = str(uuid7())
    output_schema = {
        "type": "object",
        "properties": {"response": {"type": "string"}},
        "required": ["response"],
        "additionalProperties": False,
    }
    serialized_output_schema = json.dumps(output_schema)
    messages = [
        {
            "role": "system",
            "content": [
                {"assistant_name": "Dr. Mehta", "schema": serialized_output_schema}
            ],
        },
        {"role": "user", "content": [{"country": "Japan"}]},
    ]
    result = await async_client.chat.completions.create(
        extra_headers={
            "episode_id": episode_id,
            "variant_name": "openai",
        },
        messages=messages,
        model="tensorzero::dynamic_json",
        response_format={"type": "json_schema", "schema": output_schema},
    )
    assert result.model == "openai"
    assert result.episode_id == episode_id
    json_content = json.loads(result.choices[0].message.content)
    assert "tokyo" in json_content["response"].lower()
    assert result.choices[0].message.tool_calls is None
    assert result.usage.prompt_tokens > 50
    assert result.usage.completion_tokens > 0
