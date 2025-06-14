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

import asyncio
import base64
import json
import os
from time import time
from uuid import UUID

import pytest
import pytest_asyncio
import tensorzero
from openai import AsyncOpenAI, BadRequestError
from pydantic import BaseModel, ValidationError
from tensorzero.util import uuid7

TEST_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../tensorzero-internal/tests/e2e/tensorzero.toml",
)


@pytest_asyncio.fixture
async def async_client():
    async with AsyncOpenAI(
        api_key="donotuse", base_url="http://localhost:3000/openai/v1"
    ) as client:
        yield client


@pytest.mark.asyncio
async def test_async_basic_inference_old_model_format_and_headers(async_client):
    messages = [
        {"role": "system", "content": [{"assistant_name": "Alfred Pennyworth"}]},
        {"role": "user", "content": "Hello"},
    ]

    result = await async_client.chat.completions.create(
        extra_headers={"episode_id": str(uuid7())},
        messages=messages,
        model="tensorzero::function_name::basic_test",
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
    assert result.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_async_basic_inference(async_client):
    messages = [
        {"role": "system", "content": [{"assistant_name": "Alfred Pennyworth"}]},
        {"role": "user", "content": "Hello"},
    ]

    result = await async_client.chat.completions.create(
        extra_body={
            "tensorzero::episode_id": str(uuid7()),
            "tensorzero::tags": {"foo": "bar"},
        },
        messages=messages,
        model="tensorzero::function_name::basic_test",
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
    assert result.choices[0].finish_reason == "stop"
    assert result.service_tier is None


class DummyModel(BaseModel):
    name: str


@pytest.mark.asyncio
async def test_async_basic_inference_json_schema(async_client):
    messages = [
        {"role": "system", "content": [{"assistant_name": "Alfred Pennyworth"}]},
        {"role": "user", "content": "Hello"},
    ]

    with pytest.raises(ValidationError) as exc_info:
        await async_client.beta.chat.completions.parse(
            extra_body={"tensorzero::episode_id": str(uuid7())},
            messages=messages,
            model="tensorzero::function_name::basic_test",
            temperature=0.4,
            response_format=DummyModel,
        )

    assert "Megumin gleefully" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_inference_cache(async_client):
    messages = [
        {"role": "system", "content": [{"assistant_name": "Alfred Pennyworth"}]},
        {"role": "user", "content": "Hello"},
    ]

    result = await async_client.chat.completions.create(
        messages=messages,
        model="tensorzero::function_name::basic_test",
        temperature=0.4,
    )

    assert (
        result.choices[0].message.content
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    usage = result.usage
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 10
    assert usage.total_tokens == 20

    # Test caching
    result = await async_client.chat.completions.create(
        extra_body={
            "tensorzero::cache_options": {"max_age_s": 10, "enabled": "on"},
        },
        messages=messages,
        model="tensorzero::function_name::basic_test",
        temperature=0.4,
    )

    assert (
        result.choices[0].message.content
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    usage = result.usage
    assert usage.prompt_tokens == 0  # should be cached
    assert usage.completion_tokens == 0  # should be cached
    assert usage.total_tokens == 0  # should be cached


@pytest.mark.asyncio
async def test_async_inference_streaming_with_cache(async_client):
    messages = [
        {"role": "system", "content": [{"assistant_name": "Alfred Pennyworth"}]},
        {"role": "user", "content": "Hello"},
    ]

    # First request without cache to populate the cache
    stream = await async_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": str(uuid7())},
        messages=messages,
        model="tensorzero::function_name::basic_test",
        stream=True,
        stream_options={"include_usage": True},
        seed=69,
    )

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    # Verify the response
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

    content = ""
    for i, chunk in enumerate(chunks[:-1]):  # All but the last chunk
        assert chunk.service_tier is None
        if i < len(expected_text):
            assert chunk.choices[0].delta.content == expected_text[i]
            content += chunk.choices[0].delta.content

    # Check second-to-last chunk has correct finish reason
    stop_chunk = chunks[-2]
    assert stop_chunk.choices[0].finish_reason == "stop"

    final_chunk = chunks[-1]
    assert final_chunk.usage.prompt_tokens == 10
    assert final_chunk.usage.completion_tokens == 16

    # Wait for trailing cache write to ClickHouse
    await asyncio.sleep(1)

    # Second request with cache
    stream = await async_client.chat.completions.create(
        extra_body={
            "tensorzero::episode_id": str(uuid7()),
            "tensorzero::cache_options": {"max_age_s": None, "enabled": "on"},
        },
        messages=messages,
        model="tensorzero::function_name::basic_test",
        stream=True,
        stream_options={"include_usage": True},
        seed=69,
    )

    cached_chunks = []
    async for chunk in stream:
        cached_chunks.append(chunk)

    # Verify we get the same content
    cached_content = ""
    for i, chunk in enumerate(cached_chunks[:-1]):  # All but the last chunk
        if i < len(expected_text):
            assert chunk.choices[0].delta.content == expected_text[i]
            cached_content += chunk.choices[0].delta.content

    assert content == cached_content

    # Check second-to-last chunk has the correct finish reason
    print("Chunks: ", cached_chunks)
    finish_chunk = cached_chunks[-2]
    assert finish_chunk.choices[0].finish_reason == "stop"

    final_cached_chunk = cached_chunks[-1]

    # In streaming mode, the cached response will not include usage statistics
    # This is still correct behavior as no tokens were used
    assert final_cached_chunk.usage.prompt_tokens == 0  # should be cached
    assert final_cached_chunk.usage.completion_tokens == 0  # should be cached
    assert final_cached_chunk.usage.total_tokens == 0  # should be cached


@pytest.mark.asyncio
async def test_async_inference_streaming(async_client):
    start_time = time()
    messages = [
        {"role": "system", "content": [{"assistant_name": "Alfred Pennyworth"}]},
        {"role": "user", "content": "Hello"},
    ]
    stream = await async_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": str(uuid7())},
        messages=messages,
        model="tensorzero::function_name::basic_test",
        stream=True,
        stream_options={"include_usage": True},
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
        assert (
            chunk.model == "tensorzero::function_name::basic_test::variant_name::test"
        )
        if i + 2 < len(chunks):
            assert len(chunk.choices) == 1
            assert chunk.choices[0].delta.content == expected_text[i]
            assert chunk.choices[0].finish_reason is None

    stop_chunk = chunks[-2]
    assert stop_chunk.choices[0].finish_reason == "stop"
    assert stop_chunk.choices[0].delta.content is None

    final_chunk = chunks[-1]
    assert len(final_chunk.choices) == 0
    assert final_chunk.usage.prompt_tokens == 10
    assert final_chunk.usage.completion_tokens == 16
    assert final_chunk.usage.total_tokens == 26


@pytest.mark.asyncio
async def test_async_inference_streaming_nonexistent_function(async_client):
    with pytest.raises(Exception) as exc_info:
        messages = [
            {"role": "system", "content": [{"assistant_name": "Alfred Pennyworth"}]},
            {"role": "user", "content": "Hello"},
        ]

        await async_client.chat.completions.create(
            extra_body={
                "tensorzero::episode_id": str(uuid7()),
            },
            messages=messages,
            model="tensorzero::function_name::does_not_exist",
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
            extra_body={
                "tensorzero::episode_id": str(uuid7()),
            },
            messages=messages,
            model="tensorzero::function_name::",
        )
    assert exc_info.value.status_code == 400
    assert (
        str(exc_info.value)
        == "Error code: 400 - {'error': 'Invalid request to OpenAI-compatible endpoint: function_name (passed in model field after \"tensorzero::function_name::\") cannot be empty'}"
    )


@pytest.mark.asyncio
async def test_async_inference_streaming_malformed_function(async_client):
    with pytest.raises(Exception) as exc_info:
        messages = [
            {"role": "system", "content": [{"assistant_name": "Alfred Pennyworth"}]},
            {"role": "user", "content": "Hello"},
        ]

        await async_client.chat.completions.create(
            extra_body={
                "tensorzero::episode_id": str(uuid7()),
            },
            messages=messages,
            model="chatgpt",
        )
    assert exc_info.value.status_code == 400
    assert (
        str(exc_info.value)
        == "Error code: 400 - {'error': 'Invalid request to OpenAI-compatible endpoint: `model` field must start with `tensorzero::function_name::` or `tensorzero::model_name::`. For example, `tensorzero::function_name::my_function` for a function `my_function` defined in your config, `tensorzero::model_name::my_model` for a model `my_model` defined in your config, or default functions like `tensorzero::model_name::openai::gpt-4o-mini`.'}"
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
            extra_body={"tensorzero::episode_id": str(uuid7())},
            messages=messages,
            model="tensorzero::function_name::basic_test",
            stream=True,
        )
    assert exc_info.value.status_code == 400
    assert "JSON Schema validation failed" in str(exc_info.value)


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
        extra_body={"tensorzero::episode_id": str(uuid7())},
        messages=messages,
        model="tensorzero::function_name::weather_helper",
        top_p=0.5,
    )
    assert (
        result.model
        == "tensorzero::function_name::weather_helper::variant_name::variant"
    )
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
    assert result.choices[0].finish_reason == "tool_calls"


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
        extra_body={
            "tensorzero::episode_id": str(uuid7()),
            "tensorzero::variant_name": "bad_tool",
        },
        messages=messages,
        model="tensorzero::function_name::weather_helper",
        presence_penalty=0.5,
    )
    assert (
        result.model
        == "tensorzero::function_name::weather_helper::variant_name::bad_tool"
    )
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
        extra_body={"tensorzero::episode_id": str(uuid7())},
        messages=messages,
        model="tensorzero::function_name::weather_helper",
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
    name_seen = False
    for i, chunk in enumerate(chunks):
        if previous_inference_id is not None:
            assert chunk.id == previous_inference_id
        if previous_episode_id is not None:
            assert chunk.episode_id == previous_episode_id
        previous_inference_id = chunk.id
        previous_episode_id = chunk.episode_id
        assert (
            chunk.model
            == "tensorzero::function_name::weather_helper::variant_name::variant"
        )
        if i + 1 < len(chunks):
            assert len(chunk.choices) == 1
            assert chunk.choices[0].delta.content is None
            assert len(chunk.choices[0].delta.tool_calls) == 1
            tool_call = chunk.choices[0].delta.tool_calls[0]
            assert tool_call.type == "function"
            if tool_call.function.name is not None and tool_call.function.name != "":
                assert not name_seen
                assert tool_call.function.name == "get_temperature"
                name_seen = True
            assert tool_call.function.arguments == expected_text[i]
        else:
            assert chunk.choices[0].delta.content is None
            assert chunk.choices[0].delta.tool_calls is None
            # We did not send 'include_usage'
            assert chunk.usage is None
            assert chunk.choices[0].finish_reason == "tool_calls"
    assert name_seen


@pytest.mark.asyncio
async def test_async_json_streaming(async_client):
    # We don't actually have a streaming JSON function implemented in `dummy.rs` but it doesn't matter for this test since
    # TensorZero doesn't parse the JSON output of the function for streaming calls.
    messages = [
        {"role": "system", "content": [{"assistant_name": "Alfred Pennyworth"}]},
        {"role": "user", "content": [{"country": "Japan"}]},
    ]
    stream = await async_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": str(uuid7())},
        messages=messages,
        model="tensorzero::function_name::json_success",
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
        assert (
            chunk.model == "tensorzero::function_name::json_success::variant_name::test"
        )
        if i + 1 < len(chunks):
            assert chunk.choices[0].delta.content == expected_text[i]
        else:
            assert len(chunk.choices[0].delta.content) == 0
            # We did not send 'include_usage'
            assert chunk.usage is None


@pytest.mark.asyncio
async def test_allow_developer_and_system(async_client):
    messages = [
        {
            "role": "developer",
            "content": [{"type": "text", "text": "Developer message."}],
        },
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "System message.",
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "User message."}],
        },
    ]
    episode_id = str(uuid7())

    result = await async_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": episode_id},
        messages=messages,
        model="tensorzero::model_name::dummy::echo_request_messages",
    )
    assert result.model == "tensorzero::model_name::dummy::echo_request_messages"
    assert result.episode_id == episode_id
    assert (
        result.choices[0].message.content
        == '{"system":"Developer message.\\nSystem message.","messages":[{"role":"user","content":[{"type":"text","text":"User message."}]}]}'
    )


@pytest.mark.asyncio
async def test_async_json_success_developer(async_client):
    messages = [
        {
            "role": "developer",
            "content": [
                {
                    "type": "text",
                    "tensorzero::arguments": {"assistant_name": "Alfred Pennyworth"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "tensorzero::arguments": {"country": "Japan"}}
            ],
        },
    ]
    episode_id = str(uuid7())
    result = await async_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": episode_id},
        messages=messages,
        model="tensorzero::function_name::json_success",
    )
    assert result.model == "tensorzero::function_name::json_success::variant_name::test"
    assert result.episode_id == episode_id
    assert result.choices[0].message.content == '{"answer":"Hello"}'
    assert result.choices[0].message.tool_calls is None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 10


@pytest.mark.asyncio
async def test_async_json_success_non_deprecated(async_client):
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "tensorzero::arguments": {"assistant_name": "Alfred Pennyworth"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "tensorzero::arguments": {"country": "Japan"}}
            ],
        },
    ]
    episode_id = str(uuid7())
    result = await async_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": episode_id},
        messages=messages,
        model="tensorzero::function_name::json_success",
    )
    assert result.model == "tensorzero::function_name::json_success::variant_name::test"
    assert result.episode_id == episode_id
    assert result.choices[0].message.content == '{"answer":"Hello"}'
    assert result.choices[0].message.tool_calls is None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 10


@pytest.mark.asyncio
async def test_async_json_success(async_client):
    messages = [
        {"role": "system", "content": [{"assistant_name": "Alfred Pennyworth"}]},
        {"role": "user", "content": [{"country": "Japan"}]},
    ]
    episode_id = str(uuid7())
    result = await async_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": episode_id},
        messages=messages,
        model="tensorzero::function_name::json_success",
    )
    assert result.model == "tensorzero::function_name::json_success::variant_name::test"
    assert result.episode_id == episode_id
    assert result.choices[0].message.content == '{"answer":"Hello"}'
    assert result.choices[0].message.tool_calls is None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 10


@pytest.mark.asyncio
async def test_async_json_success_strict(async_client):
    messages = [
        {"role": "system", "content": [{"assistant_name": "Alfred Pennyworth"}]},
        {"role": "user", "content": [{"country": "Japan"}]},
    ]
    episode_id = str(uuid7())
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "test",
            "description": "test",
            "schema": {
                "type": "object",
                "properties": {"response": {"type": "string"}},
                "required": ["response"],
                "additionalProperties": False,
                "strict": True,
            },
        },
    }
    result = await async_client.chat.completions.create(
        extra_body={
            "tensorzero::episode_id": episode_id,
            "tensorzero::variant_name": "test-diff-schema",
        },
        messages=messages,
        model="tensorzero::function_name::json_success",
        response_format=response_format,
    )
    assert (
        result.model
        == "tensorzero::function_name::json_success::variant_name::test-diff-schema"
    )
    assert result.episode_id == episode_id
    assert result.choices[0].message.content == '{"response":"Hello"}'
    assert result.choices[0].message.tool_calls is None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 10


@pytest.mark.asyncio
async def test_async_json_success_json_object(async_client):
    messages = [
        {"role": "system", "content": [{"assistant_name": "Alfred Pennyworth"}]},
        {"role": "user", "content": [{"country": "Japan"}]},
    ]
    episode_id = str(uuid7())
    response_format = {
        "type": "json_object",
    }
    result = await async_client.chat.completions.create(
        extra_body={
            "tensorzero::episode_id": episode_id,
            "tensorzero::variant_name": "test-diff-schema",
        },
        messages=messages,
        model="tensorzero::function_name::json_success",
        response_format=response_format,
    )
    assert (
        result.model
        == "tensorzero::function_name::json_success::variant_name::test-diff-schema"
    )
    assert result.episode_id == episode_id
    assert result.choices[0].message.content == '{"response":"Hello"}'
    assert result.choices[0].message.tool_calls is None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 10


@pytest.mark.asyncio
async def test_async_json_success_override(async_client):
    # Check that if we pass a string to a function with an input schema it is 400
    # We will add explicit support for raw text in the OpenAI API later
    messages = [
        {"role": "system", "content": [{"assistant_name": "Alfred Pennyworth"}]},
        {"role": "user", "content": [{"type": "text", "text": "Hi how are you?"}]},
        {"role": "user", "content": [{"country": "Japan"}]},
    ]
    episode_id = str(uuid7())
    with pytest.raises(BadRequestError) as exc_info:
        await async_client.chat.completions.create(
            extra_body={"tensorzero::episode_id": episode_id},
            messages=messages,
            model="tensorzero::function_name::json_success",
        )
    assert '"Hi how are you?" is not of type "object"' in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_json_invalid_system(async_client):
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"},
                }
            ],
        },
        {"role": "user", "content": [{"country": "Japan"}]},
    ]
    episode_id = str(uuid7())
    with pytest.raises(BadRequestError) as exc_info:
        await async_client.chat.completions.create(
            extra_body={"tensorzero::episode_id": episode_id},
            messages=messages,
            model="tensorzero::function_name::json_success",
        )
    assert (
        "Invalid request to OpenAI-compatible endpoint: System message must be a text content block"
        in str(exc_info.value)
    )


@pytest.mark.asyncio
async def test_async_extra_headers_param(async_client):
    messages = [
        {"role": "user", "content": "Hello, world!"},
    ]
    result = await async_client.chat.completions.create(
        extra_body={
            "tensorzero::extra_headers": [
                {
                    "model_provider_name": "tensorzero::model_name::dummy::echo_extra_info::provider_name::dummy",
                    "name": "x-my-extra-header",
                    "value": "my-extra-header-value",
                },
            ]
        },
        messages=messages,
        model="tensorzero::model_name::dummy::echo_extra_info",
    )
    assert result.model == "tensorzero::model_name::dummy::echo_extra_info"
    assert json.loads(result.choices[0].message.content) == {
        "extra_body": {"inference_extra_body": []},
        "extra_headers": {
            "inference_extra_headers": [
                {
                    "model_provider_name": "tensorzero::model_name::dummy::echo_extra_info::provider_name::dummy",
                    "name": "x-my-extra-header",
                    "value": "my-extra-header-value",
                }
            ],
            "variant_extra_headers": None,
        },
    }


@pytest.mark.asyncio
async def test_async_extra_body_param(async_client):
    messages = [
        {"role": "user", "content": "Hello, world!"},
    ]
    result = await async_client.chat.completions.create(
        extra_body={
            "tensorzero::extra_body": [
                {
                    "model_provider_name": "tensorzero::model_name::dummy::echo_extra_info::provider_name::dummy",
                    "pointer": "/thinking",
                    "value": {
                        "type": "enabled",
                        "budget_tokens": 1024,
                    },
                },
            ]
        },
        messages=messages,
        model="tensorzero::model_name::dummy::echo_extra_info",
    )
    assert result.model == "tensorzero::model_name::dummy::echo_extra_info"
    assert json.loads(result.choices[0].message.content) == {
        "extra_body": {
            "inference_extra_body": [
                {
                    "model_provider_name": "tensorzero::model_name::dummy::echo_extra_info::provider_name::dummy",
                    "pointer": "/thinking",
                    "value": {"type": "enabled", "budget_tokens": 1024},
                }
            ]
        },
        "extra_headers": {"variant_extra_headers": None, "inference_extra_headers": []},
    }


@pytest.mark.asyncio
async def test_async_json_failure(async_client):
    messages = [
        {"role": "system", "content": [{"assistant_name": "Alfred Pennyworth"}]},
        {"role": "user", "content": "Hello, world!"},
    ]
    result = await async_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": str(uuid7())},
        messages=messages,
        model="tensorzero::function_name::json_fail",
    )
    assert result.model == "tensorzero::function_name::json_fail::variant_name::test"
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
        extra_body={
            "tensorzero::episode_id": episode_id,
            "tensorzero::variant_name": "openai",
        },
        messages=messages,
        model="tensorzero::function_name::basic_test",
        tools=tools,
    )
    assert result.model == "tensorzero::function_name::basic_test::variant_name::openai"
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
async def test_dynamic_json_mode_inference_body_param_openai(async_client):
    header_episode_id = str(uuid7())
    body_episode_id = str(uuid7())
    output_schema = {
        "type": "object",
        "properties": {"response": {"type": "string"}},
        "required": ["response"],
        "additionalProperties": False,
    }
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "test",
            "description": "test",
            "schema": output_schema,
        },
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
            "episode_id": header_episode_id,
        },
        extra_body={
            "tensorzero::episode_id": body_episode_id,
            "tensorzero::variant_name": "openai",
        },
        messages=messages,
        model="tensorzero::function_name::dynamic_json",
        response_format=response_format,
    )
    assert (
        result.model == "tensorzero::function_name::dynamic_json::variant_name::openai"
    )
    assert result.episode_id == body_episode_id
    json_content = json.loads(result.choices[0].message.content)
    assert "tokyo" in json_content["response"].lower()
    assert result.choices[0].message.tool_calls is None
    assert result.usage.prompt_tokens > 50
    assert result.usage.completion_tokens > 0


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
                {"assistant_name": "Dr. Mehta", "schema": serialized_output_schema}
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
    assert (
        result.model == "tensorzero::function_name::dynamic_json::variant_name::openai"
    )
    assert result.episode_id == episode_id
    json_content = json.loads(result.choices[0].message.content)
    assert "tokyo" in json_content["response"].lower()
    assert result.choices[0].message.tool_calls is None
    assert result.usage.prompt_tokens > 50
    assert result.usage.completion_tokens > 0


@pytest.mark.asyncio
async def test_dynamic_json_mode_inference_openai_deprecated(async_client):
    episode_id = str(uuid7())
    output_schema = {
        "type": "object",
        "properties": {"response": {"type": "string"}},
        "required": ["response"],
        "additionalProperties": False,
    }
    serialized_output_schema = json.dumps(output_schema)
    # This response format is deprecated and will be rejected in a future TensorZero release.
    response_format = {
        "type": "json_schema",
        "json_schema": output_schema,
    }
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
        extra_body={
            "tensorzero::episode_id": episode_id,
            "tensorzero::variant_name": "openai",
        },
        messages=messages,
        model="tensorzero::function_name::dynamic_json",
        response_format=response_format,
    )
    assert (
        result.model == "tensorzero::function_name::dynamic_json::variant_name::openai"
    )
    assert result.episode_id == episode_id
    json_content = json.loads(result.choices[0].message.content)
    assert "tokyo" in json_content["response"].lower()
    assert result.choices[0].message.tool_calls is None
    assert result.usage.prompt_tokens > 50
    assert result.usage.completion_tokens > 0


@pytest.mark.asyncio
async def test_async_multi_system_prompt(async_client):
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "My first system input.",
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "My text input",
                },
            ],
        },
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "My second system input.",
                },
                {
                    "type": "text",
                    "text": "My third system input.",
                },
            ],
        },
    ]
    episode_id = str(uuid7())
    result = await async_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": episode_id},
        messages=messages,
        model="tensorzero::model_name::dummy::echo_request_messages",
    )
    assert (
        result.choices[0].message.content
        == '{"system":"My first system input.\\nMy second system input.\\nMy third system input.","messages":[{"role":"user","content":[{"type":"text","text":"My text input"}]}]}'
    )


@pytest.mark.asyncio
async def test_async_multi_block_image_url(async_client):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Output exactly two words describing the image",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png"
                    },
                },
            ],
        },
    ]
    episode_id = str(uuid7())
    result = await async_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": episode_id},
        messages=messages,
        model="tensorzero::model_name::openai::gpt-4o-mini",
    )
    assert "crab" in result.choices[0].message.content.lower()


@pytest.mark.asyncio
async def test_async_multi_block_image_base64(async_client):
    basepath = os.path.dirname(__file__)
    with open(
        f"{basepath}/../../../tensorzero-internal/tests/e2e/providers/ferris.png", "rb"
    ) as f:
        ferris_png = base64.b64encode(f.read()).decode("ascii")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Output exactly two words describing the image",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{ferris_png}"},
                },
            ],
        },
    ]
    episode_id = str(uuid7())
    result = await async_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": episode_id},
        messages=messages,
        model="tensorzero::model_name::openai::gpt-4o-mini",
    )
    assert "crab" in result.choices[0].message.content.lower()


@pytest.mark.asyncio
async def test_async_multi_block_file_base64(async_client):
    basepath = os.path.dirname(__file__)
    with open(
        f"{basepath}/../../../tensorzero-internal/tests/e2e/providers/deepseek_paper.pdf",
        "rb",
    ) as f:
        deepseek_paper_pdf = base64.b64encode(f.read()).decode("ascii")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Output exactly two words describing the image",
                },
                {
                    "type": "file",
                    "file": {"file_data": deepseek_paper_pdf, "filename": "test.pdf"},
                },
            ],
        },
    ]
    episode_id = str(uuid7())
    result = await async_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": episode_id},
        messages=messages,
        model="tensorzero::model_name::dummy::require_pdf",
    )
    assert result.choices[0].message.content is not None
    json_content = json.loads(result.choices[0].message.content)
    assert json_content[0]["storage_path"] == {
        "kind": {"type": "disabled"},
        "path": "observability/files/3e127d9a726f6be0fd81d73ccea97d96ec99419f59650e01d49183cd3be999ef.pdf",
    }


@pytest.mark.asyncio
async def test_async_multi_turn_parallel_tool_use(async_client):
    episode_id = str(uuid7())

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "tensorzero::arguments": {"assistant_name": "Dr. Mehta"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is the weather like in Tokyo (in Fahrenheit)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions.",
                }
            ],
        },
    ]

    response = await async_client.chat.completions.create(
        messages=messages,
        model="tensorzero::function_name::weather_helper_parallel",
        parallel_tool_calls=True,
        extra_body={
            "tensorzero::episode_id": episode_id,
            "tensorzero::variant_name": "openai",
        },
    )

    assistant_message = response.choices[0].message
    messages.append(assistant_message)

    assert len(assistant_message.tool_calls) == 2

    for tool_call in assistant_message.tool_calls:
        if tool_call.function.name == "get_temperature":
            messages.append(
                {
                    "role": "tool",
                    "content": "70",
                    "tool_call_id": tool_call.id,
                }
            )
        elif tool_call.function.name == "get_humidity":
            messages.append(
                {
                    "role": "tool",
                    "content": "30",
                    "tool_call_id": tool_call.id,
                }
            )
        else:
            raise Exception(f"Unknown tool call: {tool_call.function.name}")

    response = await async_client.chat.completions.create(
        extra_body={
            "tensorzero::episode_id": episode_id,
            "tensorzero::variant_name": "openai",
        },
        model="tensorzero::function_name::weather_helper_parallel",
        messages=messages,
    )

    assistant_message = response.choices[0].message

    assert "70" in assistant_message.content
    assert "30" in assistant_message.content


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
    assert (
        result.model
        == "tensorzero::function_name::generate_haiku::variant_name::gpt_4o_mini"
    )

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
    assert (
        result.model
        == "tensorzero::function_name::generate_haiku::variant_name::gpt_4o_mini"
    )

    tensorzero.close_patched_openai_client_gateway(patched_client)


@pytest.mark.asyncio
async def test_async_chat_function_null_response(async_client):
    """
    Test that an chat inference with null response (i.e. no generated content blocks) works as expected.
    """
    result = await async_client.chat.completions.create(
        model="tensorzero::function_name::null_chat",
        messages=[
            {
                "role": "user",
                "content": "No yapping!",
            }
        ],
    )

    assert result.model == "tensorzero::function_name::null_chat::variant_name::variant"
    assert result.choices[0].message.content is None


@pytest.mark.asyncio
async def test_async_json_function_null_response(async_client):
    """
    Test that a JSON inference with null response (i.e. no generated content blocks) works as expected.
    """
    result = await async_client.chat.completions.create(
        model="tensorzero::function_name::null_json",
        messages=[
            {
                "role": "user",
                "content": "Extract no data!",
            }
        ],
    )
    assert result.model == "tensorzero::function_name::null_json::variant_name::variant"
    assert result.choices[0].message.content is None


@pytest.mark.asyncio
async def test_async_json_function_multiple_text_blocks(async_client):
    """
    Test that a JSON inference with 2 text blocks in the message works as expected.
    """
    result = await async_client.chat.completions.create(
        model="tensorzero::model_name::dummy::multiple-text-blocks",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract no data!",
                    },
                    {
                        "type": "text",
                        "text": "Extract data!",
                    },
                ],
            }
        ],
    )
    assert result.model == "tensorzero::model_name::dummy::multiple-text-blocks"
