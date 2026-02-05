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
from time import sleep, time
from uuid import UUID

import pytest
from openai import BadRequestError
from pydantic import BaseModel, ValidationError
from uuid_utils.compat import uuid7

TEST_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../tensorzero-core/tests/e2e/config/tensorzero.*.toml",
)


@pytest.mark.asyncio
async def test_async_basic_inference(async_openai_client):
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
        {"role": "user", "content": "Hello"},
    ]

    result = await async_openai_client.chat.completions.create(
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
    assert usage.completion_tokens == 1
    assert usage.total_tokens == 11
    assert result.choices[0].finish_reason == "stop"
    assert result.service_tier is None


class DummyModel(BaseModel):
    name: str


@pytest.mark.asyncio
async def test_async_basic_inference_json_schema(async_openai_client):
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
        {"role": "user", "content": "Hello"},
    ]

    with pytest.raises(ValidationError) as exc_info:
        await async_openai_client.beta.chat.completions.parse(
            extra_body={"tensorzero::episode_id": str(uuid7())},
            messages=messages,
            model="tensorzero::function_name::basic_test",
            temperature=0.4,
            response_format=DummyModel,
        )

    assert "Megumin gleefully" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_inference_cache(async_openai_client):
    uuid = uuid7()
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "tensorzero::arguments": {"assistant_name": f"Alfred Pennyworth ({uuid})"},
                }
            ],
        },
        {"role": "user", "content": "Hello"},
    ]

    result = await async_openai_client.chat.completions.create(
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
    assert usage.completion_tokens == 1
    assert usage.total_tokens == 11
    sleep(1)

    # Test caching
    result = await async_openai_client.chat.completions.create(
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
async def test_async_inference_streaming_with_cache(async_openai_client):
    uuid = str(uuid7())
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "tensorzero::arguments": {"assistant_name": f"Alfred Pennyworth ({uuid})"},
                }
            ],
        },
        {"role": "user", "content": "Hello"},
    ]

    # First request without cache to populate the cache
    stream = await async_openai_client.chat.completions.create(
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

    # Check last chunk has correct finish reason and usage
    final_chunk = chunks[-1]
    assert final_chunk.choices[0].finish_reason == "stop"
    assert final_chunk.usage.prompt_tokens == 10
    assert final_chunk.usage.completion_tokens == 16

    # Wait for trailing cache write to ClickHouse
    await asyncio.sleep(1)

    # Second request with cache
    stream = await async_openai_client.chat.completions.create(
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

    # Check last chunk has the correct finish reason and usage
    print("Chunks: ", cached_chunks)
    final_cached_chunk = cached_chunks[-1]
    assert final_cached_chunk.choices[0].finish_reason == "stop"

    # In streaming mode, the cached response will not include usage statistics
    # This is still correct behavior as no tokens were used
    assert final_cached_chunk.usage.prompt_tokens == 0  # should be cached
    assert final_cached_chunk.usage.completion_tokens == 0  # should be cached
    assert final_cached_chunk.usage.total_tokens == 0  # should be cached


@pytest.mark.asyncio
async def test_async_inference_streaming(async_openai_client):
    start_time = time()
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
        {"role": "user", "content": "Hello"},
    ]
    stream = await async_openai_client.chat.completions.create(
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
        assert chunk.model == "tensorzero::function_name::basic_test::variant_name::test"
        if i + 1 < len(chunks):
            assert len(chunk.choices) == 1
            assert chunk.choices[0].delta.content == expected_text[i]
            assert chunk.choices[0].finish_reason is None

    final_chunk = chunks[-1]
    assert final_chunk.choices[0].finish_reason == "stop"
    assert final_chunk.choices[0].delta.content is None
    assert final_chunk.usage.prompt_tokens == 10
    assert final_chunk.usage.completion_tokens == 16
    assert final_chunk.usage.total_tokens == 26


@pytest.mark.asyncio
async def test_async_inference_streaming_nonexistent_function(async_openai_client):
    with pytest.raises(Exception) as exc_info:
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
            {"role": "user", "content": "Hello"},
        ]

        await async_openai_client.chat.completions.create(
            extra_body={
                "tensorzero::episode_id": str(uuid7()),
            },
            messages=messages,
            model="tensorzero::function_name::does_not_exist",
        )
    assert exc_info.value.status_code == 404
    if not hasattr(async_openai_client, "__tensorzero_gateway"):
        # TODO(#3192): handle json errors in patched client
        assert (
            str(exc_info.value)
            == "Error code: 404 - {'error': {'message': 'Unknown function: does_not_exist', 'error_json': {'UnknownFunction': {'name': 'does_not_exist'}}, 'tensorzero_error_json': {'UnknownFunction': {'name': 'does_not_exist'}}}}"
        )


@pytest.mark.asyncio
async def test_async_inference_streaming_missing_function(async_openai_client):
    with pytest.raises(Exception) as exc_info:
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
            {"role": "user", "content": "Hello"},
        ]

        await async_openai_client.chat.completions.create(
            extra_body={
                "tensorzero::episode_id": str(uuid7()),
            },
            messages=messages,
            model="tensorzero::function_name::",
        )
    assert exc_info.value.status_code == 400
    if not hasattr(async_openai_client, "__tensorzero_gateway"):
        # TODO(#3192): handle json errors in patched client
        assert (
            str(exc_info.value)
            == """Error code: 400 - {'error': {'message': 'Invalid request to OpenAI-compatible endpoint: function_name (passed in model field after "tensorzero::function_name::") cannot be empty', 'error_json': {'InvalidOpenAICompatibleRequest': {'message': 'function_name (passed in model field after "tensorzero::function_name::") cannot be empty'}}, 'tensorzero_error_json': {'InvalidOpenAICompatibleRequest': {'message': 'function_name (passed in model field after "tensorzero::function_name::") cannot be empty'}}}}"""
        )


@pytest.mark.asyncio
async def test_async_inference_streaming_malformed_function(async_openai_client):
    with pytest.raises(Exception) as exc_info:
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
            {"role": "user", "content": "Hello"},
        ]

        await async_openai_client.chat.completions.create(
            extra_body={
                "tensorzero::episode_id": str(uuid7()),
            },
            messages=messages,
            model="chatgpt",
        )
    assert exc_info.value.status_code == 400
    if not hasattr(async_openai_client, "__tensorzero_gateway"):
        # TODO(#3192): handle json errors in patched client
        assert (
            str(exc_info.value)
            == """Error code: 400 - {'error': {'message': 'Invalid request to OpenAI-compatible endpoint: `model` field must start with `tensorzero::function_name::` or `tensorzero::model_name::`. For example, `tensorzero::function_name::my_function` for a function `my_function` defined in your config, `tensorzero::model_name::my_model` for a model `my_model` defined in your config, or default functions like `tensorzero::model_name::openai::gpt-4o-mini`.', 'error_json': {'InvalidOpenAICompatibleRequest': {'message': '`model` field must start with `tensorzero::function_name::` or `tensorzero::model_name::`. For example, `tensorzero::function_name::my_function` for a function `my_function` defined in your config, `tensorzero::model_name::my_model` for a model `my_model` defined in your config, or default functions like `tensorzero::model_name::openai::gpt-4o-mini`.'}}, 'tensorzero_error_json': {'InvalidOpenAICompatibleRequest': {'message': '`model` field must start with `tensorzero::function_name::` or `tensorzero::model_name::`. For example, `tensorzero::function_name::my_function` for a function `my_function` defined in your config, `tensorzero::model_name::my_model` for a model `my_model` defined in your config, or default functions like `tensorzero::model_name::openai::gpt-4o-mini`.'}}}}"""
        )


@pytest.mark.asyncio
async def test_async_inference_streaming_missing_model(async_openai_client):
    with pytest.raises(Exception) as exc_info:
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
            {"role": "user", "content": "Hello"},
        ]

        await async_openai_client.chat.completions.create(
            messages=messages,
        )
    assert (
        str(exc_info.value)
        == "Missing required arguments; Expected either ('messages' and 'model') or ('messages', 'model' and 'stream') arguments to be given"
    )


@pytest.mark.asyncio
async def test_async_inference_streaming_malformed_input(async_openai_client):
    with pytest.raises(Exception) as exc_info:
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "tensorzero::arguments": {"name_of_assistant": "Alfred Pennyworth"},
                    }
                ],
            },
            {"role": "user", "content": "Hello"},
        ]
        await async_openai_client.chat.completions.create(
            extra_body={"tensorzero::episode_id": str(uuid7())},
            messages=messages,
            model="tensorzero::function_name::basic_test",
            stream=True,
        )
    assert exc_info.value.status_code == 400
    assert "JSON Schema validation failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_tool_call_inference(async_openai_client):
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
            "content": "Hi I'm visiting Brooklyn from Brazil. What's the weather?",
        },
    ]
    result = await async_openai_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": str(uuid7())},
        messages=messages,
        model="tensorzero::function_name::weather_helper",
        top_p=0.5,
    )
    assert result.model == "tensorzero::function_name::weather_helper::variant_name::variant"
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
    assert usage.completion_tokens == 1
    assert result.choices[0].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_async_malformed_tool_call_inference(async_openai_client):
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
            "content": "Hi I'm visiting Brooklyn from Brazil. What's the weather?",
        },
    ]
    result = await async_openai_client.chat.completions.create(
        extra_body={
            "tensorzero::episode_id": str(uuid7()),
            "tensorzero::variant_name": "bad_tool",
        },
        messages=messages,
        model="tensorzero::function_name::weather_helper",
        presence_penalty=0.5,
    )
    assert result.model == "tensorzero::function_name::weather_helper::variant_name::bad_tool"
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
    assert usage.completion_tokens == 1


@pytest.mark.asyncio
async def test_async_tool_call_streaming(async_openai_client):
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
            "content": "Hi I'm visiting Brooklyn from Brazil. What's the weather?",
        },
    ]
    stream = await async_openai_client.chat.completions.create(
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
        assert chunk.model == "tensorzero::function_name::weather_helper::variant_name::variant"
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
async def test_async_json_streaming(async_openai_client):
    # Pick a variant that doesn't have a dummy provider streaming special-case
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
            "content": [{"type": "text", "tensorzero::arguments": {"country": "Japan"}}],
        },
    ]
    stream = await async_openai_client.chat.completions.create(
        extra_body={
            "tensorzero::episode_id": str(uuid7()),
            "tensorzero::variant_name": "test-diff-schema",
        },
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
        assert chunk.model == "tensorzero::function_name::json_success::variant_name::test-diff-schema"
        if i + 1 < len(chunks):
            assert chunk.choices[0].delta.content == expected_text[i]
        else:
            assert len(chunk.choices[0].delta.content) == 0
            # We did not send 'include_usage'
            assert chunk.usage is None


@pytest.mark.asyncio
async def test_allow_developer_and_system(async_openai_client):
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

    result = await async_openai_client.chat.completions.create(
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
async def test_async_json_success_developer(async_openai_client):
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
            "content": [{"type": "text", "tensorzero::arguments": {"country": "Japan"}}],
        },
    ]
    episode_id = str(uuid7())
    result = await async_openai_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": episode_id},
        messages=messages,
        model="tensorzero::function_name::json_success",
    )
    assert result.model == "tensorzero::function_name::json_success::variant_name::test"
    assert result.episode_id == episode_id
    assert result.choices[0].message.content == '{"answer":"Hello"}'
    assert result.choices[0].message.tool_calls is None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 1


@pytest.mark.asyncio
async def test_async_json_success_non_deprecated(async_openai_client):
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
            "content": [{"type": "text", "tensorzero::arguments": {"country": "Japan"}}],
        },
    ]
    episode_id = str(uuid7())
    result = await async_openai_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": episode_id},
        messages=messages,
        model="tensorzero::function_name::json_success",
    )
    assert result.model == "tensorzero::function_name::json_success::variant_name::test"
    assert result.episode_id == episode_id
    assert result.choices[0].message.content == '{"answer":"Hello"}'
    assert result.choices[0].message.tool_calls is None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 1


@pytest.mark.asyncio
async def test_async_json_success(async_openai_client):
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
            "content": [{"type": "text", "tensorzero::arguments": {"country": "Japan"}}],
        },
    ]
    episode_id = str(uuid7())
    result = await async_openai_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": episode_id},
        messages=messages,
        model="tensorzero::function_name::json_success",
    )
    assert result.model == "tensorzero::function_name::json_success::variant_name::test"
    assert result.episode_id == episode_id
    assert result.choices[0].message.content == '{"answer":"Hello"}'
    assert result.choices[0].message.tool_calls is None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 1


@pytest.mark.asyncio
async def test_async_json_success_strict(async_openai_client):
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
            "content": [{"type": "text", "tensorzero::arguments": {"country": "Japan"}}],
        },
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
    result = await async_openai_client.chat.completions.create(
        extra_body={
            "tensorzero::episode_id": episode_id,
            "tensorzero::variant_name": "test-diff-schema",
        },
        messages=messages,
        model="tensorzero::function_name::json_success",
        response_format=response_format,
    )
    assert result.model == "tensorzero::function_name::json_success::variant_name::test-diff-schema"
    assert result.episode_id == episode_id
    assert result.choices[0].message.content == '{"response":"Hello"}'
    assert result.choices[0].message.tool_calls is None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 1


@pytest.mark.asyncio
async def test_async_json_success_json_object(async_openai_client):
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
            "content": [{"type": "text", "tensorzero::arguments": {"country": "Japan"}}],
        },
    ]
    episode_id = str(uuid7())
    response_format = {
        "type": "json_object",
    }
    result = await async_openai_client.chat.completions.create(
        extra_body={
            "tensorzero::episode_id": episode_id,
            "tensorzero::variant_name": "test-diff-schema",
        },
        messages=messages,
        model="tensorzero::function_name::json_success",
        response_format=response_format,
    )
    assert result.model == "tensorzero::function_name::json_success::variant_name::test-diff-schema"
    assert result.episode_id == episode_id
    assert result.choices[0].message.content == '{"response":"Hello"}'
    assert result.choices[0].message.tool_calls is None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 1


@pytest.mark.asyncio
async def test_async_json_success_override(async_openai_client):
    # Check that if we pass a string to a function with an input schema it is 400
    # We will add explicit support for raw text in the OpenAI API later
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
        {"role": "user", "content": [{"type": "text", "text": "Hi how are you?"}]},
        {
            "role": "user",
            "content": [{"type": "text", "tensorzero::arguments": {"country": "Japan"}}],
        },
    ]
    episode_id = str(uuid7())
    with pytest.raises(BadRequestError) as exc_info:
        await async_openai_client.chat.completions.create(
            extra_body={"tensorzero::episode_id": episode_id},
            messages=messages,
            model="tensorzero::function_name::json_success",
        )
    assert '"Hi how are you?" is not of type "object"' in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_json_invalid_system(async_openai_client):
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
        {
            "role": "user",
            "content": [{"type": "text", "tensorzero::arguments": {"country": "Japan"}}],
        },
    ]
    episode_id = str(uuid7())
    with pytest.raises(BadRequestError) as exc_info:
        await async_openai_client.chat.completions.create(
            extra_body={"tensorzero::episode_id": episode_id},
            messages=messages,
            model="tensorzero::function_name::json_success",
        )
    assert (
        "Invalid request to OpenAI-compatible endpoint: System message must contain only text or template content blocks"
        in str(exc_info.value)
    )


@pytest.mark.asyncio
async def test_missing_text_fields(async_openai_client):
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
            "content": [{"type": "text"}],
        },
    ]
    with pytest.raises(BadRequestError) as exc_info:
        await async_openai_client.chat.completions.create(
            messages=messages,
            model="tensorzero::function_name::json_success",
        )
    assert (
        'Invalid request to OpenAI-compatible endpoint: Invalid content block: Either `text` or `tensorzero::arguments` must be set when using `"type": "text"`'
        in str(exc_info.value)
    )


@pytest.mark.asyncio
async def test_bad_content_block_type(async_openai_client):
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
            "content": [{"type": "my_fake_type", "my": "other_field"}],
        },
    ]
    with pytest.raises(BadRequestError) as exc_info:
        await async_openai_client.chat.completions.create(
            messages=messages,
            model="tensorzero::function_name::json_success",
        )
    assert (
        "Invalid request to OpenAI-compatible endpoint: Invalid content block: unknown variant `my_fake_type`, expected one of `text`, `image_url`, `file`"
        in str(exc_info.value)
    )


@pytest.mark.asyncio
async def test_invalid_tensorzero_text_block(async_openai_client):
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
                {
                    "type": "text",
                    "text": "My other text",
                    "tensorzero::arguments": {"country": "Japan"},
                }
            ],
        },
    ]
    with pytest.raises(BadRequestError) as exc_info:
        await async_openai_client.chat.completions.create(
            messages=messages,
            model="tensorzero::function_name::json_success",
        )
    assert (
        'Invalid request to OpenAI-compatible endpoint: Invalid TensorZero content block: Only one of `text` or `tensorzero::arguments` can be set when using `"type": "text"`'
        in str(exc_info.value)
    )


@pytest.mark.asyncio
async def test_async_extra_headers_param(async_openai_client):
    messages = [
        {"role": "user", "content": "Hello, world!"},
    ]
    result = await async_openai_client.chat.completions.create(
        extra_body={
            "tensorzero::extra_headers": [
                {
                    "model_name": "dummy::echo_injected_data",
                    "provider_name": "dummy",
                    "name": "x-my-extra-header",
                    "value": "my-extra-header-value",
                },
                {
                    "variant_name": "dummy::echo_injected_data",
                    "name": "x-my-variant-header",
                    "value": "my-variant-value",
                },
                # This header will get added, and then immediately deleted by the subsequence 'delete = True' entry
                # The 'dummy::echo_injected_data' models echos back the final header map (after all 'extra_headers' replacements are applied),
                # and we assert that it only contains 'x-my-extra-header'
                {
                    "variant_name": "dummy::echo_injected_data",
                    "name": "x-my-delete-header",
                    "value": "Should be deleted",
                },
                {
                    "variant_name": "dummy::echo_injected_data",
                    "name": "x-my-delete-header",
                    "delete": True,
                },
            ]
        },
        messages=messages,
        model="tensorzero::model_name::dummy::echo_injected_data",
    )
    assert result.model == "tensorzero::model_name::dummy::echo_injected_data"
    assert json.loads(result.choices[0].message.content) == {
        "injected_body": {},
        "injected_headers": [
            ["x-my-extra-header", "my-extra-header-value"],
            ["x-my-variant-header", "my-variant-value"],
        ],
    }


@pytest.mark.asyncio
async def test_async_extra_body_param(async_openai_client):
    messages = [
        {"role": "user", "content": "Hello, world!"},
    ]
    result = await async_openai_client.chat.completions.create(
        extra_body={
            "tensorzero::extra_body": [
                {
                    "model_name": "dummy::echo_extra_info",
                    "provider_name": "dummy",
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
                    "model_name": "dummy::echo_extra_info",
                    "provider_name": "dummy",
                    "pointer": "/thinking",
                    "value": {"type": "enabled", "budget_tokens": 1024},
                }
            ]
        },
        "extra_headers": {"variant_extra_headers": None, "inference_extra_headers": []},
    }


@pytest.mark.asyncio
async def test_async_json_failure(async_openai_client):
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
        {"role": "user", "content": "Hello, world!"},
    ]
    result = await async_openai_client.chat.completions.create(
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
    assert result.usage.completion_tokens == 1


@pytest.mark.asyncio
async def test_dynamic_tool_use_inference_openai(async_openai_client):
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
    result = await async_openai_client.chat.completions.create(
        extra_body={
            "tensorzero::episode_id": episode_id,
            "tensorzero::variant_name": "openai-responses",
        },
        messages=messages,
        model="tensorzero::function_name::basic_test",
        tools=tools,
    )
    assert result.model == "tensorzero::function_name::basic_test::variant_name::openai-responses"
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
async def test_dynamic_json_mode_inference_body_param_openai(async_openai_client):
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
                {
                    "type": "text",
                    "tensorzero::arguments": {
                        "assistant_name": "Dr. Mehta",
                        "schema": serialized_output_schema,
                    },
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "tensorzero::arguments": {"country": "Japan"}}],
        },
    ]
    result = await async_openai_client.chat.completions.create(
        extra_body={
            "tensorzero::episode_id": body_episode_id,
            "tensorzero::variant_name": "openai",
        },
        messages=messages,
        model="tensorzero::function_name::dynamic_json",
        response_format=response_format,
    )
    assert result.model == "tensorzero::function_name::dynamic_json::variant_name::openai"
    assert result.episode_id == body_episode_id
    json_content = json.loads(result.choices[0].message.content)
    assert "tokyo" in json_content["response"].lower()
    assert result.choices[0].message.tool_calls is None
    assert result.usage.prompt_tokens > 50
    assert result.usage.completion_tokens > 0


@pytest.mark.asyncio
async def test_dynamic_json_mode_inference_openai(async_openai_client):
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
                    "type": "text",
                    "tensorzero::arguments": {
                        "assistant_name": "Dr. Mehta",
                        "schema": serialized_output_schema,
                    },
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "tensorzero::arguments": {"country": "Japan"}}],
        },
    ]
    result = await async_openai_client.chat.completions.create(
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
async def test_async_multi_system_prompt(async_openai_client):
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
    result = await async_openai_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": episode_id},
        messages=messages,
        model="tensorzero::model_name::dummy::echo_request_messages",
    )
    assert (
        result.choices[0].message.content
        == '{"system":"My first system input.\\nMy second system input.\\nMy third system input.","messages":[{"role":"user","content":[{"type":"text","text":"My text input"}]}]}'
    )


@pytest.mark.asyncio
async def test_async_multi_block_image_url(async_openai_client):
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
    result = await async_openai_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": episode_id},
        messages=messages,
        model="tensorzero::model_name::openai::gpt-4o-mini",
    )
    assert "crab" in result.choices[0].message.content.lower()


@pytest.mark.asyncio
async def test_async_multi_block_image_base64(async_openai_client):
    basepath = os.path.dirname(__file__)
    with open(f"{basepath}/../../../tensorzero-core/tests/e2e/providers/ferris.png", "rb") as f:
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
    result = await async_openai_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": episode_id},
        messages=messages,
        model="tensorzero::model_name::openai::gpt-4o-mini",
    )
    assert "crab" in result.choices[0].message.content.lower()


@pytest.mark.asyncio
async def test_async_multi_block_file_base64(async_openai_client):
    basepath = os.path.dirname(__file__)
    with open(
        f"{basepath}/../../../tensorzero-core/tests/e2e/providers/deepseek_paper.pdf",
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
                    "file": {"file_data": f"data:application/pdf;base64,{deepseek_paper_pdf}", "filename": "test.pdf"},
                },
            ],
        },
    ]
    episode_id = str(uuid7())
    result = await async_openai_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": episode_id},
        messages=messages,
        model="tensorzero::model_name::dummy::require_pdf",
    )
    assert result.choices[0].message.content is not None
    json_content = json.loads(result.choices[0].message.content)
    assert json_content[0]["Base64"]["storage_path"] == {
        "kind": {"type": "disabled"},
        "path": "observability/files/3e127d9a726f6be0fd81d73ccea97d96ec99419f59650e01d49183cd3be999ef.pdf",
    }


@pytest.mark.asyncio
async def test_async_multi_turn_parallel_tool_use(async_openai_client):
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

    response = await async_openai_client.chat.completions.create(
        messages=messages,
        model="tensorzero::function_name::weather_helper_parallel",
        parallel_tool_calls=True,
        extra_body={
            "tensorzero::episode_id": episode_id,
            "tensorzero::variant_name": "openai-responses",
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

    response = await async_openai_client.chat.completions.create(
        extra_body={
            "tensorzero::episode_id": episode_id,
            "tensorzero::variant_name": "openai-responses",
        },
        model="tensorzero::function_name::weather_helper_parallel",
        messages=messages,
    )

    assistant_message = response.choices[0].message

    assert "70" in assistant_message.content
    assert "30" in assistant_message.content


@pytest.mark.asyncio
async def test_async_chat_function_null_response(async_openai_client):
    """
    Test that an chat inference with null response (i.e. no generated content blocks) works as expected.
    """
    result = await async_openai_client.chat.completions.create(
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
async def test_async_json_function_null_response(async_openai_client):
    """
    Test that a JSON inference with null response (i.e. no generated content blocks) works as expected.
    """
    result = await async_openai_client.chat.completions.create(
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
async def test_async_json_function_multiple_text_blocks(async_openai_client):
    """
    Test that a JSON inference with 2 text blocks in the message works as expected.
    """
    result = await async_openai_client.chat.completions.create(
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


@pytest.mark.asyncio
async def test_async_inference_tensorzero_raw_text(async_openai_client):
    """
    Test that chat inference with a tensorzero::raw_text block works correctly
    """
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "tensorzero::arguments": {"assistant_name": "Megumin"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is the capital of Japan?"}],
        },
    ]
    response = await async_openai_client.chat.completions.create(
        messages=messages,
        model="tensorzero::function_name::openai_with_assistant_schema",
    )

    assert "tokyo" in response.choices[0].message.content.lower()

    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tensorzero::raw_text",
                    "value": "You're a mischievous assistant that says fake information. Very concise.",
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is the capital of Japan?"}],
        },
    ]
    response = await async_openai_client.chat.completions.create(
        messages=messages,
        model="tensorzero::function_name::openai_with_assistant_schema",
    )

    assert "tokyo" not in response.choices[0].message.content.lower()
    assert response.model == "tensorzero::function_name::openai_with_assistant_schema::variant_name::openai"


@pytest.mark.asyncio
async def test_async_inference_tensorzero_template(async_openai_client):
    """
    Test that chat inference with a tensorzero::template block works correctly
    """
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tensorzero::template",
                    "name": "assistant",
                    "arguments": {"assistant_name": "Megumin"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is the capital of Japan?"}],
        },
    ]
    response = await async_openai_client.chat.completions.create(
        messages=messages,
        model="tensorzero::function_name::openai_with_assistant_schema",
    )

    assert "tokyo" in response.choices[0].message.content.lower()


@pytest.mark.asyncio
async def test_openai_custom_tool_text_format(async_openai_client):
    """
    Test OpenAI custom tool with text format output
    """
    episode_id = str(uuid7())
    messages = [
        {
            "role": "user",
            "content": "Generate Python code to print 'Hello, World!' using the code_generator tool.",
        },
    ]
    tools = [
        {
            "type": "custom",
            "custom": {
                "name": "code_generator",
                "description": "Generates Python code snippets based on requirements",
                "format": {"type": "text"},
            },
        }
    ]
    result = await async_openai_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": episode_id},
        messages=messages,
        model="tensorzero::model_name::openai::responses::gpt-5-codex",
        tools=tools,
    )
    assert result.model == "tensorzero::model_name::openai::responses::gpt-5-codex"
    assert result.episode_id == episode_id
    # Check that we got tool calls in the response
    assert result.choices[0].message.tool_calls is not None
    assert len(result.choices[0].message.tool_calls) >= 1
    # Find the code_generator tool call
    code_generator_calls = [tc for tc in result.choices[0].message.tool_calls if tc.function.name == "code_generator"]
    assert len(code_generator_calls) == 1
    tool_call = code_generator_calls[0]
    assert tool_call.type == "function"
    assert tool_call.function.name == "code_generator"
    assert tool_call.function.arguments is not None
    assert len(tool_call.function.arguments) > 0


@pytest.mark.asyncio
async def test_openai_custom_tool_grammar_lark(async_openai_client):
    """
    Test OpenAI custom tool with Lark grammar format
    """
    episode_id = str(uuid7())
    # Simple arithmetic grammar in Lark format
    lark_grammar = """
start: expr

expr: term ((ADD | SUB) term)*
term: factor ((MUL | DIV) factor)*
factor: NUMBER
      | "(" expr ")"

ADD: "+"
SUB: "-"
MUL: "*"
DIV: "/"

NUMBER: /\\d+(\\.\\d+)?/

%import common.WS
%ignore WS
"""
    messages = [
        {"role": "user", "content": "Use the calculator tool to compute 5 + 3 * 2"},
    ]
    tools = [
        {
            "type": "custom",
            "custom": {
                "name": "calculator",
                "description": "Evaluates arithmetic expressions",
                "format": {
                    "type": "grammar",
                    "grammar": {"syntax": "lark", "definition": lark_grammar},
                },
            },
        }
    ]
    result = await async_openai_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": episode_id},
        messages=messages,
        model="tensorzero::model_name::openai::responses::gpt-5-codex",
        tools=tools,
    )
    assert result.model == "tensorzero::model_name::openai::responses::gpt-5-codex"
    assert result.episode_id == episode_id
    # Check that we got tool calls in the response
    assert result.choices[0].message.tool_calls is not None
    assert len(result.choices[0].message.tool_calls) >= 1
    # Find the calculator tool call
    calculator_calls = [tc for tc in result.choices[0].message.tool_calls if tc.function.name == "calculator"]
    assert len(calculator_calls) == 1
    tool_call = calculator_calls[0]
    assert tool_call.type == "function"
    assert tool_call.function.name == "calculator"
    assert tool_call.function.arguments is not None
    assert len(tool_call.function.arguments) > 0
