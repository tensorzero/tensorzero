"""
Tests for the TensorZero client

We use pytest and pytest-asyncio to run the tests.

These tests cover the major functionality of the client but do not
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

import base64
import inspect
import json
import os
import threading
import time
import typing as t
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from os import path
from uuid import UUID

import pytest
import pytest_asyncio
import tensorzero
from openai import AsyncOpenAI, OpenAI
from pytest import FixtureRequest
from tensorzero import (
    AsyncTensorZeroGateway,
    ChatDatapointInsert,
    ChatInferenceDatapointInput,
    ChatInferenceResponse,
    DynamicEvaluationRunResponse,
    FeedbackResponse,
    FileBase64,
    FileUrl,
    FinishReason,
    ImageBase64,
    ImageUrl,
    InferenceChunk,
    JsonDatapointInsert,
    JsonInferenceDatapointInput,
    JsonInferenceResponse,
    RawText,
    TensorZeroError,
    TensorZeroGateway,
    TensorZeroInternalError,
    Text,
    TextChunk,
    ThoughtChunk,
    ToolCall,
    ToolResult,
)
from tensorzero.types import (
    ChatChunk,
    ChatDatapoint,
    JsonChunk,
    JsonDatapoint,
    ProviderExtraBody,
    Thought,
    ToolCallChunk,
    VariantExtraBody,
)
from uuid_utils import uuid7

TEST_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../tensorzero-internal/tests/e2e/tensorzero.toml",
)


class ClientType(Enum):
    HttpGateway = 0
    EmbeddedGateway = 1


# TODO - get type checking working with this decorator
@pytest_asyncio.fixture(params=[ClientType.HttpGateway, ClientType.EmbeddedGateway])  # type: ignore
async def async_client(request: FixtureRequest):
    if request.param == ClientType.HttpGateway:
        client_fut = AsyncTensorZeroGateway.build_http(
            gateway_url="http://localhost:3000"
        )
        assert inspect.isawaitable(client_fut)
        async with await client_fut as client:
            yield client
    else:
        client_fut = AsyncTensorZeroGateway.build_embedded(
            config_file=TEST_CONFIG_FILE,
            clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero-python-e2e",
        )
        assert inspect.isawaitable(client_fut)
        async with await client_fut as client:
            yield client


def test_sync_embedded_gateway_no_config():
    with pytest.warns(UserWarning, match="No config file provided"):
        client = TensorZeroGateway.build_embedded()
    with pytest.raises(TensorZeroError) as exc_info:
        client.inference(function_name="my_missing_func", input={})

    assert exc_info.value.status_code == 404
    assert exc_info.value.text == '{"error":"Unknown function: my_missing_func"}'


@pytest.mark.asyncio
async def test_async_embedded_gateway_no_config():
    with pytest.warns(UserWarning, match="No config file provided"):
        client_fut = AsyncTensorZeroGateway.build_embedded()
        assert inspect.isawaitable(client_fut)
        client = await client_fut
    with pytest.raises(TensorZeroError) as exc_info:
        await client.inference(function_name="my_missing_func", input={})

    assert exc_info.value.status_code == 404
    assert exc_info.value.text == '{"error":"Unknown function: my_missing_func"}'


@dataclass
class CountData:
    count: int


@pytest.mark.asyncio
async def test_async_gil_unlock(async_client: AsyncTensorZeroGateway):
    input = {
        "system": {"assistant_name": "Alfred Pennyworth"},
        "messages": [{"role": "user", "content": [Text(type="text", text="Hello")]}],
    }

    count_data = CountData(count=0)

    # pass 'count_data' to the thread
    def incr_count(count_data: CountData):
        while True:
            count_data.count += 1
            time.sleep(0.1)

    thread = threading.Thread(target=incr_count, args=(count_data,), daemon=True)

    start = time.time()
    thread.start()
    await async_client.inference(
        function_name="basic_test",
        input=input,
        variant_name="slow",
        tags={"key": "value"},
    )
    val = count_data.count
    end = time.time()

    # The special 'slow' variant should take at least 5 seconds to run
    assert end - start >= 5
    # Verify that our thread was still looping during this time
    assert val >= 20


def test_sync_gil_unlock(sync_client: TensorZeroGateway):
    input = {
        "system": {"assistant_name": "Alfred Pennyworth"},
        "messages": [{"role": "user", "content": [Text(type="text", text="Hello")]}],
    }

    count_data = CountData(count=0)

    # pass 'count_data' to the thread
    def incr_count(count_data: CountData):
        while True:
            count_data.count += 1
            time.sleep(0.1)

    thread = threading.Thread(target=incr_count, args=(count_data,), daemon=True)

    start = time.time()
    thread.start()
    sync_client.inference(
        function_name="basic_test",
        input=input,
        variant_name="slow",
        tags={"key": "value"},
    )
    val = count_data.count
    end = time.time()

    # The special 'slow' variant should take at least 5 seconds to run
    assert end - start >= 5
    # Verify that our thread was still looping during this time
    assert val >= 20


@pytest.mark.asyncio
async def test_async_basic_inference(async_client: AsyncTensorZeroGateway):
    input = {
        "system": {"assistant_name": "Alfred Pennyworth"},
        "messages": [{"role": "user", "content": [Text(type="text", text="Hello")]}],
    }
    input_copy = deepcopy(input)
    result = await async_client.inference(
        function_name="basic_test",
        input=input,
        episode_id=uuid7(),  # This would not typically be done but this partially verifies that uuid7 is using a correct implementation
        # because the gateway validates some of the properties needed
    )
    assert isinstance(result, ChatInferenceResponse)
    assert input == input_copy, "Input should not be modified by the client"
    assert result.variant_name == "test"
    assert isinstance(result, ChatInferenceResponse)
    content = result.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert isinstance(content[0], Text)
    assert (
        content[0].text
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    usage = result.usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 10
    assert result.finish_reason == FinishReason.STOP
    time.sleep(1)

    # Test caching
    result = await async_client.inference(
        function_name="basic_test",
        input=input,
        cache_options={"max_age_s": 10, "enabled": "on"},
    )
    assert isinstance(result, ChatInferenceResponse)
    assert result.variant_name == "test"
    content = result.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert isinstance(content[0], Text)
    assert (
        content[0].text
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    usage = result.usage
    assert usage.input_tokens == 0  # should be cached
    assert usage.output_tokens == 0  # should be cached
    assert result.finish_reason == FinishReason.STOP


@pytest.mark.asyncio
async def test_async_client_build_http_sync():
    client_ = AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000",
        async_setup=False,
    )
    assert isinstance(client_, AsyncTensorZeroGateway)
    async with client_ as client:
        input = {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [
                {"role": "user", "content": [Text(type="text", text="Hello")]}
            ],
        }
        input_copy = deepcopy(input)
        result = await client.inference(
            function_name="basic_test",
            input=input,
            episode_id=uuid7(),  # This would not typically be done but this partially verifies that uuid7 is using a correct implementation
            # because the gateway validates some of the properties needed
        )
        assert isinstance(result, ChatInferenceResponse)
        assert input == input_copy, "Input should not be modified by the client"
        assert result.variant_name == "test"
        assert isinstance(result, ChatInferenceResponse)
        content = result.content
        assert len(content) == 1
        assert content[0].type == "text"
        assert isinstance(content[0], Text)
        assert (
            content[0].text
            == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
        )
        usage = result.usage
        assert usage.input_tokens == 10
        assert usage.output_tokens == 10
        assert result.finish_reason == FinishReason.STOP


@pytest.mark.asyncio
async def test_async_client_build_embedded_sync():
    client_ = AsyncTensorZeroGateway.build_embedded(
        config_file=TEST_CONFIG_FILE,
        clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero-python-e2e",
        async_setup=False,
    )
    assert isinstance(client_, AsyncTensorZeroGateway)
    async with client_ as client:
        input = {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [
                {"role": "user", "content": [Text(type="text", text="Hello")]}
            ],
        }
        input_copy = deepcopy(input)
        result = await client.inference(
            function_name="basic_test",
            input=input,
            episode_id=uuid7(),  # This would not typically be done but this partially verifies that uuid7 is using a correct implementation
            # because the gateway validates some of the properties needed
        )
        assert isinstance(result, ChatInferenceResponse)
        assert input == input_copy, "Input should not be modified by the client"
        assert result.variant_name == "test"
        assert isinstance(result, ChatInferenceResponse)
        content = result.content
        assert len(content) == 1
        assert content[0].type == "text"
        assert isinstance(content[0], Text)
        assert (
            content[0].text
            == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
        )
        usage = result.usage
        assert usage.input_tokens == 10
        assert usage.output_tokens == 10
        assert result.finish_reason == FinishReason.STOP


@pytest.mark.asyncio
async def test_async_reasoning_inference(async_client: AsyncTensorZeroGateway):
    result = await async_client.inference(
        function_name="basic_test",
        variant_name="reasoner",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
        tags={"key": "value"},
    )
    assert isinstance(result, ChatInferenceResponse)
    assert result.variant_name == "reasoner"
    assert result.original_response is None
    content = result.content
    assert len(content) == 2
    assert isinstance(content[0], Thought)
    assert content[0].type == "thought"
    assert content[0].text == "hmmm"
    assert isinstance(content[1], Text)
    assert content[1].type == "text"
    assert (
        content[1].text
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    usage = result.usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 10


@pytest.mark.asyncio
async def test_async_default_function_inference(async_client: AsyncTensorZeroGateway):
    input = {
        "system": "You are a helpful assistant named Alfred Pennyworth.",
        "messages": [{"role": "user", "content": [RawText(value="Hello")]}],
    }
    input_copy = deepcopy(input)
    result = await async_client.inference(
        model_name="dummy::test",
        input=input,
        episode_id=uuid7(),  # This would not typically be done but this partially verifies that uuid7 is using a correct implementation
        # because the gateway validates some of the properties needed
        tags={"key": "value"},
    )
    assert isinstance(result, ChatInferenceResponse)
    assert input == input_copy, "Input should not be modified by the client"
    assert result.variant_name == "dummy::test"
    assert isinstance(result, ChatInferenceResponse)
    content = result.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert isinstance(content[0], Text)
    assert (
        content[0].text
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    usage = result.usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 10


@pytest.mark.asyncio
async def test_async_default_function_inference_plain_dict(
    async_client: AsyncTensorZeroGateway,
):
    input = {
        "system": "You are a helpful assistant named Alfred Pennyworth.",
        "messages": [
            {"role": "user", "content": [{"type": "raw_text", "value": "Hello"}]}
        ],
    }
    input_copy = deepcopy(input)
    result = await async_client.inference(
        model_name="dummy::test",
        input=input,
        episode_id=uuid7(),  # This would not typically be done but this partially verifies that uuid7 is using a correct implementation
        # because the gateway validates some of the properties needed
        tags={"key": "value"},
    )
    assert isinstance(result, ChatInferenceResponse)
    assert input == input_copy, "Input should not be modified by the client"
    assert result.variant_name == "dummy::test"
    content = result.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert isinstance(content[0], Text)
    assert (
        content[0].text
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    usage = result.usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 10


@pytest.mark.asyncio
async def test_async_inference_streaming(async_client: AsyncTensorZeroGateway):
    stream = await async_client.inference(
        function_name="basic_test",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
        tags={"key": "value"},
        stream=True,
    )
    assert isinstance(stream, t.AsyncIterator)

    chunks: t.List[InferenceChunk] = []
    previous_chunk_timestamp = None
    last_chunk_duration = -1
    async for chunk in stream:
        if previous_chunk_timestamp is not None:
            last_chunk_duration = time.time() - previous_chunk_timestamp
        previous_chunk_timestamp = time.time()
        chunks.append(chunk)

    assert last_chunk_duration > 0.0

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
            assert chunk.inference_id == previous_inference_id
        if previous_episode_id is not None:
            assert chunk.episode_id == previous_episode_id
        previous_inference_id = chunk.inference_id
        previous_episode_id = chunk.episode_id
        variant_name = chunk.variant_name
        assert variant_name == "test"
        assert isinstance(chunk, ChatChunk)
        if i + 1 < len(chunks):
            assert len(chunk.content) == 1
            assert isinstance(chunk.content[0], TextChunk)
            assert chunk.content[0].type == "text"
            assert chunk.content[0].text == expected_text[i]
        else:
            assert len(chunk.content) == 0
            assert chunk.usage is not None
            assert chunk.usage.input_tokens == 10
            assert chunk.usage.output_tokens == 16
            assert chunk.finish_reason == FinishReason.STOP


@pytest.mark.asyncio
async def test_async_reasoning_inference_streaming(
    async_client: AsyncTensorZeroGateway,
):
    stream = await async_client.inference(
        function_name="basic_test",
        variant_name="reasoner",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
        tags={"key": "value"},
        stream=True,
    )
    assert isinstance(stream, t.AsyncIterator)

    chunks: t.List[InferenceChunk] = []
    previous_chunk_timestamp = None
    last_chunk_duration = -1
    async for chunk in stream:
        if previous_chunk_timestamp is not None:
            last_chunk_duration = time.time() - previous_chunk_timestamp
        previous_chunk_timestamp = time.time()
        chunks.append(chunk)

    assert last_chunk_duration > 0
    expected_thinking = [
        "hmmm",
        "hmmm",
    ]
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
            assert chunk.inference_id == previous_inference_id
        if previous_episode_id is not None:
            assert chunk.episode_id == previous_episode_id
        previous_inference_id = chunk.inference_id
        previous_episode_id = chunk.episode_id
        variant_name = chunk.variant_name
        assert variant_name == "reasoner"
        assert isinstance(chunk, ChatChunk)
        if i < len(expected_thinking):
            assert len(chunk.content) == 1
            assert chunk.content[0].type == "thought"
            assert isinstance(chunk.content[0], ThoughtChunk)
            assert chunk.content[0].text == expected_thinking[i]
        elif i < len(expected_thinking) + len(expected_text):
            assert len(chunk.content) == 1
            assert chunk.content[0].type == "text"
            assert isinstance(chunk.content[0], TextChunk)
            assert chunk.content[0].text == expected_text[i - len(expected_thinking)]
        else:
            assert len(chunk.content) == 0
            assert chunk.usage is not None
            assert chunk.usage.input_tokens == 10
            assert chunk.usage.output_tokens == 10
            assert chunk.finish_reason == FinishReason.STOP


@pytest.mark.asyncio
async def test_async_inference_streaming_nonexistent_function(
    async_client: AsyncTensorZeroGateway,
):
    with pytest.raises(TensorZeroError) as exc_info:
        stream = await async_client.inference(
            function_name="does_not_exist",
            input={
                "system": {"assistant_name": "Alfred Pennyworth"},
                "messages": [{"role": "user", "content": "Hello"}],
            },
            stream=True,
        )
        assert isinstance(stream, t.AsyncIterator)

        # The httpx client won't make a request until you start consuming the stream
        async for _chunk in stream:
            pass

    assert exc_info.value.status_code == 404
    assert (
        str(exc_info.value)
        == 'TensorZeroError (status code 404): {"error":"Unknown function: does_not_exist"}'
    )


@pytest.mark.asyncio
async def test_async_inference_streaming_malformed_input(
    async_client: AsyncTensorZeroGateway,
):
    with pytest.raises(TensorZeroError) as exc_info:
        stream = await async_client.inference(
            function_name="basic_test",
            input={
                "system": {"name_of_assistant": "Alfred Pennyworth"},  # WRONG
                "messages": [{"role": "user", "content": "Hello"}],
            },
            stream=True,
        )
        assert isinstance(stream, t.AsyncIterator)

        # The httpx client won't make a request until you start consuming the stream
        async for _chunk in stream:
            pass

    assert exc_info.value.status_code == 400
    assert "JSON Schema validation failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_tool_call_inference(async_client: AsyncTensorZeroGateway):
    result = await async_client.inference(
        function_name="weather_helper",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hi I'm visiting Brooklyn from Brazil. What's the weather?",
                }
            ],
        },
    )
    assert isinstance(result, ChatInferenceResponse)
    assert result.variant_name == "variant"
    assert isinstance(result, ChatInferenceResponse)
    content = result.content
    assert len(content) == 1
    assert content[0].type == "tool_call"
    assert isinstance(content[0], ToolCall)
    assert content[0].raw_name == "get_temperature"
    assert content[0].id == "0"
    assert content[0].raw_arguments == '{"location":"Brooklyn","units":"celsius"}'
    assert content[0].name == "get_temperature"
    assert content[0].arguments == {"location": "Brooklyn", "units": "celsius"}
    usage = result.usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 10
    assert result.finish_reason == FinishReason.TOOL_CALL


@pytest.mark.asyncio
async def test_async_malformed_tool_call_inference(
    async_client: AsyncTensorZeroGateway,
):
    result = await async_client.inference(
        function_name="weather_helper",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hi I'm visiting Brooklyn from Brazil. What's the weather?",
                }
            ],
        },
        variant_name="bad_tool",
    )
    assert isinstance(result, ChatInferenceResponse)
    assert result.variant_name == "bad_tool"
    assert isinstance(result, ChatInferenceResponse)
    content = result.content
    assert len(content) == 1
    assert isinstance(content[0], ToolCall)
    assert content[0].type == "tool_call"
    assert content[0].raw_name == "get_temperature"
    assert content[0].id == "0"
    assert content[0].raw_arguments == '{"location":"Brooklyn","units":"Celsius"}'
    assert content[0].name == "get_temperature"
    assert content[0].arguments is None
    usage = result.usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 10


@pytest.mark.asyncio
async def test_async_tool_call_streaming(async_client: AsyncTensorZeroGateway):
    stream = await async_client.inference(
        function_name="weather_helper",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hi I'm visiting Brooklyn from Brazil. What's the weather?",
                }
            ],
        },
        stream=True,
    )
    assert isinstance(stream, t.AsyncIterator)
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
            assert chunk.inference_id == previous_inference_id
        if previous_episode_id is not None:
            assert chunk.episode_id == previous_episode_id
        previous_inference_id = chunk.inference_id
        previous_episode_id = chunk.episode_id
        variant_name = chunk.variant_name
        assert variant_name == "variant"
        assert isinstance(chunk, ChatChunk)
        if i + 1 < len(chunks):
            assert len(chunk.content) == 1
            assert isinstance(chunk.content[0], ToolCallChunk)
            assert chunk.content[0].type == "tool_call"
            assert chunk.content[0].raw_name == "get_temperature"
            assert chunk.content[0].id == "0"
            assert chunk.content[0].raw_arguments == expected_text[i]
        else:
            assert len(chunk.content) == 0
            assert chunk.usage is not None
            assert chunk.usage.input_tokens == 10
            assert chunk.usage.output_tokens == 5
            assert chunk.finish_reason == FinishReason.TOOL_CALL


@pytest.mark.asyncio
async def test_async_json_streaming(async_client: AsyncTensorZeroGateway):
    # We don't actually have a streaming JSON function implemented in `dummy.rs` but it doesn't matter for this test since
    # TensorZero doesn't parse the JSON output of the function for streaming calls.
    stream = await async_client.inference(
        function_name="json_success",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [
                {"role": "user", "content": {"country": "Japan"}},
                {"role": "assistant", "content": "ok"},
                # This function has a user schema but we can bypass with RawText
                {"role": "user", "content": [RawText(value="Hello")]},
            ],
        },
        stream=True,
    )
    assert isinstance(stream, t.AsyncIterator)
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
            assert chunk.inference_id == previous_inference_id
        if previous_episode_id is not None:
            assert chunk.episode_id == previous_episode_id
        previous_inference_id = chunk.inference_id
        previous_episode_id = chunk.episode_id
        variant_name = chunk.variant_name
        assert variant_name == "test"
        assert isinstance(chunk, JsonChunk)
        if i + 1 < len(chunks):
            assert chunk.raw == expected_text[i]
        else:
            assert chunk.usage is not None
            assert chunk.usage.input_tokens == 10
            assert chunk.usage.output_tokens == 16


@pytest.mark.asyncio
async def test_async_json_streaming_reasoning(async_client: AsyncTensorZeroGateway):
    stream = await async_client.inference(
        function_name="json_success",
        variant_name="json_reasoner",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "arguments": {"country": "Japan"}}],
                }
            ],
        },
        stream=True,
    )
    assert isinstance(stream, t.AsyncIterator)
    chunks = [chunk async for chunk in stream]
    expected_text = [
        '{"name"',
        ':"John"',
        ',"age"',
        ":30",
        "}",
    ]
    previous_inference_id = None
    previous_episode_id = None
    for i, chunk in enumerate(chunks):
        if previous_inference_id is not None:
            assert chunk.inference_id == previous_inference_id
        if previous_episode_id is not None:
            assert chunk.episode_id == previous_episode_id
        previous_inference_id = chunk.inference_id
        previous_episode_id = chunk.episode_id
        variant_name = chunk.variant_name
        assert variant_name == "json_reasoner"
        assert isinstance(chunk, JsonChunk)
        if i < len(expected_text):
            assert chunk.raw == expected_text[i]
        else:
            assert chunk.raw == ""
            assert chunk.usage is not None
            assert chunk.usage.input_tokens == 10
            assert chunk.usage.output_tokens == 10


@pytest.mark.asyncio
async def test_async_json_success(async_client: AsyncTensorZeroGateway):
    result = await async_client.inference(
        function_name="json_success",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "arguments": {"country": "Japan"}}],
                }
            ],
        },
        output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
        stream=False,
    )
    assert isinstance(result, JsonInferenceResponse)
    assert result.variant_name == "test"
    assert result.output.raw == '{"answer":"Hello"}'
    assert result.output.parsed == {"answer": "Hello"}
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 10


@pytest.mark.asyncio
async def test_async_json_reasoning(async_client: AsyncTensorZeroGateway):
    result = await async_client.inference(
        function_name="json_success",
        variant_name="json_reasoner",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "arguments": {"country": "Japan"}}],
                }
            ],
        },
        stream=False,
    )
    assert isinstance(result, JsonInferenceResponse)
    assert result.variant_name == "json_reasoner"
    assert result.output.raw == '{"answer":"Hello"}'
    assert result.output.parsed == {"answer": "Hello"}
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 10


@pytest.mark.asyncio
async def test_async_json_failure(async_client: AsyncTensorZeroGateway):
    result = await async_client.inference(
        function_name="json_fail",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello, world!"}],
        },
        stream=False,
    )
    assert isinstance(result, JsonInferenceResponse)
    assert result.variant_name == "test"
    assert (
        result.output.raw
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    assert result.output.parsed is None
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 10


@pytest.mark.asyncio
async def test_async_feedback(async_client: AsyncTensorZeroGateway):
    # Run inference to get a valid inference id and episode id.
    result = await async_client.inference(
        function_name="basic_test",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert isinstance(result, ChatInferenceResponse)
    inference_id = result.inference_id
    episode_id = result.episode_id

    result = await async_client.feedback(
        metric_name="user_rating", value=5, episode_id=episode_id
    )
    assert isinstance(result, FeedbackResponse)

    result = await async_client.feedback(
        metric_name="task_success", value=True, inference_id=inference_id
    )
    assert isinstance(result, FeedbackResponse)

    result = await async_client.feedback(
        metric_name="demonstration",
        value="hi how are you",
        inference_id=inference_id,
        tags={"author": "Alice"},
    )
    assert isinstance(result, FeedbackResponse)


@pytest.mark.asyncio
async def test_async_feedback_invalid_input(async_client: AsyncTensorZeroGateway):
    with pytest.raises(TensorZeroError):
        await async_client.feedback(metric_name="test_metric", value=5)

    with pytest.raises(TensorZeroError):
        await async_client.feedback(
            metric_name="test_metric",
            value=5,
            episode_id=UUID("123e4567-e89b-12d3-a456-426614174000"),
            inference_id=UUID("123e4567-e89b-12d3-a456-426614174001"),
        )


@pytest.mark.asyncio
async def test_async_tensorzero_error(async_client: AsyncTensorZeroGateway):
    with pytest.raises(TensorZeroError) as excinfo:
        await async_client.inference(
            function_name="not_a_function", input={"messages": []}
        )

    assert (
        str(excinfo.value)
        == 'TensorZeroError (status code 404): {"error":"Unknown function: not_a_function"}'
    )


@pytest.mark.asyncio
async def test_async_dynamic_credentials(async_client: AsyncTensorZeroGateway):
    result = await async_client.inference(
        function_name="basic_test",
        variant_name="test_dynamic_api_key",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
        credentials={"DUMMY_API_KEY": "good_key"},
    )
    assert isinstance(result, ChatInferenceResponse)
    assert result.variant_name == "test_dynamic_api_key"
    assert isinstance(result, ChatInferenceResponse)
    content = result.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert isinstance(content[0], Text)
    assert (
        content[0].text
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    usage = result.usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 10


def test_sync_error():
    with pytest.raises(Exception) as exc_info:
        with TensorZeroGateway.build_http(gateway_url="http://localhost:3000"):
            raise Exception("My error")
    assert str(exc_info.value) == "My error"


@pytest.mark.asyncio
async def test_async_error():
    with pytest.raises(Exception) as exc_info:
        client_fut = AsyncTensorZeroGateway.build_http(
            gateway_url="http://localhost:3000"
        )
        assert isinstance(client_fut, t.Awaitable)
        async with await client_fut:
            raise Exception("My error")
    assert str(exc_info.value) == "My error"


@pytest.fixture(params=[ClientType.HttpGateway, ClientType.EmbeddedGateway])
def sync_client(request: FixtureRequest):
    if request.param == ClientType.HttpGateway:
        with TensorZeroGateway.build_http(
            gateway_url="http://localhost:3000"
        ) as client:
            yield client
    else:
        with TensorZeroGateway.build_embedded(
            config_file=TEST_CONFIG_FILE,
            clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero-python-e2e",
        ) as client:
            yield client


def test_sync_inference_caching(sync_client: TensorZeroGateway):
    result = sync_client.inference(
        function_name="basic_test",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
        tags={"key": "value"},
    )
    assert isinstance(result, ChatInferenceResponse)
    assert result.variant_name == "test"
    content = result.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert isinstance(content[0], Text)
    assert (
        content[0].text
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    usage = result.usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 10

    # Test caching
    result = sync_client.inference(
        function_name="basic_test",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
        cache_options={"max_age_s": 10, "enabled": "on"},
        tags={"key": "value"},
    )
    assert isinstance(result, ChatInferenceResponse)
    assert result.variant_name == "test"
    content = result.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert isinstance(content[0], Text)
    assert (
        content[0].text
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    usage = result.usage
    assert usage.input_tokens == 0  # should be cached
    assert usage.output_tokens == 0  # should be cached


def test_sync_inference_streaming_caching(sync_client: TensorZeroGateway):
    stream = sync_client.inference(
        function_name="basic_test",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
        stream=True,
    )
    assert isinstance(stream, t.Iterator)

    chunks: t.List[ChatChunk] = []
    for chunk in stream:
        assert isinstance(chunk, ChatChunk)
        chunks.append(chunk)

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

    for i, chunk in enumerate(chunks[:-1]):
        assert len(chunk.content) == 1
        assert chunk.content[0].type == "text"
        assert isinstance(chunk.content[0], TextChunk)
        assert chunk.content[0].text == expected_text[i]

    final_chunk = chunks[-1]
    assert len(final_chunk.content) == 0
    assert final_chunk.usage is not None
    assert final_chunk.usage.input_tokens == 10
    assert final_chunk.usage.output_tokens == 16

    # Test caching
    stream = sync_client.inference(
        function_name="basic_test",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
        stream=True,
        cache_options={"max_age_s": 10, "enabled": "on"},
    )
    assert isinstance(stream, t.Iterator)

    chunks: t.List[ChatChunk] = []
    for chunk in stream:
        assert isinstance(chunk, ChatChunk)
        chunks.append(chunk)

    for i, chunk in enumerate(chunks[:-1]):
        assert len(chunk.content) == 1
        assert chunk.content[0].type == "text"
        assert isinstance(chunk.content[0], TextChunk)
        assert chunk.content[0].text == expected_text[i]

    final_chunk = chunks[-1]
    assert len(final_chunk.content) == 0
    if final_chunk.usage is not None:
        assert final_chunk.usage.input_tokens == 0  # should be cached
        assert final_chunk.usage.output_tokens == 0  # should be cached


def test_default_function_inference(sync_client: TensorZeroGateway):
    input = {
        "system": "You are a helpful assistant named Alfred Pennyworth.",
        "messages": [{"role": "user", "content": [Text(type="text", text="Hello")]}],
    }
    input_copy = deepcopy(input)
    result = sync_client.inference(
        model_name="dummy::test",
        input=input,
        episode_id=uuid7(),  # This would not typically be done but this partially verifies that uuid7 is using a correct implementation
        # because the gateway validates some of the properties needed
        tags={"key": "value"},
    )
    assert isinstance(result, ChatInferenceResponse)
    assert input == input_copy, "Input should not be modified by the client"
    assert result.variant_name == "dummy::test"
    assert isinstance(result, ChatInferenceResponse)
    content = result.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert isinstance(content[0], Text)
    assert (
        content[0].text
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    usage = result.usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 10


def test_image_inference_base64(sync_client: TensorZeroGateway):
    basepath = path.dirname(__file__)
    with open(
        f"{basepath}/../../../tensorzero-internal/tests/e2e/providers/ferris.png", "rb"
    ) as f:
        ferris_png = base64.b64encode(f.read()).decode("ascii")

    input = {
        "system": "You are a helpful assistant named Alfred Pennyworth.",
        "messages": [
            {
                "role": "user",
                "content": [
                    ImageBase64(
                        data=ferris_png,
                        mime_type="image/png",
                    )
                ],
            }
        ],
    }
    input_copy = deepcopy(input)
    result = sync_client.inference(
        model_name="dummy::extract_images",
        input=input,
        episode_id=uuid7(),  # This would not typically be done but this partially verifies that uuid7 is using a correct implementation
        # because the gateway validates some of the properties needed
    )
    assert isinstance(result, ChatInferenceResponse)
    assert input == input_copy, "Input should not be modified by the client"
    assert result.variant_name == "dummy::extract_images"
    content = result.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert isinstance(content[0], Text)
    assert content[0].text is not None
    json_content = json.loads(content[0].text)
    assert json_content == [
        {
            "file": {"url": None, "mime_type": "image/png"},
            "storage_path": {
                "kind": {"type": "disabled"},
                "path": "observability/files/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png",
            },
        }
    ]


def test_file_inference_base64(sync_client: TensorZeroGateway):
    # Test image with File block
    basepath = path.dirname(__file__)
    with open(
        f"{basepath}/../../../tensorzero-internal/tests/e2e/providers/ferris.png", "rb"
    ) as f:
        ferris_png = base64.b64encode(f.read()).decode("ascii")

    input = {
        "system": "You are a helpful assistant named Alfred Pennyworth.",
        "messages": [
            {
                "role": "user",
                "content": [
                    FileBase64(
                        data=ferris_png,
                        mime_type="image/png",
                    )
                ],
            }
        ],
    }
    input_copy = deepcopy(input)
    result = sync_client.inference(
        model_name="dummy::extract_images",
        input=input,
        episode_id=uuid7(),  # This would not typically be done but this partially verifies that uuid7 is using a correct implementation
        # because the gateway validates some of the properties needed
    )
    assert isinstance(result, ChatInferenceResponse)
    assert input == input_copy, "Input should not be modified by the client"
    assert result.variant_name == "dummy::extract_images"
    content = result.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert isinstance(content[0], Text)
    assert content[0].text is not None
    json_content = json.loads(content[0].text)
    assert json_content == [
        {
            "file": {"url": None, "mime_type": "image/png"},
            "storage_path": {
                "kind": {"type": "disabled"},
                "path": "observability/files/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png",
            },
        }
    ]
    # Test pdf with File block
    basepath = path.dirname(__file__)
    with open(
        f"{basepath}/../../../tensorzero-internal/tests/e2e/providers/deepseek_paper.pdf",
        "rb",
    ) as f:
        deepseek_paper_pdf = base64.b64encode(f.read()).decode("ascii")

    input = {
        "system": "You are a helpful assistant named Alfred Pennyworth.",
        "messages": [
            {
                "role": "user",
                "content": [
                    FileBase64(
                        data=deepseek_paper_pdf,
                        mime_type="application/pdf",
                    )
                ],
            }
        ],
    }
    input_copy = deepcopy(input)
    result = sync_client.inference(
        model_name="dummy::require_pdf",
        input=input,
        episode_id=uuid7(),
    )
    assert isinstance(result, ChatInferenceResponse)
    assert input == input_copy, "Input should not be modified by the client"
    assert result.variant_name == "dummy::require_pdf"
    content = result.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert isinstance(content[0], Text)
    assert content[0].text is not None
    print(content[0].text)
    json_content = json.loads(content[0].text)
    assert json_content == [
        {
            "file": {"url": None, "mime_type": "application/pdf"},
            "storage_path": {
                "kind": {"type": "disabled"},
                "path": "observability/files/3e127d9a726f6be0fd81d73ccea97d96ec99419f59650e01d49183cd3be999ef.pdf",
            },
        }
    ]


def test_image_inference_url(sync_client: TensorZeroGateway):
    input = {
        "system": "You are a helpful assistant named Alfred Pennyworth.",
        "messages": [
            {
                "role": "user",
                "content": [
                    ImageUrl(
                        url="https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png"
                    )
                ],
            }
        ],
    }
    input_copy = deepcopy(input)
    result = sync_client.inference(
        model_name="dummy::extract_images",
        input=input,
        episode_id=uuid7(),  # This would not typically be done but this partially verifies that uuid7 is using a correct implementation
        # because the gateway validates some of the properties needed
    )
    assert isinstance(result, ChatInferenceResponse)
    assert input == input_copy, "Input should not be modified by the client"
    assert result.variant_name == "dummy::extract_images"
    assert isinstance(result, ChatInferenceResponse)
    content = result.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert isinstance(content[0], Text)
    assert content[0].text is not None
    json_content = json.loads(content[0].text)
    assert json_content == [
        {
            "file": {
                "url": "https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png",
                "mime_type": "image/png",
            },
            "storage_path": {
                "kind": {"type": "disabled"},
                "path": "observability/files/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png",
            },
        }
    ]


def test_file_inference_url(sync_client: TensorZeroGateway):
    input = {
        "system": "You are a helpful assistant named Alfred Pennyworth.",
        "messages": [
            {
                "role": "user",
                "content": [
                    FileUrl(
                        url="https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png"
                    )
                ],
            }
        ],
    }
    input_copy = deepcopy(input)
    result = sync_client.inference(
        model_name="dummy::extract_images",
        input=input,
        episode_id=uuid7(),  # This would not typically be done but this partially verifies that uuid7 is using a correct implementation
        # because the gateway validates some of the properties needed
    )
    assert isinstance(result, ChatInferenceResponse)
    assert input == input_copy, "Input should not be modified by the client"
    assert result.variant_name == "dummy::extract_images"
    content = result.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert isinstance(content[0], Text)
    assert content[0].text is not None
    json_content = json.loads(content[0].text)
    print(json_content)
    assert json_content == [
        {
            "file": {
                "url": "https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png",
                "mime_type": "image/png",
            },
            "storage_path": {
                "kind": {"type": "disabled"},
                "path": "observability/files/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png",
            },
        }
    ]


def test_sync_malformed_inference(sync_client: TensorZeroGateway):
    with pytest.raises(TensorZeroError) as exc_info:
        sync_client.inference(
            function_name="basic_test",
            input={
                "system": {"name_of_assistant": "Alfred Pennyworth"},  # WRONG
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
    assert exc_info.value.status_code == 400


def test_sync_inference_streaming(sync_client: TensorZeroGateway):
    stream = sync_client.inference(
        function_name="basic_test",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
        stream=True,
        tags={"key": "value"},
    )
    assert isinstance(stream, t.Iterator)

    chunks: t.List[InferenceChunk] = []
    previous_chunk_timestamp = None
    last_chunk_duration = -1
    for chunk in stream:
        if previous_chunk_timestamp is not None:
            last_chunk_duration = time.time() - previous_chunk_timestamp
        previous_chunk_timestamp = time.time()
        chunks.append(chunk)

    assert last_chunk_duration > 0.0

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
            assert chunk.inference_id == previous_inference_id
        if previous_episode_id is not None:
            assert chunk.episode_id == previous_episode_id
        previous_inference_id = chunk.inference_id
        previous_episode_id = chunk.episode_id
        variant_name = chunk.variant_name
        assert variant_name == "test"
        assert isinstance(chunk, ChatChunk)
        if i + 1 < len(chunks):
            assert len(chunk.content) == 1
            assert chunk.content[0].type == "text"
            assert isinstance(chunk.content[0], TextChunk)
            assert chunk.content[0].text == expected_text[i]
        else:
            assert len(chunk.content) == 0
            assert chunk.usage is not None
            assert chunk.usage.input_tokens == 10
            assert chunk.usage.output_tokens == 16


def test_sync_inference_streaming_nonexistent_function(sync_client: TensorZeroGateway):
    with pytest.raises(TensorZeroError) as exc_info:
        stream = sync_client.inference(
            function_name="does_not_exist",
            input={
                "system": {"assistant_name": "Alfred Pennyworth"},
                "messages": [{"role": "user", "content": "Hello"}],
            },
            stream=True,
        )
        assert isinstance(stream, t.Iterator)

        # The httpx client won't make a request until you start consuming the stream
        for _chunk in stream:
            pass

    assert exc_info.value.status_code == 404


def test_sync_inference_streaming_malformed_input(sync_client: TensorZeroGateway):
    with pytest.raises(TensorZeroError) as exc_info:
        stream = sync_client.inference(
            function_name="basic_test",
            input={
                "system": {"name_of_assistant": "Alfred Pennyworth"},  # WRONG
                "messages": [{"role": "user", "content": "Hello"}],
            },
            stream=True,
        )
        assert isinstance(stream, t.Iterator)

        # The httpx client won't make a request until you start consuming the stream
        for _chunk in stream:
            pass

    assert exc_info.value.status_code == 400
    assert "JSON Schema validation failed" in str(exc_info.value)


def test_sync_tool_call_inference(sync_client: TensorZeroGateway):
    result = sync_client.inference(
        function_name="weather_helper",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hi I'm visiting Brooklyn from Brazil. What's the weather?",
                }
            ],
        },
    )

    assert isinstance(result, ChatInferenceResponse)
    assert result.variant_name == "variant"
    content = result.content
    assert len(content) == 1
    assert isinstance(content[0], ToolCall)
    assert content[0].type == "tool_call"
    assert content[0].raw_name == "get_temperature"
    assert content[0].id == "0"
    assert content[0].raw_arguments == '{"location":"Brooklyn","units":"celsius"}'
    assert content[0].name == "get_temperature"
    assert content[0].arguments == {"location": "Brooklyn", "units": "celsius"}
    usage = result.usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 10


def test_sync_reasoning_inference(sync_client: TensorZeroGateway):
    result = sync_client.inference(
        function_name="basic_test",
        variant_name="reasoner",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
        tags={"key": "value"},
    )
    assert isinstance(result, ChatInferenceResponse)
    assert result.variant_name == "reasoner"
    assert isinstance(result, ChatInferenceResponse)
    content = result.content
    assert len(content) == 2
    assert isinstance(content[0], Thought)
    assert content[0].type == "thought"
    assert content[0].text == "hmmm"
    assert content[1].type == "text"
    assert isinstance(content[1], Text)
    assert (
        content[1].text
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    usage = result.usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 10


def test_sync_malformed_tool_call_inference(sync_client: TensorZeroGateway):
    result = sync_client.inference(
        function_name="weather_helper",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hi I'm visiting Brooklyn from Brazil. What's the weather?",
                }
            ],
        },
        variant_name="bad_tool",
    )
    assert isinstance(result, ChatInferenceResponse)
    assert result.variant_name == "bad_tool"
    assert isinstance(result, ChatInferenceResponse)
    content = result.content
    assert len(content) == 1
    assert content[0].type == "tool_call"
    assert isinstance(content[0], ToolCall)
    assert content[0].raw_name == "get_temperature"
    assert content[0].id == "0"
    assert content[0].raw_arguments == '{"location":"Brooklyn","units":"Celsius"}'
    assert content[0].name == "get_temperature"
    assert content[0].arguments is None
    usage = result.usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 10


def test_sync_tool_call_streaming(sync_client: TensorZeroGateway):
    stream = sync_client.inference(
        function_name="weather_helper",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hi I'm visiting Brooklyn from Brazil. What's the weather?",
                }
            ],
        },
        stream=True,
    )
    assert isinstance(stream, t.Iterator)
    chunks = list(stream)
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
            assert chunk.inference_id == previous_inference_id
        if previous_episode_id is not None:
            assert chunk.episode_id == previous_episode_id
        previous_inference_id = chunk.inference_id
        previous_episode_id = chunk.episode_id
        variant_name = chunk.variant_name
        assert variant_name == "variant"
        assert isinstance(chunk, ChatChunk)
        if i + 1 < len(chunks):
            assert len(chunk.content) == 1
            assert isinstance(chunk.content[0], ToolCallChunk)
            assert chunk.content[0].type == "tool_call"
            assert chunk.content[0].raw_name == "get_temperature"
            assert chunk.content[0].id == "0"
            assert chunk.content[0].raw_arguments == expected_text[i]
        else:
            assert len(chunk.content) == 0
            assert chunk.usage is not None
            assert chunk.usage.input_tokens == 10
            assert chunk.usage.output_tokens == 5


def test_sync_reasoning_inference_streaming(sync_client: TensorZeroGateway):
    stream = sync_client.inference(
        function_name="basic_test",
        variant_name="reasoner",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
        tags={"key": "value"},
        stream=True,
    )
    assert isinstance(stream, t.Iterator)

    chunks: t.List[InferenceChunk] = []
    previous_chunk_timestamp = None
    last_chunk_duration = -1
    for chunk in stream:
        if previous_chunk_timestamp is not None:
            last_chunk_duration = time.time() - previous_chunk_timestamp
        previous_chunk_timestamp = time.time()
        chunks.append(chunk)

    assert last_chunk_duration > 0
    expected_thinking = [
        "hmmm",
        "hmmm",
    ]
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
            assert chunk.inference_id == previous_inference_id
        if previous_episode_id is not None:
            assert chunk.episode_id == previous_episode_id
        previous_inference_id = chunk.inference_id
        previous_episode_id = chunk.episode_id
        variant_name = chunk.variant_name
        assert variant_name == "reasoner"
        assert isinstance(chunk, ChatChunk)
        if i < len(expected_thinking):
            assert len(chunk.content) == 1
            assert chunk.content[0].type == "thought"
            assert isinstance(chunk.content[0], ThoughtChunk)
            assert chunk.content[0].text == expected_thinking[i]
        elif i < len(expected_thinking) + len(expected_text):
            assert len(chunk.content) == 1
            assert isinstance(chunk.content[0], TextChunk)
            assert chunk.content[0].type == "text"
            assert chunk.content[0].text == expected_text[i - len(expected_thinking)]
        else:
            assert len(chunk.content) == 0
            assert chunk.usage is not None
            assert chunk.usage.input_tokens == 10
            assert chunk.usage.output_tokens == 10


def test_sync_json_streaming(sync_client: TensorZeroGateway):
    # We don't actually have a streaming JSON function implemented in `dummy.rs` but it doesn't matter for this test since
    # TensorZero doesn't parse the JSON output of the function for streaming calls.
    stream = sync_client.inference(
        function_name="json_success",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "arguments": {"country": "Japan"}}],
                }
            ],
        },
        stream=True,
    )
    assert isinstance(stream, t.Iterator)
    chunks = list(stream)
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
            assert chunk.inference_id == previous_inference_id
        if previous_episode_id is not None:
            assert chunk.episode_id == previous_episode_id
        previous_inference_id = chunk.inference_id
        previous_episode_id = chunk.episode_id
        variant_name = chunk.variant_name
        assert variant_name == "test"
        assert isinstance(chunk, JsonChunk)
        if i + 1 < len(chunks):
            assert chunk.raw == expected_text[i]
        else:
            assert chunk.usage is not None
            assert chunk.usage.input_tokens == 10
            assert chunk.usage.output_tokens == 16


def test_sync_json_streaming_reasoning(sync_client: TensorZeroGateway):
    stream = sync_client.inference(
        function_name="json_success",
        variant_name="json_reasoner",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "arguments": {"country": "Japan"}}],
                }
            ],
        },
        stream=True,
    )
    assert isinstance(stream, t.Iterator)
    chunks = list(stream)
    expected_text = [
        '{"name"',
        ':"John"',
        ',"age"',
        ":30",
        "}",
    ]
    previous_inference_id = None
    previous_episode_id = None
    for i, chunk in enumerate(chunks):
        if previous_inference_id is not None:
            assert chunk.inference_id == previous_inference_id
        if previous_episode_id is not None:
            assert chunk.episode_id == previous_episode_id
        previous_inference_id = chunk.inference_id
        previous_episode_id = chunk.episode_id
        variant_name = chunk.variant_name
        assert variant_name == "json_reasoner"
        assert isinstance(chunk, JsonChunk)
        if i < len(expected_text):
            assert chunk.raw == expected_text[i]
        else:
            assert chunk.raw == ""
            assert chunk.usage is not None
            assert chunk.usage.input_tokens == 10
            assert chunk.usage.output_tokens == 10


def test_sync_json_success(sync_client: TensorZeroGateway):
    result = sync_client.inference(
        function_name="json_success",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "arguments": {"country": "Japan"}}],
                }
            ],
        },
        output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
        stream=False,
    )
    assert isinstance(result, JsonInferenceResponse)
    assert result.variant_name == "test"
    assert result.output.raw == '{"answer":"Hello"}'
    assert result.output.parsed == {"answer": "Hello"}
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 10


def test_sync_json_reasoning(sync_client: TensorZeroGateway):
    result = sync_client.inference(
        function_name="json_success",
        variant_name="json_reasoner",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "arguments": {"country": "Japan"}}],
                }
            ],
        },
        stream=False,
    )
    assert isinstance(result, JsonInferenceResponse)
    assert result.variant_name == "json_reasoner"
    assert result.output.raw == '{"answer":"Hello"}'
    assert result.output.parsed == {"answer": "Hello"}
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 10


def test_sync_json_failure(sync_client: TensorZeroGateway):
    result = sync_client.inference(
        function_name="json_fail",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello, world!"}],
        },
        stream=False,
    )
    assert isinstance(result, JsonInferenceResponse)
    assert result.variant_name == "test"
    assert (
        result.output.raw
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    assert result.output.parsed is None
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 10


def test_sync_feedback(sync_client: TensorZeroGateway):
    result = sync_client.inference(
        function_name="basic_test",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert isinstance(result, ChatInferenceResponse)
    inference_id = result.inference_id
    episode_id = result.episode_id

    result = sync_client.feedback(
        metric_name="user_rating", value=5, episode_id=episode_id
    )
    assert isinstance(result, FeedbackResponse)

    result = sync_client.feedback(
        metric_name="task_success", value=True, inference_id=inference_id
    )
    assert isinstance(result, FeedbackResponse)

    result = sync_client.feedback(
        metric_name="demonstration",
        value="hi how are you",
        inference_id=inference_id,
        tags={"author": "Alice"},
    )
    assert isinstance(result, FeedbackResponse)


def test_sync_feedback_invalid_input(sync_client: TensorZeroGateway):
    with pytest.raises(TensorZeroError):
        sync_client.feedback(metric_name="test_metric", value=5)

    with pytest.raises(TensorZeroError):
        sync_client.feedback(
            metric_name="test_metric",
            value=5,
            episode_id=UUID("123e4567-e89b-12d3-a456-426614174000"),
            inference_id=UUID("123e4567-e89b-12d3-a456-426614174001"),
        )


def test_sync_tensorzero_error(sync_client: TensorZeroGateway):
    with pytest.raises(TensorZeroError) as excinfo:
        sync_client.inference(function_name="not_a_function", input={"messages": []})

    assert (
        str(excinfo.value)
        == 'TensorZeroError (status code 404): {"error":"Unknown function: not_a_function"}'
    )


def test_sync_basic_inference_with_content_block(sync_client: TensorZeroGateway):
    result = sync_client.inference(
        function_name="basic_test",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        Text(type="text", text="Hello"),
                        ToolCall(
                            type="tool_call",
                            id="1",
                            name="test_tool",
                            raw_arguments=json.dumps({"arg": "value"}),
                            raw_name="test_tool",
                            arguments={"arg": "value"},
                        ),
                        ToolResult(
                            name="test_tool",
                            result="success",
                            id="1",
                        ),
                    ],
                }
            ],
        },
    )
    assert isinstance(result, ChatInferenceResponse)
    assert result.variant_name == "test"
    content = result.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert isinstance(content[0], Text)
    assert (
        content[0].text
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    usage = result.usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 10


def test_sync_basic_inference_with_content_block_plain_dict(
    sync_client: TensorZeroGateway,
):
    result = sync_client.inference(
        function_name="basic_test",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {
                            "type": "tool_call",
                            "id": "1",
                            "name": "test",
                            "arguments": json.dumps({"arg": "value"}),
                        },
                        {
                            "type": "tool_result",
                            "name": "test",
                            "result": "success",
                            "id": "1",
                        },
                    ],
                }
            ],
        },
    )
    assert isinstance(result, ChatInferenceResponse)
    assert result.variant_name == "test"
    content = result.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert isinstance(content[0], Text)
    assert (
        content[0].text
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    usage = result.usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 10


def test_prepare_inference_request(sync_client: TensorZeroGateway):
    # Test a simple request with string input and a structured system message
    # This is a private method, so we ignore type checking
    request = sync_client._prepare_inference_request(  # type: ignore
        function_name="basic_test",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert request["input"]["messages"][0]["content"] == [
        {"type": "text", "text": "Hello"}
    ]
    assert request["input"]["system"] == {"assistant_name": "Alfred Pennyworth"}
    assert request["function_name"] == "basic_test"

    # Test a complex request that covers every argument of the client
    episode_id = uuid7()
    request = sync_client._prepare_inference_request(  # type: ignore
        function_name="basic_test",
        input={
            "system": "you are the bad guy",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        ToolCall(
                            type="tool_call",
                            id="1",
                            name="test_tool",
                            raw_arguments=json.dumps({"arg": "value"}),
                            raw_name="test_tool",
                            arguments={"arg": "value"},
                        )
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        ToolResult(
                            name="test_tool",
                            result="success",
                            id="1",
                        )
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        Text(type="text", arguments={"foo": "bar"}),
                        ToolResult(name="drill", result="screwed", id="aaaa"),
                    ],
                },
            ],
        },
        stream=True,
        dryrun=False,
        episode_id=episode_id,
        output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
        variant_name="baz",
        params={"chat_completion": {"temperature": 0.1}},
        tool_choice="auto",
        parallel_tool_calls=True,
        additional_tools=[
            {"name": "drill", "parameters": '{"foo": "bar"}', "description": "drills"}
        ],
    )

    assert request["input"]["messages"][0]["content"][0] == {
        "type": "tool_call",
        "id": "1",
        "name": "test_tool",
        "arguments": {"arg": "value"},
        "raw_name": "test_tool",
        "raw_arguments": '{"arg": "value"}',
    }
    assert request["input"]["messages"][1]["content"][0] == {
        "type": "tool_result",
        "name": "test_tool",
        "result": "success",
        "id": "1",
    }
    assert request["input"]["messages"][2]["content"][0] == {
        "type": "text",
        "arguments": {"foo": "bar"},
    }
    assert request["input"]["messages"][2]["content"][1] == {
        "type": "tool_result",
        "name": "drill",
        "result": "screwed",
        "id": "aaaa",
    }
    assert request["input"]["system"] == "you are the bad guy"
    assert request["stream"]
    assert not request["dryrun"]
    assert request["episode_id"] == str(episode_id)
    assert request["output_schema"] == {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }
    assert request["params"]["chat_completion"]["temperature"] == 0.1
    assert request["tool_choice"] == "auto"
    assert request["additional_tools"][0] == {
        "name": "drill",
        "parameters": '{"foo": "bar"}',
        "description": "drills",
        "strict": False,
    }
    assert len(request["additional_tools"]) == 1  # type: ignore
    assert request["variant_name"] == "baz"
    assert request["function_name"] == "basic_test"
    assert request["parallel_tool_calls"]


def test_extra_headers_raw(sync_client: TensorZeroGateway):
    with pytest.raises(TensorZeroError) as exc_info:
        sync_client.inference(
            function_name="basic_test",
            variant_name="openai",
            input={
                "system": {"assistant_name": "Alfred Pennyworth"},
                "messages": [{"role": "user", "content": "Write me a haiku"}],
            },
            extra_headers=[
                {
                    "variant_name": "openai",
                    "name": "Authorization",
                    "value": "fake_auth_token",
                }
            ],
        )
    assert "You didn't provide an API key" in str(exc_info.value)


def test_extra_body_raw(sync_client: TensorZeroGateway):
    result = sync_client.inference(
        function_name="basic_test",
        variant_name="openai",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Write me a haiku"}],
        },
        extra_body=[
            {"variant_name": "openai", "pointer": "/max_completion_tokens", "value": 2}
        ],
    )
    assert isinstance(result, ChatInferenceResponse)
    assert result.variant_name == "openai"
    content = result.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert isinstance(content[0], Text)
    assert content[0].text is not None
    assert len(content[0].text.split(" ")) <= 2
    usage = result.usage
    assert usage.output_tokens == 2


def test_extra_body_types(sync_client: TensorZeroGateway):
    result = sync_client.inference(
        function_name="basic_test",
        variant_name="openai",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [
                {
                    "role": "user",
                    "content": "Write me a haiku starting with the word 'Potato', and output a JSON object with the key 'haiku'",
                }
            ],
        },
        extra_body=[
            VariantExtraBody(
                variant_name="openai",
                pointer="/response_format",
                value={"type": "json_object"},
            ),
            ProviderExtraBody(
                model_provider_name="tensorzero::model_name::gpt-4o-mini-2024-07-18::provider_name::openai",
                pointer="/stop",
                value="Potato",
            ),
        ],
    )
    assert isinstance(result, ChatInferenceResponse)
    assert result.variant_name == "openai"
    content = result.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert isinstance(content[0], Text)
    assert content[0].text is not None
    assert '"haiku"' in content[0].text
    assert "Potato" not in content[0].text


def test_sync_dynamic_credentials(sync_client: TensorZeroGateway):
    result = sync_client.inference(
        function_name="basic_test",
        variant_name="test_dynamic_api_key",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
        credentials={"DUMMY_API_KEY": "good_key"},
    )
    assert isinstance(result, ChatInferenceResponse)
    assert result.variant_name == "test_dynamic_api_key"
    content = result.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert isinstance(content[0], Text)
    assert (
        content[0].text
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    usage = result.usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 10


def test_sync_err_in_stream(sync_client: TensorZeroGateway):
    result = sync_client.inference(
        function_name="basic_test",
        variant_name="err_in_stream",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
        stream=True,
    )
    assert isinstance(result, t.Iterator)

    next(result)
    next(result)
    next(result)
    with pytest.raises(TensorZeroError) as exc_info:
        next(result)
    assert "Dummy error in stream" in str(exc_info.value)
    remaining_chunks = list(result)
    assert len(remaining_chunks) == 13


@pytest.mark.asyncio
async def test_async_err_in_stream(async_client: AsyncTensorZeroGateway):
    result = await async_client.inference(
        function_name="basic_test",
        variant_name="err_in_stream",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
        stream=True,
    )
    assert isinstance(result, t.AsyncIterator)

    # anext() was added in Python 3.10, use __anext__() for older versions
    await result.__anext__()
    await result.__anext__()
    await result.__anext__()
    with pytest.raises(TensorZeroError) as exc_info:
        await result.__anext__()
    assert "Dummy error in stream" in str(exc_info.value)
    # Make this an async collect into a list
    remaining_chunks: t.List[InferenceChunk] = []
    async for chunk in result:
        remaining_chunks.append(chunk)
    assert len(remaining_chunks) == 13


@pytest.mark.asyncio
async def test_async_timeout_int_http():
    client_fut = AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000",
        timeout=1,
    )
    assert isinstance(client_fut, t.Awaitable)
    async with await client_fut as async_client:
        with pytest.raises(TensorZeroInternalError) as exc_info:
            await async_client.inference(
                function_name="basic_test",
                variant_name="slow",
                input={
                    "system": {"assistant_name": "TensorZero bot"},
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
        assert "HTTP Error: request timed out" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_timeout_int_embedded():
    client_fut = AsyncTensorZeroGateway.build_embedded(
        config_file=TEST_CONFIG_FILE,
        clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero-python-e2e",
        timeout=1,
    )
    assert inspect.isawaitable(client_fut)
    async with await client_fut as async_client:
        with pytest.raises(TensorZeroInternalError) as exc_info:
            await async_client.inference(
                function_name="basic_test",
                variant_name="slow",
                input={
                    "system": {"assistant_name": "TensorZero bot"},
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
        assert "HTTP Error: request timed out" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_timeout_float_http():
    client_fut = AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000",
        timeout=0.1,
    )
    assert inspect.isawaitable(client_fut)
    async with await client_fut as async_client:
        with pytest.raises(TensorZeroInternalError) as exc_info:
            await async_client.inference(
                function_name="basic_test",
                variant_name="slow",
                input={
                    "system": {"assistant_name": "TensorZero bot"},
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
        assert "HTTP Error: request timed out" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_timeout_float_embedded():
    client_fut = AsyncTensorZeroGateway.build_embedded(
        config_file=TEST_CONFIG_FILE,
        clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero-python-e2e",
        timeout=0.1,
    )
    assert inspect.isawaitable(client_fut)
    async with await client_fut as async_client:
        with pytest.raises(TensorZeroInternalError) as exc_info:
            await async_client.inference(
                function_name="basic_test",
                variant_name="slow",
                input={
                    "system": {"assistant_name": "TensorZero bot"},
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
        assert "HTTP Error: request timed out" in str(exc_info.value)


def test_sync_timeout_invalid():
    with pytest.raises(ValueError) as exc_info:
        TensorZeroGateway.build_http(gateway_url="http://localhost:3000", timeout=-1)
    assert (
        "Invalid timeout: cannot convert float seconds to Duration: value is negative"
        == str(exc_info.value)
    )


@pytest.mark.asyncio
async def test_async_non_verbose_errors():
    client_fut = AsyncTensorZeroGateway.build_http(
        gateway_url="http://tensorzero.invalid:3000", verbose_errors=False
    )
    assert inspect.isawaitable(client_fut)
    async with await client_fut as async_client:
        with pytest.raises(TensorZeroInternalError) as exc_info:
            await async_client.inference(
                function_name="basic_test",
                variant_name="slow",
                input={"messages": [{"role": "user", "content": "Hello"}]},
            )

        assert "dns error" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_verbose_errors():
    client_fut = AsyncTensorZeroGateway.build_http(
        gateway_url="http://tensorzero.invalid:3000", verbose_errors=True
    )
    assert inspect.isawaitable(client_fut)
    async with await client_fut as async_client:
        with pytest.raises(TensorZeroInternalError) as exc_info:
            await async_client.inference(
                function_name="basic_test",
                variant_name="slow",
                input={"messages": [{"role": "user", "content": "Hello"}]},
            )

        assert "dns error" in str(exc_info.value)


def test_sync_non_verbose_errors():
    with TensorZeroGateway.build_http(
        gateway_url="http://tensorzero.invalid:3000", verbose_errors=False
    ) as async_client:
        with pytest.raises(TensorZeroInternalError) as exc_info:
            async_client.inference(
                function_name="basic_test",
                variant_name="slow",
                input={"messages": [{"role": "user", "content": "Hello"}]},
            )

        assert "dns error" not in str(exc_info.value)


def test_sync_verbose_errors():
    with TensorZeroGateway.build_http(
        gateway_url="http://tensorzero.invalid:3000", verbose_errors=True
    ) as async_client:
        with pytest.raises(TensorZeroInternalError) as exc_info:
            async_client.inference(
                function_name="basic_test",
                variant_name="slow",
                input={"messages": [{"role": "user", "content": "Hello"}]},
            )

        assert "dns error" in str(exc_info.value)


def test_uuid7_import():
    from tensorzero.util import uuid7

    assert uuid7() is not None


def test_patch_sync_openai_client_sync_setup():
    client = OpenAI()
    tensorzero.patch_openai_client(
        client,
        config_file="../../examples/quickstart/config/tensorzero.toml",
        clickhouse_url=None,
        async_setup=False,
    )
    response = client.chat.completions.create(
        model="tensorzero::model_name::dummy::json",
        messages=[
            {
                "role": "user",
                "content": "Write a haiku about artificial intelligence.",
            }
        ],
    )
    assert response.choices[0].message.content == '{"answer":"Hello"}'


@pytest.mark.asyncio
async def test_patch_sync_openai_client_async_setup():
    client = OpenAI()
    patch_fut = tensorzero.patch_openai_client(
        client,
        config_file="../../examples/quickstart/config/tensorzero.toml",
        async_setup=True,
    )
    assert isinstance(patch_fut, t.Awaitable)
    client = await patch_fut
    response = client.chat.completions.create(
        model="tensorzero::model_name::dummy::json",
        messages=[
            {
                "role": "user",
                "content": "Write a haiku about artificial intelligence.",
            }
        ],
    )
    assert response.choices[0].message.content == '{"answer":"Hello"}'


def test_patch_openai_client_no_config():
    client = OpenAI()
    with pytest.warns(UserWarning, match="No config file provided"):
        tensorzero.patch_openai_client(client, async_setup=False)
    response = client.chat.completions.create(
        model="tensorzero::model_name::dummy::json",
        messages=[
            {
                "role": "user",
                "content": "Write a haiku about artificial intelligence.",
            }
        ],
    )
    assert response.choices[0].message.content == '{"answer":"Hello"}'


def test_patch_openai_client_with_config():
    client = OpenAI()
    tensorzero.patch_openai_client(
        client,
        config_file="../../tensorzero-internal/tests/e2e/tensorzero.toml",
        async_setup=False,
    )
    response = client.chat.completions.create(
        model="tensorzero::function_name::json_success",
        messages=[  # type: ignore
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "tensorzero::arguments": {
                            "assistant_name": "Alfred Pennyworth"
                        },
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "tensorzero::arguments": {"country": "Japan"}}
                ],
            },
        ],
    )
    assert response.choices[0].message.content == '{"answer":"Hello"}'


@pytest.mark.asyncio
async def test_patch_async_openai_client_sync_setup():
    client = AsyncOpenAI()
    tensorzero.patch_openai_client(
        client,
        config_file="../../examples/quickstart/config/tensorzero.toml",
        clickhouse_url=None,
        async_setup=False,
    )
    response = await client.chat.completions.create(
        model="tensorzero::model_name::dummy::json",
        messages=[
            {
                "role": "user",
                "content": "Write a haiku about artificial intelligence.",
            }
        ],
    )
    assert response.choices[0].message.content == '{"answer":"Hello"}'


@pytest.mark.asyncio
async def test_patch_async_openai_client_async_setup():
    client = AsyncOpenAI()
    patch_fut = tensorzero.patch_openai_client(
        client,
        config_file="../../examples/quickstart/config/tensorzero.toml",
        async_setup=True,
    )
    assert isinstance(patch_fut, t.Awaitable)
    client = await patch_fut
    response = await client.chat.completions.create(
        model="tensorzero::model_name::dummy::json",
        messages=[
            {
                "role": "user",
                "content": "Write a haiku about artificial intelligence.",
            }
        ],
    )
    assert response.choices[0].message.content == '{"answer":"Hello"}'


@pytest.mark.asyncio
async def test_patch_openai_missing_await():
    client = OpenAI()
    patch_fut = tensorzero.patch_openai_client(
        client,
        config_file="../../examples/quickstart/config/tensorzero.toml",
        clickhouse_url=None,
        async_setup=True,
    )

    with pytest.raises(RuntimeError) as exc_info:
        client.chat.completions.create(
            model="tensorzero::model_name::openai::gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Write a haiku about artificial intelligence.",
                }
            ],
        )
    assert (
        str(exc_info.value)
        == "TensorZero: Please await the result of `tensorzero.patch_openai_client` before using the client."
    )
    assert isinstance(patch_fut, t.Awaitable)
    # Await this before we exit the test, to avoid spurious 'Event loop is closed' errors
    await patch_fut


@pytest.mark.asyncio
async def test_patch_async_openai_missing_await():
    client = AsyncOpenAI()
    patch_fut = tensorzero.patch_openai_client(
        client,
        config_file="../../examples/quickstart/config/tensorzero.toml",
        clickhouse_url=None,
        async_setup=True,
    )
    with pytest.raises(RuntimeError) as exc_info:
        await client.chat.completions.create(
            model="tensorzero::model_name::openai::gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Write a haiku about artificial intelligence.",
                }
            ],
        )
    assert (
        str(exc_info.value)
        == "TensorZero: Please await the result of `tensorzero.patch_openai_client` before using the client."
    )
    # Await this before we exit the test, to avoid spurious 'Event loop is closed' errors
    assert isinstance(patch_fut, t.Awaitable)
    await patch_fut


def test_repeated_patch_openai_client_sync_setup():
    sync_client = OpenAI()
    tensorzero.patch_openai_client(
        sync_client,
        config_file="../../examples/quickstart/config/tensorzero.toml",
        async_setup=False,
    )
    with pytest.raises(RuntimeError) as exc_info:
        tensorzero.patch_openai_client(
            sync_client,
            config_file="../../examples/quickstart/config/tensorzero.toml",
            async_setup=False,
        )
    assert (
        str(exc_info.value)
        == "TensorZero: Already called 'tensorzero.patch_openai_client' on this OpenAI client."
    )

    async_client = AsyncOpenAI()
    tensorzero.patch_openai_client(
        async_client,
        config_file="../../examples/quickstart/config/tensorzero.toml",
        async_setup=False,
    )
    with pytest.raises(RuntimeError) as exc_info:
        tensorzero.patch_openai_client(
            async_client,
            config_file="../../examples/quickstart/config/tensorzero.toml",
            async_setup=False,
        )
    assert (
        str(exc_info.value)
        == "TensorZero: Already called 'tensorzero.patch_openai_client' on this OpenAI client."
    )


@pytest.mark.asyncio
async def test_repeated_patch_openai_client_async_setup():
    sync_client = OpenAI()
    patch_fut = tensorzero.patch_openai_client(
        sync_client,
        config_file="../../examples/quickstart/config/tensorzero.toml",
        async_setup=True,
    )
    assert isinstance(patch_fut, t.Awaitable)
    await patch_fut

    with pytest.raises(RuntimeError) as exc_info:
        new_patch_fut = tensorzero.patch_openai_client(
            sync_client, config_file="../../examples/quickstart/config/tensorzero.toml"
        )
        assert isinstance(new_patch_fut, t.Awaitable)
        await new_patch_fut
    assert (
        str(exc_info.value)
        == "TensorZero: Already called 'tensorzero.patch_openai_client' on this OpenAI client."
    )

    async_client = AsyncOpenAI()
    async_patch_fut = tensorzero.patch_openai_client(
        async_client,
        config_file="../../examples/quickstart/config/tensorzero.toml",
        async_setup=True,
    )
    assert isinstance(async_patch_fut, t.Awaitable)
    await async_patch_fut
    with pytest.raises(RuntimeError) as exc_info:
        new_async_patch_fut = tensorzero.patch_openai_client(
            async_client, config_file="../../examples/quickstart/config/tensorzero.toml"
        )
        assert isinstance(new_async_patch_fut, t.Awaitable)
        await new_async_patch_fut
    assert (
        str(exc_info.value)
        == "TensorZero: Already called 'tensorzero.patch_openai_client' on this OpenAI client."
    )


@pytest.mark.asyncio
async def test_close_patch_openai_client():
    sync_client = OpenAI()
    patch_fut = tensorzero.patch_openai_client(
        sync_client,
        config_file="../../examples/quickstart/config/tensorzero.toml",
        async_setup=True,
    )
    assert isinstance(patch_fut, t.Awaitable)
    await patch_fut
    tensorzero.close_patched_openai_client_gateway(sync_client)


@pytest.mark.asyncio
async def test_async_multi_turn_parallel_tool_use(async_client: AsyncTensorZeroGateway):
    episode_id = str(uuid7())

    system = {"assistant_name": "Dr. Mehta"}

    messages: t.List[t.Dict[str, t.Any]] = [
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

    response = await async_client.inference(
        function_name="weather_helper_parallel",
        variant_name="openai",
        episode_id=episode_id,
        input={
            "messages": messages,
            "system": system,
        },
        parallel_tool_calls=True,
    )
    assert isinstance(response, ChatInferenceResponse)

    messages.append(
        {
            "role": "assistant",
            "content": response.content,
        }
    )

    assert len(response.content) == 2

    new_content_blocks: t.List[t.Dict[str, t.Any]] = []

    for content_block in response.content:
        if content_block.type == "text":
            print("Got a text block...")
        elif content_block.type == "tool_call":
            assert isinstance(content_block, ToolCall)
            if content_block.name == "get_temperature":
                print("Calling get_temperature tool...")
                new_content_blocks.append(
                    {
                        "type": "tool_result",
                        "id": content_block.id,
                        "name": "get_temperature",
                        "result": "70",
                    }
                )
            elif content_block.name == "get_humidity":
                print("Calling get_humidity tool...")
                new_content_blocks.append(
                    {
                        "type": "tool_result",
                        "id": content_block.id,
                        "name": "get_humidity",
                        "result": "30",
                    }
                )
            else:
                print("Unknown tool call")
        else:
            print("Unknown content block type")

    messages.append(
        {
            "role": "user",
            "content": new_content_blocks,
        }
    )

    response = await async_client.inference(
        function_name="weather_helper_parallel",
        variant_name="openai",
        episode_id=episode_id,
        input={
            "messages": messages,
            "system": system,
        },
    )
    assert isinstance(response, ChatInferenceResponse)
    assert isinstance(response.content[0], Text)
    assistant_message = response.content[0].text
    assert assistant_message is not None

    assert "70" in assistant_message
    assert "30" in assistant_message


def test_text_arguments_deprecation_1170_warning(sync_client: TensorZeroGateway):
    """Test that using Text with dictionary for text parameter works but emits DeprecationWarning for #1170."""

    with pytest.warns(
        DeprecationWarning,
        match=r"Please use `ContentBlock\(type=\"text\", arguments=...\)` when providing arguments for a prompt template/schema. In a future release, `Text\(type=\"text\", text=...\)` will require a string literal.",
    ):
        response = sync_client.inference(
            function_name="json_success",
            input={
                "system": {"assistant_name": "Alfred Pennyworth"},
                "messages": [
                    {
                        "role": "user",
                        # Intentionally ignore the type error to check the deprecation warning
                        "content": [Text(type="text", text={"country": "Japan"})],  # type: ignore
                    }
                ],
            },
        )

    assert isinstance(response, JsonInferenceResponse)
    assert response.variant_name == "test"
    assert response.output.raw == '{"answer":"Hello"}'
    assert response.output.parsed == {"answer": "Hello"}
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 10
    assert response.finish_reason == FinishReason.STOP


def test_content_block_text_init_validation():
    """Test Text initialization validation for text and arguments parameters."""

    # Test providing neither `text` nor `arguments` fails
    with pytest.raises(
        ValueError, match=r"Either `text` or `arguments` must be provided."
    ):
        Text(type="text")

    # Test providing both `text` and `arguments` fails
    with pytest.raises(
        ValueError, match=r"Only one of `text` or `arguments` must be provided."
    ):
        Text(type="text", text="Hello", arguments={"foo": "bar"})

    # Test with valid `text` parameter
    text = Text(type="text", text="Hello")
    assert text.text == "Hello"
    assert text.arguments is None

    # Test with valid `arguments` parameter
    arguments = {"foo": "bar"}
    text = Text(type="text", arguments=arguments)
    assert text.text is None
    assert text.arguments == arguments


def test_sync_dynamic_evaluation_run(sync_client: TensorZeroGateway):
    response = sync_client.dynamic_evaluation_run(
        variants={"basic_test": "test2"},
        tags={"foo": "bar"},
    )
    assert isinstance(response, DynamicEvaluationRunResponse)
    run_id = response.run_id
    assert isinstance(run_id, UUID)
    assert run_id is not None

    # Get the episode id
    episode_id = sync_client.dynamic_evaluation_run_episode(
        run_id=run_id,
        task_name="basic_test",
    ).episode_id

    inference_response = sync_client.inference(
        function_name="basic_test",
        episode_id=episode_id,
        input={
            "messages": [{"role": "user", "content": "Hello, world!"}],
            "system": {"assistant_name": "Dr. Mehta"},
        },
    )
    assert isinstance(inference_response, ChatInferenceResponse)
    first_content_block = inference_response.content[0]
    assert isinstance(first_content_block, Text)
    assert first_content_block.text is not None
    assert first_content_block.text.startswith("Megumin")
    assert inference_response.variant_name == "test2"


@pytest.mark.asyncio
async def test_async_dynamic_evaluation_run(async_client: AsyncTensorZeroGateway):
    response = await async_client.dynamic_evaluation_run(
        variants={"basic_test": "test2"},
        tags={"foo": "bar"},
    )
    assert isinstance(response, DynamicEvaluationRunResponse)
    run_id = response.run_id
    assert isinstance(run_id, UUID)
    assert run_id is not None

    # Get the episode id
    episode_response = await async_client.dynamic_evaluation_run_episode(
        run_id=run_id,
        task_name="basic_test",
    )
    episode_id = episode_response.episode_id
    inference_response = await async_client.inference(
        function_name="basic_test",
        episode_id=episode_id,
        input={
            "messages": [{"role": "user", "content": "Hello, world!"}],
            "system": {"assistant_name": "Dr. Mehta"},
        },
    )
    assert isinstance(inference_response, ChatInferenceResponse)
    first_content_block = inference_response.content[0]
    assert isinstance(first_content_block, Text)
    assert first_content_block.text is not None
    assert first_content_block.text.startswith("Megumin")
    assert inference_response.variant_name == "test2"


def test_sync_chat_function_null_response(sync_client: TensorZeroGateway):
    """
    Test that an chat inference with null response (i.e. no generated content blocks) works as expected.
    """
    result = sync_client.inference(
        function_name="null_chat",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "No yapping!",
                }
            ],
        },
    )
    assert isinstance(result, ChatInferenceResponse)
    assert len(result.content) == 0


def test_sync_json_function_null_response(sync_client: TensorZeroGateway):
    """
    Test that a JSON inference with null response (i.e. no generated content blocks) works as expected.
    """
    result = sync_client.inference(
        function_name="null_json",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "Extract no data!",
                }
            ],
        },
    )
    assert isinstance(result, JsonInferenceResponse)
    assert result.output.raw is None
    assert result.output.parsed is None


def test_sync_bulk_insert_delete_datapoints(sync_client: TensorZeroGateway):
    datapoints = [
        ChatDatapointInsert(
            function_name="basic_test",
            input={
                "system": {"assistant_name": "foo"},
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "bar"}]}
                ],
            },
            output=[{"type": "text", "text": "foobar"}],
            allowed_tools=None,
            additional_tools=None,
            tool_choice="auto",
            parallel_tool_calls=False,
            tags=None,
        ),
        # Ensure deprecated ChatInferenceDatapointInput is still supported
        ChatInferenceDatapointInput(
            function_name="basic_test",
            input={
                "system": {"assistant_name": "Dummy"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "My synthetic input"}],
                    }
                ],
            },
            output=[
                {
                    "type": "tool_call",
                    "name": "get_temperature",
                    "arguments": {"location": "New York", "units": "fahrenheit"},
                }
            ],
            additional_tools=[
                {
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
                    "name": "get_temperature",
                    "strict": False,
                }
            ],
            tool_choice="auto",
            parallel_tool_calls=False,
            allowed_tools=None,
            tags=None,
        ),
        JsonDatapointInsert(
            function_name="json_success",
            input={
                "system": {"assistant_name": "foo"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "arguments": {"country": "US"}}],
                    }
                ],
            },
            output={"answer": "Hello"},
            output_schema=None,
            tags=None,
        ),
        # Ensure deprecated JsonInferenceDatapointInput is still supported
        JsonInferenceDatapointInput(
            function_name="json_success",
            input={
                "system": {"assistant_name": "foo"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "arguments": {"country": "US"}}],
                    }
                ],
            },
            output={"response": "Hello"},
            output_schema={
                "type": "object",
                "properties": {"response": {"type": "string"}},
            },
            tags=None,
        ),
    ]
    datapoint_ids = sync_client.bulk_insert_datapoints(
        dataset_name="test", datapoints=datapoints
    )
    assert len(datapoint_ids) == 4
    assert isinstance(datapoint_ids[0], UUID)
    assert isinstance(datapoint_ids[1], UUID)
    assert isinstance(datapoint_ids[2], UUID)
    assert isinstance(datapoint_ids[3], UUID)

    sync_client.delete_datapoint(dataset_name="test", datapoint_id=datapoint_ids[0])
    sync_client.delete_datapoint(dataset_name="test", datapoint_id=datapoint_ids[1])
    sync_client.delete_datapoint(dataset_name="test", datapoint_id=datapoint_ids[2])
    sync_client.delete_datapoint(dataset_name="test", datapoint_id=datapoint_ids[3])


@pytest.mark.asyncio
async def test_async_bulk_insert_delete_datapoints(
    async_client: AsyncTensorZeroGateway,
):
    datapoints = [
        ChatDatapointInsert(
            function_name="basic_test",
            input={
                "system": {"assistant_name": "foo"},
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "bar"}]}
                ],
            },
        ),
        ChatDatapointInsert(
            function_name="basic_test",
            input={
                "system": {"assistant_name": "Dummy"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "My synthetic input"}],
                    }
                ],
            },
            output=[
                {
                    "type": "tool_call",
                    "name": "get_temperature",
                    "arguments": {"location": "New York", "units": "fahrenheit"},
                }
            ],
            additional_tools=[
                {
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
                    "name": "get_temperature",
                    "strict": False,
                }
            ],
            tool_choice="auto",
            parallel_tool_calls=False,
            allowed_tools=None,
            tags=None,
        ),
        JsonDatapointInsert(
            function_name="json_success",
            input={
                "system": {"assistant_name": "foo"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "arguments": {"country": "US"}}],
                    }
                ],
            },
        ),
        JsonDatapointInsert(
            function_name="json_success",
            input={
                "system": {"assistant_name": "foo"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "arguments": {"country": "US"}}],
                    }
                ],
            },
            output={"response": "Hello"},
            output_schema={
                "type": "object",
                "properties": {"response": {"type": "string"}},
            },
            tags=None,
        ),
    ]
    dataset_name = f"test_{uuid7()}"
    datapoint_ids = await async_client.bulk_insert_datapoints(
        dataset_name=dataset_name, datapoints=datapoints
    )
    assert len(datapoint_ids) == 4
    assert isinstance(datapoint_ids[0], UUID)
    assert isinstance(datapoint_ids[1], UUID)
    assert isinstance(datapoint_ids[2], UUID)
    assert isinstance(datapoint_ids[3], UUID)

    # Get a chat datapoint
    datapoint = await async_client.get_datapoint(
        dataset_name=dataset_name, datapoint_id=datapoint_ids[0]
    )
    print(datapoint)
    assert isinstance(datapoint, ChatDatapoint)
    assert datapoint.function_name == "basic_test"
    assert datapoint.input == {
        "system": {"assistant_name": "foo"},
        "messages": [{"role": "user", "content": [{"type": "text", "value": "bar"}]}],
    }
    assert datapoint.output is None

    # Get a json datapoint
    datapoint = await async_client.get_datapoint(
        dataset_name=dataset_name, datapoint_id=datapoint_ids[2]
    )
    assert isinstance(datapoint, JsonDatapoint)
    assert datapoint.function_name == "json_success"
    assert datapoint.input == {
        "system": {"assistant_name": "foo"},
        "messages": [
            {"role": "user", "content": [{"type": "text", "value": {"country": "US"}}]}
        ],
    }
    assert datapoint.output is None

    # List datapoints
    listed_datapoints = await async_client.list_datapoints(
        dataset_name=dataset_name,
    )
    assert len(listed_datapoints) == 4
    # Assert that there are 2 chat and 2 json datapoints
    chat_datapoints = [dp for dp in listed_datapoints if isinstance(dp, ChatDatapoint)]
    json_datapoints = [dp for dp in listed_datapoints if isinstance(dp, JsonDatapoint)]
    assert len(chat_datapoints) == 2
    assert len(json_datapoints) == 2

    await async_client.delete_datapoint(
        dataset_name=dataset_name, datapoint_id=datapoint_ids[0]
    )
    await async_client.delete_datapoint(
        dataset_name=dataset_name, datapoint_id=datapoint_ids[1]
    )
    await async_client.delete_datapoint(
        dataset_name=dataset_name, datapoint_id=datapoint_ids[2]
    )
    await async_client.delete_datapoint(
        dataset_name=dataset_name, datapoint_id=datapoint_ids[3]
    )


def test_sync_invalid_input(sync_client: TensorZeroGateway):
    with pytest.raises(TensorZeroInternalError) as exc_info:
        sync_client.inference(
            function_name="json_success",
            input={"messages": [{"role": "user", "content": ["Invalid", "Content"]}]},
        )

    assert (
        str(exc_info.value)
        == 'Failed to deserialize JSON to tensorzero::client_input::ClientInput: messages[0].content[0]: invalid type: string "Invalid", expected object at line 1 column 54'
    )


@pytest.mark.asyncio
async def test_list_nonexistent_dataset(async_client: AsyncTensorZeroGateway):
    res = await async_client.list_datapoints(dataset_name="nonexistent_dataset")
    assert res == []


@pytest.mark.asyncio
async def test_get_nonexistent_datapoint(async_client: AsyncTensorZeroGateway):
    datapoint_id: UUID = uuid7()  # type: ignore
    with pytest.raises(TensorZeroError) as exc_info:
        await async_client.get_datapoint(
            dataset_name="nonexistent_dataset", datapoint_id=datapoint_id
        )
    assert "Datapoint not found for" in str(exc_info.value)
    assert "404" in str(exc_info.value)


def test_sync_multiple_text_blocks(sync_client: TensorZeroGateway):
    sync_client.inference(
        model_name="dummy::multiple-text-blocks",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": "world"},
                    ],
                }
            ]
        },
    )


def test_sync_include_original_response_chat(sync_client: TensorZeroGateway):
    response = sync_client.inference(
        model_name="dummy::good",
        input={"messages": [{"role": "user", "content": "Hello, world!"}]},
        include_original_response=True,
    )
    assert isinstance(response, ChatInferenceResponse)
    assert (
        response.original_response
        == '{\n  "id": "id",\n  "object": "text.completion",\n  "created": 1618870400,\n  "model": "text-davinci-002",\n  "choices": [\n    {\n      "text": "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.",\n      "index": 0,\n      "logprobs": null,\n      "finish_reason": null\n    }\n  ]\n}'
    )


def test_sync_include_original_response_json(sync_client: TensorZeroGateway):
    response = sync_client.inference(
        function_name="json_success",
        input={
            "system": {"assistant_name": "foo"},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "arguments": {"country": "US"}}],
                }
            ],
        },
        include_original_response=True,
    )
    assert isinstance(response, JsonInferenceResponse)
    assert response.original_response == '{"answer":"Hello"}'
