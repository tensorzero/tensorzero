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

import asyncio
import base64
import inspect
import json
import os
import tempfile
import threading
import time
import typing as t
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from os import path
from uuid import UUID

import pytest
import tensorzero
from clickhouse_connect import get_client  # type: ignore
from openai import AsyncOpenAI, OpenAI
from pytest import CaptureFixture
from tensorzero import (
    AlwaysExtraBody,
    AlwaysExtraBodyDelete,
    AsyncTensorZeroGateway,
    ChatInferenceResponse,
    DynamicEvaluationRunResponse,
    FeedbackResponse,
    FileBase64,
    FileUrl,
    FinishReason,
    ImageBase64,
    ImageUrl,
    InferenceChunk,
    JsonInferenceResponse,
    ModelProviderExtraBody,
    ModelProviderExtraBodyDelete,
    RawText,
    TensorZeroError,
    TensorZeroGateway,
    TensorZeroInternalError,
    Text,
    TextChunk,
    ThoughtChunk,
    ToolCall,
    ToolResult,
    VariantExtraBody,
    VariantExtraBodyDelete,
)
from tensorzero.types import (
    ChatChunk,
    JsonChunk,
    Template,
    Thought,
    ToolCallChunk,
)
from uuid_utils import uuid7

TEST_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../tensorzero-core/tests/e2e/config/tensorzero.*.toml",
)

# Test image with File block
basepath = path.dirname(__file__)
with open(f"{basepath}/../../../tensorzero-core/tests/e2e/providers/ferris.png", "rb") as f:
    ferris_png = base64.b64encode(f.read()).decode("ascii")


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
    assert usage.output_tokens == 1
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
            "messages": [{"role": "user", "content": [Text(type="text", text="Hello")]}],
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
        assert usage.output_tokens == 1
        assert result.finish_reason == FinishReason.STOP


@pytest.mark.asyncio
async def test_async_client_build_embedded_sync():
    client_ = AsyncTensorZeroGateway.build_embedded(
        config_file=TEST_CONFIG_FILE,
        clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero_e2e_tests",
        async_setup=False,
    )
    assert isinstance(client_, AsyncTensorZeroGateway)
    async with client_ as client:
        input = {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": [Text(type="text", text="Hello")]}],
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
        assert usage.output_tokens == 1
        assert result.finish_reason == FinishReason.STOP


@pytest.mark.asyncio
async def test_async_thought_input(async_client: AsyncTensorZeroGateway):
    result = await async_client.inference(
        model_name="dummy::echo_request_messages",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "thought",
                            "text": "my_first_thought",
                            "signature": "my_first_signature",
                        },
                        Thought(
                            text="my_second_thought",
                            signature="my_second_signature",
                            provider_type="dummy",
                        ),
                        Thought(
                            text="my_discarded_thought",
                            signature="my_discarded_signature",
                            provider_type="wrong_provider_type",
                        ),
                    ],
                }
            ],
        },
        tags={"key": "value"},
    )
    assert isinstance(result, ChatInferenceResponse)
    assert len(result.content) == 1
    assert isinstance(result.content[0], Text)
    # The last thought should be discarded, since 'provider_type' does not match
    assert (
        result.content[0].text
        == '{"system":null,"messages":[{"role":"user","content":[{"type":"thought","text":"my_first_thought","signature":"my_first_signature"},{"type":"thought","text":"my_second_thought","signature":"my_second_signature","provider_type":"dummy"}]}]}'
    )


@pytest.mark.asyncio
async def test_async_thought_signature_only_input(
    async_client: AsyncTensorZeroGateway,
):
    result = await async_client.inference(
        model_name="dummy::echo_request_messages",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "thought",
                            "signature": "my_first_signature",
                        },
                        Thought(signature="my_second_signature"),
                    ],
                }
            ],
        },
        tags={"key": "value"},
    )
    assert isinstance(result, ChatInferenceResponse)
    assert len(result.content) == 1
    assert isinstance(result.content[0], Text)
    assert (
        result.content[0].text
        == '{"system":null,"messages":[{"role":"user","content":[{"type":"thought","text":null,"signature":"my_first_signature"},{"type":"thought","text":null,"signature":"my_second_signature"}]}]}'
    )


def test_display_thought():
    t1 = Thought(signature="my_signature")
    print("str t1")
    print(str(t1))
    print("repr t1")
    print(repr(t1))
    assert str(t1) == "Thought(text=None, type='thought', signature='my_signature', summary=None, provider_type=None)"
    assert repr(t1) == "Thought(text=None, type='thought', signature='my_signature', summary=None, provider_type=None)"

    t2 = Thought(text="my_text", signature="my_signature")
    assert (
        str(t2) == "Thought(text='my_text', type='thought', signature='my_signature', summary=None, provider_type=None)"
    )
    assert (
        repr(t2)
        == "Thought(text='my_text', type='thought', signature='my_signature', summary=None, provider_type=None)"
    )

    t3 = Thought(text="my_text")
    assert str(t3) == "Thought(text='my_text', type='thought', signature=None, summary=None, provider_type=None)"
    assert repr(t3) == "Thought(text='my_text', type='thought', signature=None, summary=None, provider_type=None)"


@pytest.mark.asyncio
async def test_async_reasoning_inference(async_client: AsyncTensorZeroGateway):
    result = await async_client.inference(
        model_name="dummy::reasoner_with_signature",
        input={
            "messages": [{"role": "user", "content": "Hello"}],
        },
        tags={"key": "value"},
    )
    assert isinstance(result, ChatInferenceResponse)
    assert result.variant_name == "dummy::reasoner_with_signature"
    assert result.original_response is None
    content = result.content
    assert len(content) == 2
    assert isinstance(content[0], Thought)
    assert content[0].type == "thought"
    assert content[0].text == "hmmm"
    assert content[0].signature == "my_signature"
    assert isinstance(content[1], Text)
    assert content[1].type == "text"
    assert (
        content[1].text
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    usage = result.usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 2


@pytest.mark.asyncio
async def test_async_default_function_inference(
    async_client: AsyncTensorZeroGateway,
):
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
    assert usage.output_tokens == 1


@pytest.mark.asyncio
async def test_async_default_function_inference_plain_dict(
    async_client: AsyncTensorZeroGateway,
):
    input = {
        "system": "You are a helpful assistant named Alfred Pennyworth.",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "raw_text", "value": "Hello"}],
            }
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
    assert usage.output_tokens == 1


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
            assert chunk.content[0].signature is None
        elif i < len(expected_thinking) + len(expected_text):
            assert len(chunk.content) == 1
            assert chunk.content[0].type == "text"
            assert isinstance(chunk.content[0], TextChunk)
            assert chunk.content[0].text == expected_text[i - len(expected_thinking)]
        else:
            assert len(chunk.content) == 0
            assert chunk.usage is not None
            assert chunk.usage.input_tokens == 10
            assert chunk.usage.output_tokens == 18
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
    assert '"error":"Unknown function: does_not_exist"' in str(exc_info.value)


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
    assert usage.output_tokens == 1
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
    assert usage.output_tokens == 1


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
            if i == 0:
                assert chunk.content[0].raw_name == "get_temperature"
            else:
                assert chunk.content[0].raw_name == ""
            assert chunk.content[0].type == "tool_call"
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
    # Pick a variant that doesn't have a dummy provider streaming special-case
    stream = await async_client.inference(
        function_name="json_success",
        variant_name="test-diff-schema",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "arguments": {"country": "Japan"}}],
                },
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
        print("Chunk: ", chunk)
        if previous_inference_id is not None:
            assert chunk.inference_id == previous_inference_id
        if previous_episode_id is not None:
            assert chunk.episode_id == previous_episode_id
        previous_inference_id = chunk.inference_id
        previous_episode_id = chunk.episode_id
        variant_name = chunk.variant_name
        assert variant_name == "test-diff-schema"
        assert isinstance(chunk, JsonChunk)
        if i + 1 < len(chunks):
            assert chunk.raw == expected_text[i]
        else:
            assert chunk.usage is not None
            assert chunk.usage.input_tokens == 10
            assert chunk.usage.output_tokens == 16


@pytest.mark.asyncio
async def test_async_json_streaming_reasoning(
    async_client: AsyncTensorZeroGateway,
):
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
            assert chunk.usage.output_tokens == 7


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
        output_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        },
        stream=False,
    )
    assert isinstance(result, JsonInferenceResponse)
    assert result.variant_name == "test"
    assert result.output.raw == '{"answer":"Hello"}'
    assert result.output.parsed == {"answer": "Hello"}
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 1


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
    assert result.usage.output_tokens == 2


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
    assert result.usage.output_tokens == 1


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

    result = await async_client.feedback(metric_name="user_rating", value=5, episode_id=episode_id)
    assert isinstance(result, FeedbackResponse)
    assert isinstance(result.feedback_id, UUID)

    result = await async_client.feedback(metric_name="task_success", value=True, inference_id=inference_id)
    assert isinstance(result, FeedbackResponse)

    result = await async_client.feedback(
        metric_name="demonstration",
        value="hi how are you",
        inference_id=inference_id,
        tags={"author": "Alice"},
    )
    assert isinstance(result, FeedbackResponse)


@pytest.mark.asyncio
async def test_async_feedback_invalid_input(
    async_client: AsyncTensorZeroGateway,
):
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
async def test_async_tensorzero_error_http():
    async_client = AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000",
        verbose_errors=True,
        async_setup=False,
    )
    assert isinstance(async_client, AsyncTensorZeroGateway)
    with pytest.raises(TensorZeroError) as excinfo:
        await async_client.inference(function_name="not_a_function", input={"messages": []})

    assert (
        str(excinfo.value)
        == 'TensorZeroError (status code 404): {"error":"Unknown function: not_a_function","error_json":{"UnknownFunction":{"name":"not_a_function"}}}'
    )


@pytest.mark.asyncio
async def test_async_tensorzero_error_embedded():
    async_client = AsyncTensorZeroGateway.build_embedded(
        config_file=TEST_CONFIG_FILE,
        clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero_e2e_tests",
        async_setup=False,
    )
    assert isinstance(async_client, AsyncTensorZeroGateway)
    with pytest.raises(TensorZeroError) as excinfo:
        await async_client.inference(function_name="not_a_function", input={"messages": []})

    assert str(excinfo.value) == 'TensorZeroError (status code 404): {"error":"Unknown function: not_a_function"}'


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
    assert usage.output_tokens == 1


def test_sync_error():
    with pytest.raises(Exception) as exc_info:
        with TensorZeroGateway.build_http(gateway_url="http://localhost:3000"):
            raise Exception("My error")
    assert str(exc_info.value) == "My error"


@pytest.mark.asyncio
async def test_async_error():
    with pytest.raises(Exception) as exc_info:
        client_fut = AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000")
        assert isinstance(client_fut, t.Awaitable)
        async with await client_fut:
            raise Exception("My error")
    assert str(exc_info.value) == "My error"


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
    assert usage.output_tokens == 1

    # Wait for the cache entry to be written to ClickHouse
    time.sleep(1)

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

    # Wait for the cache entry to be written to ClickHouse
    time.sleep(1)

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
    assert usage.output_tokens == 1


def test_image_inference_base64(sync_client: TensorZeroGateway):
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
            "Base64": {
                "source_url": None,
                "mime_type": "image/png",
                "data": ferris_png,
                "storage_path": {
                    "kind": {"type": "disabled"},
                    "path": "observability/files/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png",
                },
            }
        }
    ]


def test_file_inference_base64_infer_mime_type(sync_client: TensorZeroGateway):
    input = {
        "system": "You are a helpful assistant named Alfred Pennyworth.",
        "messages": [
            {
                "role": "user",
                "content": [
                    FileBase64(
                        data=ferris_png,
                        mime_type=None,
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
            "Base64": {
                "source_url": None,
                "mime_type": "image/png",
                "data": ferris_png,
                "storage_path": {
                    "kind": {"type": "disabled"},
                    "path": "observability/files/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png",
                },
            }
        }
    ]


def test_file_inference_base64_bad_content_no_mime_type(sync_client: TensorZeroGateway):
    input = {
        "system": "You are a helpful assistant named Alfred Pennyworth.",
        "messages": [
            {
                "role": "user",
                "content": [
                    FileBase64(
                        data=base64.b64encode(b"Hello, world!").decode("ascii"),
                        mime_type=None,
                    )
                ],
            }
        ],
    }
    with pytest.raises(TensorZeroInternalError) as exc_info:
        sync_client.inference(
            model_name="dummy::extract_images",
            input=input,
            episode_id=uuid7(),  # This would not typically be done but this partially verifies that uuid7 is using a correct implementation
            # because the gateway validates some of the properties needed
        )
    assert (
        str(exc_info.value)
        == "Failed to deserialize JSON to tensorzero_types::message::Input: messages[0].content[0]: Invalid mime type: No mime type provided and unable to infer from data at line 1 column 177"
    )


def test_file_inference_base64(sync_client: TensorZeroGateway):
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
            "Base64": {
                "source_url": None,
                "mime_type": "image/png",
                "data": ferris_png,
                "storage_path": {
                    "kind": {"type": "disabled"},
                    "path": "observability/files/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png",
                },
            }
        }
    ]
    # Test pdf with File block
    basepath = path.dirname(__file__)
    with open(
        f"{basepath}/../../../tensorzero-core/tests/e2e/providers/deepseek_paper.pdf",
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
            "Base64": {
                "source_url": None,
                "mime_type": "application/pdf",
                "data": deepseek_paper_pdf,
                "storage_path": {
                    "kind": {"type": "disabled"},
                    "path": "observability/files/3e127d9a726f6be0fd81d73ccea97d96ec99419f59650e01d49183cd3be999ef.pdf",
                },
            }
        }
    ]


def test_image_inference_url_wrong_mime_type(sync_client: TensorZeroGateway):
    input = {
        "system": "You are a helpful assistant named Alfred Pennyworth.",
        "messages": [
            {
                "role": "user",
                "content": [
                    ImageUrl(
                        url="https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png",
                        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
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
            "Url": {
                "file_url": {
                    "url": "https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png",
                    "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "filename": None,
                }
            }
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
            "Url": {
                "file_url": {
                    "url": "https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png",
                    "mime_type": None,
                    "filename": None,
                }
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
            "Url": {
                "file_url": {
                    "url": "https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png",
                    "mime_type": None,
                    "filename": None,
                }
            }
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


def test_sync_inference_streaming_nonexistent_function(
    sync_client: TensorZeroGateway,
):
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


def test_sync_inference_streaming_malformed_input(
    sync_client: TensorZeroGateway,
):
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
    assert usage.output_tokens == 1


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
    assert usage.output_tokens == 2


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
    assert usage.output_tokens == 1


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
            if i == 0:
                assert chunk.content[0].raw_name == "get_temperature"
            else:
                assert chunk.content[0].raw_name == ""
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
            assert chunk.usage.output_tokens == 18


def test_sync_json_streaming(sync_client: TensorZeroGateway):
    # Pick a variant that doesn't have a dummy provider streaming special-case
    stream = sync_client.inference(
        function_name="json_success",
        variant_name="test-diff-schema",
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
        assert variant_name == "test-diff-schema"
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
            assert chunk.usage.output_tokens == 7


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
        output_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        },
        stream=False,
    )
    assert isinstance(result, JsonInferenceResponse)
    assert result.variant_name == "test"
    assert result.output.raw == '{"answer":"Hello"}'
    assert result.output.parsed == {"answer": "Hello"}
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 1


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
    assert result.usage.output_tokens == 2


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
    assert result.usage.output_tokens == 1


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

    result = sync_client.feedback(metric_name="user_rating", value=5, episode_id=episode_id)
    assert isinstance(result, FeedbackResponse)

    result = sync_client.feedback(metric_name="task_success", value=True, inference_id=inference_id)
    assert isinstance(result, FeedbackResponse)
    assert isinstance(result.feedback_id, UUID)

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


def test_sync_tensorzero_error_http():
    sync_client = TensorZeroGateway.build_http(
        gateway_url="http://localhost:3000",
        verbose_errors=True,
    )
    with pytest.raises(TensorZeroError) as excinfo:
        sync_client.inference(function_name="not_a_function", input={"messages": []})

    assert (
        str(excinfo.value)
        == 'TensorZeroError (status code 404): {"error":"Unknown function: not_a_function","error_json":{"UnknownFunction":{"name":"not_a_function"}}}'
    )


def test_sync_tensorzero_error_embedded():
    sync_client = TensorZeroGateway.build_embedded(
        config_file=TEST_CONFIG_FILE,
        clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero_e2e_tests",
    )
    with pytest.raises(TensorZeroError) as excinfo:
        sync_client.inference(function_name="not_a_function", input={"messages": []})

    assert str(excinfo.value) == 'TensorZeroError (status code 404): {"error":"Unknown function: not_a_function"}'


def test_sync_basic_inference_with_content_block(
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
    assert usage.output_tokens == 1


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
                            "arguments": {"arg": "value"},
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
    assert usage.output_tokens == 1


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
    assert request["input"]["messages"][0]["content"] == [{"type": "text", "text": "Hello"}]
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
        output_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        },
        variant_name="baz",
        params={"chat_completion": {"temperature": 0.1}},
        tool_choice="auto",
        parallel_tool_calls=True,
        additional_tools=[
            {
                "name": "drill",
                "parameters": '{"foo": "bar"}',
                "description": "drills",
            }
        ],
    )

    assert request["input"]["messages"][0]["content"][0] == {
        "type": "tool_call",
        "id": "1",
        "name": "test_tool",
        "arguments": '{"arg":"value"}',
    }
    assert request["input"]["messages"][1]["content"][0] == {
        "type": "tool_result",
        "name": "test_tool",
        "result": "success",
        "id": "1",
    }
    assert request["input"]["messages"][2]["content"][0] == {
        "type": "template",
        "name": "user",
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
        "type": "function",
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
        id = uuid7()
        sync_client.inference(
            function_name="basic_test",
            variant_name="openai",
            input={
                "system": {"assistant_name": "Alfred Pennyworth"},
                "messages": [{"role": "user", "content": f"Write me a haiku {id}"}],
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


def test_otlp_traces_extra_headers(sync_client: TensorZeroGateway):
    """Test that otlp_traces_extra_headers parameter is accepted and doesn't break inference."""
    result = sync_client.inference(
        function_name="basic_test",
        variant_name="openai",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Write me a haiku"}],
        },
        otlp_traces_extra_headers={
            "My-Custom-Header": "My-Custom-Value",
            "Another-Header": "Another-Value",
        },
    )
    # Verify the inference completed successfully
    assert isinstance(result, ChatInferenceResponse)
    assert result.variant_name == "openai"
    content = result.content
    assert len(content) >= 1
    assert content[0].type == "text"
    assert isinstance(content[0], Text)
    assert content[0].text is not None


def test_extra_body_raw(sync_client: TensorZeroGateway):
    result = sync_client.inference(
        function_name="basic_test",
        variant_name="openai",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Write me a haiku"}],
        },
        extra_body=[
            {
                "variant_name": "openai",
                "pointer": "/max_completion_tokens",
                "value": 2,
            }
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
            ModelProviderExtraBody(
                model_name="gpt-4o-mini-2024-07-18",
                provider_name="openai",
                pointer="/stop",
                value="Potato",
            ),
            ModelProviderExtraBody(
                model_name="gpt-4o-mini-2024-07-18",
                provider_name="openai",
                pointer="/should_be_deleted_provider",
                value=2,
            ),
            ModelProviderExtraBodyDelete(
                model_name="gpt-4o-mini-2024-07-18",
                provider_name="openai",
                pointer="/should_be_deleted_provider",
            ),
            VariantExtraBody(
                variant_name="openai",
                pointer="/should_be_deleted_variant",
                value=2,
            ),
            VariantExtraBodyDelete(
                variant_name="openai",
                pointer="/should_be_deleted_variant",
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


def test_all_extra_body(sync_client: TensorZeroGateway):
    """Test that AlwaysExtraBody applies to all variants."""
    result = sync_client.inference(
        function_name="basic_test",
        variant_name="openai",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Write me a haiku"}],
        },
        extra_body=[
            AlwaysExtraBody(
                pointer="/max_completion_tokens",
                value=2,
            )
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


def test_all_extra_body_with_delete(sync_client: TensorZeroGateway):
    """Test that AlwaysExtraBody can delete fields across all variants."""
    result = sync_client.inference(
        function_name="basic_test",
        variant_name="openai",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Write me a haiku"}],
        },
        extra_body=[
            AlwaysExtraBody(
                pointer="/should_be_deleted_all",
                value=2,
            ),
            AlwaysExtraBodyDelete(
                pointer="/should_be_deleted_all",
            ),
            AlwaysExtraBody(
                pointer="/max_completion_tokens",
                value=10,
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
    usage = result.usage
    assert usage.output_tokens <= 10


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
    assert usage.output_tokens == 1


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
        clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero_e2e_tests",
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
        clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero_e2e_tests",
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
    assert "Invalid timeout: cannot convert float seconds to Duration: value is negative" == str(exc_info.value)


# If TENSORZERO_E2E_PROXY is set, then we'll make requests using the proxy, which results in different error messages
# if the target URL is unreachable.
@contextmanager
def without_env_tensorzero_e2e_proxy():
    old_value = os.environ.pop("TENSORZERO_E2E_PROXY", None)
    try:
        yield
    finally:
        if old_value is not None:
            os.environ["TENSORZERO_E2E_PROXY"] = old_value


@pytest.mark.asyncio
async def test_async_non_verbose_errors():
    with without_env_tensorzero_e2e_proxy():
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
    with without_env_tensorzero_e2e_proxy():
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
    with without_env_tensorzero_e2e_proxy():
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
    with without_env_tensorzero_e2e_proxy():
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
                "content": "Write a haiku about TensorZero.",
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
                "content": "Write a haiku about TensorZero.",
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
                "content": "Write a haiku about TensorZero.",
            }
        ],
    )
    assert response.choices[0].message.content == '{"answer":"Hello"}'


def test_patch_openai_client_with_config():
    client = OpenAI()
    tensorzero.patch_openai_client(
        client,
        config_file="../../tensorzero-core/tests/e2e/config/tensorzero.*.toml",
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
                        "tensorzero::arguments": {"assistant_name": "Alfred Pennyworth"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "tensorzero::arguments": {"country": "Japan"},
                    }
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
                "content": "Write a haiku about TensorZero.",
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
                "content": "Write a haiku about TensorZero.",
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
                    "content": "Write a haiku about TensorZero.",
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
                    "content": "Write a haiku about TensorZero.",
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
    assert str(exc_info.value) == "TensorZero: Already called 'tensorzero.patch_openai_client' on this OpenAI client."

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
    assert str(exc_info.value) == "TensorZero: Already called 'tensorzero.patch_openai_client' on this OpenAI client."


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
            sync_client,
            config_file="../../examples/quickstart/config/tensorzero.toml",
        )
        assert isinstance(new_patch_fut, t.Awaitable)
        await new_patch_fut
    assert str(exc_info.value) == "TensorZero: Already called 'tensorzero.patch_openai_client' on this OpenAI client."

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
            async_client,
            config_file="../../examples/quickstart/config/tensorzero.toml",
        )
        assert isinstance(new_async_patch_fut, t.Awaitable)
        await new_async_patch_fut
    assert str(exc_info.value) == "TensorZero: Already called 'tensorzero.patch_openai_client' on this OpenAI client."


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
async def test_async_multi_turn_parallel_tool_use(
    async_client: AsyncTensorZeroGateway,
):
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


def test_text_arguments_deprecation_1170_warning(
    sync_client: TensorZeroGateway,
):
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
                        "content": [
                            Text(type="text", text={"country": "Japan"}),  # type: ignore
                        ],
                    }
                ],
            },
        )

    assert isinstance(response, JsonInferenceResponse)
    assert response.variant_name == "test"
    assert response.output.raw == '{"answer":"Hello"}'
    assert response.output.parsed == {"answer": "Hello"}
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 1
    assert response.finish_reason == FinishReason.STOP


def test_content_block_text_init_validation():
    """Test Text initialization validation for text and arguments parameters."""

    # Test providing neither `text` nor `arguments` fails
    with pytest.raises(ValueError, match=r"Either `text` or `arguments` must be provided."):
        Text(type="text")

    # Test providing both `text` and `arguments` fails
    with pytest.raises(ValueError, match=r"Only one of `text` or `arguments` must be provided."):
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
async def test_async_dynamic_evaluation_run(
    async_client: AsyncTensorZeroGateway,
):
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


def test_sync_invalid_input(sync_client: TensorZeroGateway):
    with pytest.raises(TensorZeroInternalError) as exc_info:
        sync_client.inference(
            function_name="json_success",
            input={"messages": [{"role": "user", "content": ["Invalid", "Content"]}]},
        )

    assert (
        str(exc_info.value)
        == 'Failed to deserialize JSON to tensorzero_types::message::Input: messages[0].content[0]: invalid type: string "Invalid", expected object at line 1 column 54'
    )


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
                    "content": [{"type": "template", "name": "user", "arguments": {"country": "US"}}],
                }
            ],
        },
        include_original_response=True,
    )
    assert isinstance(response, JsonInferenceResponse)
    assert response.original_response == '{"answer":"Hello"}'


def test_sync_clickhouse_batch_writes():
    # Create a temp file and write to it
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(b"gateway.observability.enabled = true\n")
        temp_file.write(b"gateway.observability.batch_writes.enabled = true\n")
        temp_file.write(b"gateway.observability.batch_writes.__force_allow_embedded_batch_writes = true\n")
        temp_file.flush()
        clickhouse_url = "http://chuser:chpassword@127.0.0.1:8123/tensorzero_e2e_tests"
        client = TensorZeroGateway.build_embedded(
            config_file=temp_file.name,
            clickhouse_url=clickhouse_url,
        )
        num_inferences = 100
        results: t.List[t.Any] = []
        episode_id = str(uuid7())
        for _ in range(num_inferences):
            results.append(
                client.inference(
                    model_name="dummy::good",
                    episode_id=episode_id,
                    input={
                        "messages": [{"role": "user", "content": "Hello, world!"}],
                    },
                )
            )

        assert len(results) == num_inferences

        # Wait for results to be written to ClickHouse
        time.sleep(1)

        expected_inference_ids = set(result.inference_id for result in results)

        clickhouse_client = get_client(dsn=clickhouse_url)
        clickhouse_result = clickhouse_client.query_df(  # type: ignore
            f"SELECT * FROM ChatInference where episode_id = '{episode_id}'"
        )
        assert len(clickhouse_result) == num_inferences  # type: ignore

        actual_inference_ids = set(row.id for row in clickhouse_result.iloc)  # type: ignore
        assert actual_inference_ids == expected_inference_ids


@pytest.mark.asyncio
async def test_async_clickhouse_batch_writes():
    # Create a temp file and write to it
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(b"gateway.observability.enabled = true\n")
        temp_file.write(b"gateway.observability.batch_writes.enabled = true\n")
        temp_file.write(b"gateway.observability.batch_writes.__force_allow_embedded_batch_writes = true\n")
        temp_file.flush()
        clickhouse_url = "http://chuser:chpassword@127.0.0.1:8123/tensorzero_e2e_tests"
        client_fut = AsyncTensorZeroGateway.build_embedded(
            config_file=temp_file.name,
            clickhouse_url=clickhouse_url,
        )
        assert inspect.isawaitable(client_fut)
        client = await client_fut
        num_inferences = 100
        futures: t.List[t.Awaitable[t.Any]] = []
        episode_id = str(uuid7())
        for _ in range(num_inferences):
            futures.append(
                client.inference(
                    model_name="dummy::good",
                    episode_id=episode_id,
                    input={
                        "messages": [{"role": "user", "content": "Hello, world!"}],
                    },
                )
            )

        results = await asyncio.gather(*futures)
        assert len(results) == num_inferences

        # Wait for results to be written to ClickHouse
        await asyncio.sleep(1)

        expected_inference_ids = set(result.inference_id for result in results)

        clickhouse_client = get_client(dsn=clickhouse_url)
        clickhouse_result = clickhouse_client.query_df(  # type: ignore
            f"SELECT * FROM ChatInference where episode_id = '{episode_id}'"
        )
        assert len(clickhouse_result) == num_inferences  # type: ignore

        actual_inference_ids = set(row.id for row in clickhouse_result.iloc)  # type: ignore
        assert actual_inference_ids == expected_inference_ids


def test_sync_cannot_enable_batch_writes():
    # Create a temp file and write to it
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(b"gateway.observability.enabled = true\n")
        temp_file.write(b"gateway.observability.batch_writes.enabled = true\n")
        temp_file.flush()
        clickhouse_url = "http://chuser:chpassword@127.0.0.1:8123/tensorzero_e2e_tests"
        with pytest.raises(TensorZeroInternalError) as exc_info:
            TensorZeroGateway.build_embedded(
                config_file=temp_file.name,
                clickhouse_url=clickhouse_url,
            )
        assert (
            str(exc_info.value)
            == """Failed to construct TensorZero client: Clickhouse(Other { source: TensorZeroInternalError(Error(Config { message: "`[gateway.observability.batch_writes]` is not yet supported in embedded gateway mode" })) })"""
        )


@pytest.mark.asyncio
async def test_async_cannot_enable_batch_writes():
    # Create a temp file and write to it
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(b"gateway.observability.enabled = true\n")
        temp_file.write(b"gateway.observability.batch_writes.enabled = true\n")
        temp_file.flush()
        clickhouse_url = "http://chuser:chpassword@127.0.0.1:8123/tensorzero_e2e_tests"
        client_fut = AsyncTensorZeroGateway.build_embedded(
            config_file=temp_file.name,
            clickhouse_url=clickhouse_url,
        )
        assert inspect.isawaitable(client_fut)
        with pytest.raises(TensorZeroInternalError) as exc_info:
            await client_fut
        assert (
            str(exc_info.value)
            == """Failed to construct TensorZero client: Clickhouse(Other { source: TensorZeroInternalError(Error(Config { message: "`[gateway.observability.batch_writes]` is not yet supported in embedded gateway mode" })) })"""
        )


def test_sync_chat_function_named_template(sync_client: TensorZeroGateway):
    """
    Test that an chat inference with null response (i.e. no generated content blocks) works as expected.
    """
    result = sync_client.inference(
        function_name="basic_test_template_no_schema",
        variant_name="test",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        Template(
                            name="my_custom_template",
                            arguments={
                                "first_variable": "first_from_python",
                                "second_variable": "second_from_python",
                            },
                        ),
                        {
                            "type": "template",
                            "name": "my_custom_template",
                            "arguments": {
                                "first_variable": "first_from_dict",
                                "second_variable": "second_from_dict",
                            },
                        },
                    ],
                }
            ],
        },
    )
    assert isinstance(result, ChatInferenceResponse)
    assert len(result.content) == 1
    assert isinstance(result.content[0], Text)
    assert (
        result.content[0].text
        == """{"system":"The system text was `none`","messages":[{"role":"user","content":[{"type":"text","text":"New template: first_variable=first_from_python second_variable=second_from_python"},{"type":"text","text":"New template: first_variable=first_from_dict second_variable=second_from_dict"}]}]}"""
    )


def test_http_client_no_spurious_log(capfd: CaptureFixture[str]):
    client = TensorZeroGateway.build_http(
        gateway_url="http://localhost:3000",
        verbose_errors=True,
    )
    assert client is not None
    captured = capfd.readouterr()
    if os.environ.get("TENSORZERO_E2E_PROXY") is not None:
        # We'll get some logs lines in CI due to TENSORZERO_E2E_PROXY being set
        for line in captured.out.splitlines():
            assert "Using proxy URL from TENSORZERO_E2E_PROXY" in line, f"Unexpected log line: {line}"
    else:
        assert captured.out == ""

    assert captured.err == ""


@pytest.mark.asyncio
async def test_async_http_client_no_spurious_log(capfd: CaptureFixture[str]):
    client_fut = AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000",
        verbose_errors=True,
    )
    assert inspect.isawaitable(client_fut)
    client = await client_fut
    assert client is not None
    captured = capfd.readouterr()
    if os.environ.get("TENSORZERO_E2E_PROXY") is not None:
        # We'll get some logs lines in CI due to TENSORZERO_E2E_PROXY being set
        for line in captured.out.splitlines():
            assert "Using proxy URL from TENSORZERO_E2E_PROXY" in line, f"Unexpected log line: {line}"
    else:
        assert captured.out == ""
    assert captured.err == ""


def test_embedded_client_no_spurious_log(capfd: CaptureFixture[str]):
    client = TensorZeroGateway.build_embedded(
        config_file=TEST_CONFIG_FILE,
        clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero_e2e_tests",
    )
    assert client is not None
    captured = capfd.readouterr()
    assert captured.err == ""
    if os.environ.get("TENSORZERO_E2E_PROXY") is not None:
        # We'll get some logs lines in CI due to TENSORZERO_E2E_PROXY being set
        for line in captured.out.splitlines():
            assert "Using proxy URL from TENSORZERO_E2E_PROXY" in line, f"Unexpected log line: {line}"
    else:
        assert captured.out == ""


@pytest.mark.asyncio
async def test_async_embedded_client_no_spurious_log(
    capfd: CaptureFixture[str],
):
    client_fut = AsyncTensorZeroGateway.build_embedded(
        config_file=TEST_CONFIG_FILE,
        clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero_e2e_tests",
    )
    assert inspect.isawaitable(client_fut)
    client = await client_fut
    assert client is not None
    captured = capfd.readouterr()
    assert captured.err == ""
    if os.environ.get("TENSORZERO_E2E_PROXY") is not None:
        # We'll get some logs lines in CI due to TENSORZERO_E2E_PROXY being set, b
        for line in captured.out.splitlines():
            assert "Using proxy URL from TENSORZERO_E2E_PROXY" in line, f"Unexpected log line: {line}"
    else:
        assert captured.out == ""


@pytest.mark.asyncio
async def test_async_otlp_traces_extra_headers(
    async_client: AsyncTensorZeroGateway,
):
    """Test that otlp_traces_extra_headers parameter is accepted in async inference."""
    result = await async_client.inference(
        function_name="basic_test",
        variant_name="openai",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Write me a haiku"}],
        },
        otlp_traces_extra_headers={
            "My-Async-Header": "My-Async-Value",
            "Test-Header": "Test-Value",
        },
    )
    # Verify the inference completed successfully
    assert isinstance(result, ChatInferenceResponse)
    assert result.variant_name == "openai"
    content = result.content
    assert len(content) >= 1
    assert content[0].type == "text"
    assert isinstance(content[0], Text)
    assert content[0].text is not None


def test_capfd_captured_warnings(capfd: CaptureFixture[str]):
    client = TensorZeroGateway.build_embedded()
    assert client is not None
    captured = capfd.readouterr()
    assert captured.err == ""
    assert "Disabling observability:" in captured.out
