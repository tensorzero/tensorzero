# type: ignore
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

from copy import deepcopy
from time import sleep, time
from uuid import UUID

import pytest
import pytest_asyncio
from tensorzero import (
    AsyncTensorZeroGateway,
    ChatInferenceResponse,
    FeedbackResponse,
    JsonInferenceResponse,
    TensorZeroGateway,
    Text,
    ToolCall,
    ToolResult,
)
from tensorzero.types import TensorZeroError
from tensorzero.util import uuid7


@pytest_asyncio.fixture
async def async_client():
    async with AsyncTensorZeroGateway("http://localhost:3000") as client:
        yield client


@pytest.mark.asyncio
async def test_async_basic_inference(async_client):
    input = {
        "system": {"assistant_name": "Alfred Pennyworth"},
        "messages": [{"role": "user", "content": Text(type="text", text="Hello")}],
    }
    input_copy = deepcopy(input)
    result = await async_client.inference(
        function_name="basic_test",
        input=input,
        episode_id=uuid7(),  # This would not typically be done but this partially verifies that uuid7 is using a correct implementation
        # because the gateway validates some of the properties needed
    )
    assert input == input_copy, "Input should not be modified by the client"
    assert result.variant_name == "test"
    assert isinstance(result, ChatInferenceResponse)
    content = result.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert (
        content[0].text
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    usage = result.usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 10


@pytest.mark.asyncio
async def test_async_inference_streaming(async_client):
    start_time = time()
    stream = await async_client.inference(
        function_name="basic_test",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
        stream=True,
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
            assert chunk.inference_id == previous_inference_id
        if previous_episode_id is not None:
            assert chunk.episode_id == previous_episode_id
        previous_inference_id = chunk.inference_id
        previous_episode_id = chunk.episode_id
        variant_name = chunk.variant_name
        assert variant_name == "test"
        if i + 1 < len(chunks):
            assert len(chunk.content) == 1
            assert chunk.content[0].type == "text"
            assert chunk.content[0].text == expected_text[i]
        else:
            assert len(chunk.content) == 0
            assert chunk.usage.input_tokens == 10
            assert chunk.usage.output_tokens == 16


@pytest.mark.asyncio
async def test_async_inference_streaming_nonexistent_function(async_client):
    with pytest.raises(TensorZeroError) as exc_info:
        await async_client.inference(
            function_name="does_not_exist",
            input={
                "system": {"assistant_name": "Alfred Pennyworth"},
                "messages": [{"role": "user", "content": "Hello"}],
            },
            stream=True,
        )
    assert exc_info.value.status_code == 404
    assert (
        str(exc_info.value)
        == 'TensorZeroError (status code 404): {"error":"Unknown function: does_not_exist"}'
    )


@pytest.mark.asyncio
async def test_async_inference_streaming_malformed_input(async_client):
    with pytest.raises(TensorZeroError) as exc_info:
        await async_client.inference(
            function_name="basic_test",
            input={
                "system": {"name_of_assistant": "Alfred Pennyworth"},  # WRONG
                "messages": [{"role": "user", "content": "Hello"}],
            },
            stream=True,
        )
    assert exc_info.value.status_code == 400
    assert (
        str(exc_info.value)
        == 'TensorZeroError (status code 400): {"error":"JSON Schema validation failed for Function:\\n\\n\\"assistant_name\\" is a required property\\nData: {\\"name_of_assistant\\":\\"Alfred Pennyworth\\"}Schema: {\\"type\\":\\"object\\",\\"properties\\":{\\"assistant_name\\":{\\"type\\":\\"string\\"}},\\"required\\":[\\"assistant_name\\"]}"}'
    )


@pytest.mark.asyncio
async def test_async_tool_call_inference(async_client):
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
    assert result.variant_name == "variant"
    assert isinstance(result, ChatInferenceResponse)
    content = result.content
    assert len(content) == 1
    assert content[0].type == "tool_call"
    assert content[0].raw_name == "get_temperature"
    assert content[0].id == "0"
    assert content[0].raw_arguments == '{"location":"Brooklyn","units":"celsius"}'
    assert content[0].name == "get_temperature"
    assert content[0].arguments == {"location": "Brooklyn", "units": "celsius"}
    usage = result.usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 10


@pytest.mark.asyncio
async def test_async_malformed_tool_call_inference(async_client):
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
    assert result.variant_name == "bad_tool"
    assert isinstance(result, ChatInferenceResponse)
    content = result.content
    assert len(content) == 1
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
async def test_async_tool_call_streaming(async_client):
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
        if i + 1 < len(chunks):
            assert len(chunk.content) == 1
            assert chunk.content[0].type == "tool_call"
            assert chunk.content[0].raw_name == "get_temperature"
            assert chunk.content[0].id == "0"
            assert chunk.content[0].raw_arguments == expected_text[i]
        else:
            assert len(chunk.content) == 0
            assert chunk.usage.input_tokens == 10
            assert chunk.usage.output_tokens == 5


@pytest.mark.asyncio
async def test_async_json_streaming(async_client):
    # We don't actually have a streaming JSON function implemented in `dummy.rs` but it doesn't matter for this test since
    # TensorZero doesn't parse the JSON output of the function for streaming calls.
    stream = await async_client.inference(
        function_name="json_success",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": {"country": "Japan"}}],
        },
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
            assert chunk.inference_id == previous_inference_id
        if previous_episode_id is not None:
            assert chunk.episode_id == previous_episode_id
        previous_inference_id = chunk.inference_id
        previous_episode_id = chunk.episode_id
        variant_name = chunk.variant_name
        assert variant_name == "test"
        if i + 1 < len(chunks):
            assert chunk.raw == expected_text[i]
        else:
            assert chunk.usage.input_tokens == 10
            assert chunk.usage.output_tokens == 16


@pytest.mark.asyncio
async def test_async_json_success(async_client):
    result = await async_client.inference(
        function_name="json_success",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": {"country": "Japan"}}],
        },
        stream=False,
    )
    assert result.variant_name == "test"
    assert isinstance(result, JsonInferenceResponse)
    assert result.output.raw == '{"answer":"Hello"}'
    assert result.output.parsed == {"answer": "Hello"}
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 10


@pytest.mark.asyncio
async def test_async_json_failure(async_client):
    result = await async_client.inference(
        function_name="json_fail",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello, world!"}],
        },
        stream=False,
    )
    assert result.variant_name == "test"
    assert isinstance(result, JsonInferenceResponse)
    assert (
        result.output.raw
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    assert result.output.parsed is None
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 10


@pytest.mark.asyncio
async def test_async_feedback(async_client):
    result = await async_client.feedback(
        metric_name="user_rating", value=5, episode_id=uuid7()
    )
    assert isinstance(result, FeedbackResponse)

    result = await async_client.feedback(
        metric_name="task_success", value=True, inference_id=uuid7()
    )
    assert isinstance(result, FeedbackResponse)

    # For demonstrations, we validate that the format is correct and the inference exists.
    result = await async_client.inference(
        function_name="basic_test",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    # Wait for the inference to be created in ClickHouse
    sleep(1)
    result = await async_client.feedback(
        metric_name="demonstration",
        value="hi how are you",
        inference_id=result.inference_id,
        tags={"author": "Alice"},
    )
    assert isinstance(result, FeedbackResponse)


@pytest.mark.asyncio
async def test_async_feedback_invalid_input(async_client):
    with pytest.raises(ValueError):
        await async_client.feedback(metric_name="test_metric", value=5)

    with pytest.raises(ValueError):
        await async_client.feedback(
            metric_name="test_metric",
            value=5,
            episode_id=UUID("123e4567-e89b-12d3-a456-426614174000"),
            inference_id=UUID("123e4567-e89b-12d3-a456-426614174001"),
        )


@pytest.mark.asyncio
async def test_async_tensorzero_error(async_client):
    with pytest.raises(TensorZeroError) as excinfo:
        await async_client.inference(
            function_name="not_a_function", input={"messages": []}
        )

    assert (
        str(excinfo.value)
        == 'TensorZeroError (status code 404): {"error":"Unknown function: not_a_function"}'
    )


@pytest.fixture
def sync_client():
    with TensorZeroGateway("http://localhost:3000") as client:
        yield client


def test_sync_basic_inference(sync_client):
    result = sync_client.inference(
        function_name="basic_test",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert result.variant_name == "test"
    assert isinstance(result, ChatInferenceResponse)
    content = result.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert (
        content[0].text
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    usage = result.usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 10


def test_sync_malformed_inference(sync_client):
    with pytest.raises(TensorZeroError) as exc_info:
        sync_client.inference(
            function_name="basic_test",
            input={
                "system": {"name_of_assistant": "Alfred Pennyworth"},  # WRONG
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
    assert exc_info.value.status_code == 400


def test_sync_inference_streaming(sync_client):
    start_time = time()
    stream = sync_client.inference(
        function_name="basic_test",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
        stream=True,
    )
    first_chunk_duration = None
    chunks = []
    for chunk in stream:
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
            assert chunk.inference_id == previous_inference_id
        if previous_episode_id is not None:
            assert chunk.episode_id == previous_episode_id
        previous_inference_id = chunk.inference_id
        previous_episode_id = chunk.episode_id
        variant_name = chunk.variant_name
        assert variant_name == "test"
        if i + 1 < len(chunks):
            assert len(chunk.content) == 1
            assert chunk.content[0].type == "text"
            assert chunk.content[0].text == expected_text[i]
        else:
            assert len(chunk.content) == 0
            assert chunk.usage.input_tokens == 10
            assert chunk.usage.output_tokens == 16


def test_sync_inference_streaming_nonexistent_function(sync_client):
    with pytest.raises(TensorZeroError) as exc_info:
        sync_client.inference(
            function_name="does_not_exist",
            input={
                "system": {"assistant_name": "Alfred Pennyworth"},
                "messages": [{"role": "user", "content": "Hello"}],
            },
            stream=True,
        )
    assert exc_info.value.status_code == 404


def test_sync_tool_call_inference(sync_client):
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
    assert result.variant_name == "variant"
    assert isinstance(result, ChatInferenceResponse)
    content = result.content
    assert len(content) == 1
    assert content[0].type == "tool_call"
    assert content[0].raw_name == "get_temperature"
    assert content[0].id == "0"
    assert content[0].raw_arguments == '{"location":"Brooklyn","units":"celsius"}'
    assert content[0].name == "get_temperature"
    assert content[0].arguments == {"location": "Brooklyn", "units": "celsius"}
    usage = result.usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 10


def test_sync_malformed_tool_call_inference(sync_client):
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
    assert result.variant_name == "bad_tool"
    assert isinstance(result, ChatInferenceResponse)
    content = result.content
    assert len(content) == 1
    assert content[0].type == "tool_call"
    assert content[0].raw_name == "get_temperature"
    assert content[0].id == "0"
    assert content[0].raw_arguments == '{"location":"Brooklyn","units":"Celsius"}'
    assert content[0].name == "get_temperature"
    assert content[0].arguments is None
    usage = result.usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 10


def test_sync_tool_call_streaming(sync_client):
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
        if i + 1 < len(chunks):
            assert len(chunk.content) == 1
            assert chunk.content[0].type == "tool_call"
            assert chunk.content[0].raw_name == "get_temperature"
            assert chunk.content[0].id == "0"
            assert chunk.content[0].raw_arguments == expected_text[i]
        else:
            assert len(chunk.content) == 0
            assert chunk.usage.input_tokens == 10
            assert chunk.usage.output_tokens == 5


def test_sync_json_streaming(sync_client):
    # We don't actually have a streaming JSON function implemented in `dummy.rs` but it doesn't matter for this test since
    # TensorZero doesn't parse the JSON output of the function for streaming calls.
    stream = sync_client.inference(
        function_name="json_success",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": {"country": "Japan"}}],
        },
        stream=True,
    )
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
        if i + 1 < len(chunks):
            assert chunk.raw == expected_text[i]
        else:
            assert chunk.usage.input_tokens == 10
            assert chunk.usage.output_tokens == 16


def test_sync_json_success(sync_client):
    result = sync_client.inference(
        function_name="json_success",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": {"country": "Japan"}}],
        },
        stream=False,
    )
    assert result.variant_name == "test"
    assert isinstance(result, JsonInferenceResponse)
    assert result.output.raw == '{"answer":"Hello"}'
    assert result.output.parsed == {"answer": "Hello"}
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 10


def test_sync_json_failure(sync_client):
    result = sync_client.inference(
        function_name="json_fail",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello, world!"}],
        },
        stream=False,
    )
    assert result.variant_name == "test"
    assert isinstance(result, JsonInferenceResponse)
    assert (
        result.output.raw
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    assert result.output.parsed is None
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 10


def test_sync_feedback(sync_client):
    result = sync_client.feedback(
        metric_name="user_rating", value=5, episode_id=uuid7()
    )
    assert isinstance(result, FeedbackResponse)

    result = sync_client.feedback(
        metric_name="task_success", value=True, inference_id=uuid7()
    )
    assert isinstance(result, FeedbackResponse)
    result = sync_client.inference(
        function_name="basic_test",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    # Wait for the inference to be created in ClickHouse
    sleep(1)

    result = sync_client.feedback(
        metric_name="demonstration",
        value="hi how are you",
        inference_id=result.inference_id,
        tags={"author": "Alice"},
    )
    assert isinstance(result, FeedbackResponse)


def test_sync_feedback_invalid_input(sync_client):
    with pytest.raises(ValueError):
        sync_client.feedback(metric_name="test_metric", value=5)

    with pytest.raises(ValueError):
        sync_client.feedback(
            metric_name="test_metric",
            value=5,
            episode_id=UUID("123e4567-e89b-12d3-a456-426614174000"),
            inference_id=UUID("123e4567-e89b-12d3-a456-426614174001"),
        )


def test_sync_tensorzero_error(sync_client):
    with pytest.raises(TensorZeroError) as excinfo:
        sync_client.inference(function_name="not_a_function", input={"messages": []})

    assert (
        str(excinfo.value)
        == 'TensorZeroError (status code 404): {"error":"Unknown function: not_a_function"}'
    )


def test_sync_basic_inference_with_content_block(sync_client):
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
                            name="test",
                            raw_arguments={"arg": "value"},
                            raw_name="test_tool",
                            arguments={"arg": "value"},
                        ),
                        ToolResult(
                            name="test",
                            result="success",
                            id="1",
                        ),
                    ],
                }
            ],
        },
    )
    assert result.variant_name == "test"
    assert isinstance(result, ChatInferenceResponse)
    content = result.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert (
        content[0].text
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    usage = result.usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 10
