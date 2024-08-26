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

from uuid import UUID

import pytest
import pytest_asyncio
from tensorzero import (
    AsyncTensorZero,
    FeedbackResponse,
    Text,
    TextChunk,
    ToolCall,
    ToolCallChunk,
)
from uuid_extensions import uuid7


@pytest_asyncio.fixture
async def client():
    async with AsyncTensorZero("http://localhost:3000") as client:
        yield client


@pytest.mark.asyncio
async def test_basic_inference(client):
    result = await client.inference(
        function_name="basic_test",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert result.variant_name == "test"
    output = result.output
    assert len(output) == 1
    assert isinstance(output[0], Text)
    assert (
        output[0].text
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    usage = result.usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 10


@pytest.mark.asyncio
async def test_inference_streaming(client):
    stream = await client.inference(
        function_name="basic_test",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello"}],
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
            assert len(chunk.content) == 1
            assert isinstance(chunk.content[0], TextChunk)
            assert chunk.content[0].text == expected_text[i]
        else:
            assert len(chunk.content) == 0
            assert chunk.usage.input_tokens == 10
            assert chunk.usage.output_tokens == 16


@pytest.mark.asyncio
async def test_tool_call_inference(client):
    result = await client.inference(
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
    output = result.output
    assert len(output) == 1
    assert isinstance(output[0], ToolCall)
    assert output[0].name == "get_temperature"
    assert output[0].id == "0"
    assert output[0].arguments == '{"location":"Brooklyn","units":"celsius"}'
    assert output[0].parsed_name == "get_temperature"
    assert output[0].parsed_arguments == {"location": "Brooklyn", "units": "celsius"}
    usage = result.usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 10


@pytest.mark.asyncio
async def test_malformed_tool_call_inference(client):
    result = await client.inference(
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
    output = result.output
    assert len(output) == 1
    assert isinstance(output[0], ToolCall)
    assert output[0].name == "get_temperature"
    assert output[0].id == "0"
    assert output[0].arguments == '{"location":"Brooklyn","units":"Celsius"}'
    assert output[0].parsed_name == "get_temperature"
    assert output[0].parsed_arguments is None
    usage = result.usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 10


@pytest.mark.asyncio
async def test_tool_call_streaming(client):
    stream = await client.inference(
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
            assert isinstance(chunk.content[0], ToolCallChunk)
            assert chunk.content[0].name == "get_temperature"
            assert chunk.content[0].id == "0"
            assert chunk.content[0].arguments == expected_text[i]
        else:
            assert len(chunk.content) == 0
            assert chunk.usage.input_tokens == 10
            assert chunk.usage.output_tokens == 5


@pytest.mark.asyncio
async def test_json_streaming(client):
    # We don't actually have a streaming JSON function implemented in `dummy.rs` but it doesn't matter for this test since
    # TensorZero doesn't parse the JSON output of the function for streaming calls.
    stream = await client.inference(
        function_name="json_success",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello, world!"}],
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
async def test_json_sucess(client):
    result = await client.inference(
        function_name="json_success",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello, world!"}],
        },
        stream=False,
    )
    assert result.variant_name == "test"
    assert result.output.raw == '{"answer":"Hello"}'
    assert result.output.parsed == {"answer": "Hello"}
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 10


@pytest.mark.asyncio
async def test_json_failure(client):
    result = await client.inference(
        function_name="json_fail",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello, world!"}],
        },
        stream=False,
    )
    assert result.variant_name == "test"
    assert (
        result.output.raw
        == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    )
    assert result.output.parsed is None
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 10


@pytest.mark.asyncio
async def test_feedback(client):
    result = await client.feedback(
        metric_name="user_rating", value=5, episode_id=uuid7()
    )
    assert isinstance(result, FeedbackResponse)

    result = await client.feedback(
        metric_name="task_success", value=True, inference_id=uuid7()
    )
    assert isinstance(result, FeedbackResponse)

    result = await client.feedback(
        metric_name="demonstration", value="hi how are you", inference_id=uuid7()
    )
    assert isinstance(result, FeedbackResponse)


@pytest.mark.asyncio
async def test_feedback_invalid_input(client):
    with pytest.raises(ValueError):
        await client.feedback(metric_name="test_metric", value=5)

    with pytest.raises(ValueError):
        await client.feedback(
            metric_name="test_metric",
            value=5,
            episode_id=UUID("123e4567-e89b-12d3-a456-426614174000"),
            inference_id=UUID("123e4567-e89b-12d3-a456-426614174001"),
        )


# Add more tests as needed for edge cases, error handling, etc.
