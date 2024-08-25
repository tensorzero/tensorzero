import pytest
import asyncio
from uuid import UUID
from uuid_extensions import uuid7
from typing import Dict, Any

import pytest_asyncio
from tensorzero import TensorZeroClient, ChatInferenceResponse, ContentBlock, Text, TextChunk


"""
TODOs:
 - [ ] Write top-level documentation for the client and for the tests
 - [ ] Add a set of dataclasses for output types
 - [ ] Add tests covering function calling and JSON functions
 - [ ] Add a description to the pyproject.toml
"""


@pytest_asyncio.fixture
async def client():
    async with TensorZeroClient("http://localhost:3000") as client:
        yield client

@pytest.mark.asyncio
async def test_basic_inference(client):
    result = await client.inference(
        function_name="basic_test",
        input={"system": {"assistant_name": "Alfred Pennyworth"}, "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert result.variant_name == "test"
    output = result.output
    assert len(output) == 1
    assert isinstance(output[0], Text)
    assert output[0].text == "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    usage = result.usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 10


@pytest.mark.asyncio
async def test_inference_streaming(client):
    stream = await client.inference(
        function_name="basic_test",
        input={"system": {"assistant_name": "Alfred Pennyworth"}, "messages": [{"role": "user", "content": "Hello"}]},
        stream=True
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
        " pizza."
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
async def test_feedback(client):

    result = await client.feedback(
        metric_name="user_rating",
        value=5,
        episode_id=uuid7()
    )
    assert len(result) == 1
    feedback_id = UUID(result["feedback_id"])

    result = await client.feedback(
        metric_name="task_success",
        value=True,
        inference_id=uuid7()
    )
    assert len(result) == 1
    feedback_id = UUID(result["feedback_id"])

    result = await client.feedback(
        metric_name="demonstration",
        value="hi how are you",
        inference_id=uuid7()
    )
    assert len(result) == 1
    feedback_id = UUID(result["feedback_id"])



@pytest.mark.asyncio
async def test_feedback_invalid_input(client):
    with pytest.raises(ValueError):
        await client.feedback(
            metric_name="test_metric",
            value=5
        )

    with pytest.raises(ValueError):
        await client.feedback(
            metric_name="test_metric",
            value=5,
            episode_id=UUID("123e4567-e89b-12d3-a456-426614174000"),
            inference_id=UUID("123e4567-e89b-12d3-a456-426614174001")
        )

# Add more tests as needed for edge cases, error handling, etc.
