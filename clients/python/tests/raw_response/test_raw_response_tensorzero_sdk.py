"""
Tests for include_raw_response parameter in the TensorZero Python SDK.

These tests verify that raw provider-specific response data is correctly returned
when include_raw_response is set to True.
"""

from typing import AsyncIterator, Iterator

import pytest
from tensorzero import (
    AsyncTensorZeroGateway,
    ChatInferenceResponse,
    RawResponseEntry,
    TensorZeroGateway,
    Text,
)
from uuid_utils import uuid7


def assert_raw_response_entry_structure(entry: RawResponseEntry) -> None:
    """Verify the structure of a RawResponseEntry."""
    assert entry.model_inference_id is not None, "Entry should have model_inference_id"
    assert entry.provider_type is not None, "Entry should have provider_type"
    assert isinstance(entry.provider_type, str), "provider_type should be a string"
    assert entry.api_type is not None, "Entry should have api_type"
    assert entry.api_type in ("chat_completions", "responses", "embeddings"), (
        f"api_type should be 'chat_completions', 'responses', or 'embeddings', got {entry.api_type}"
    )
    assert entry.data is not None, "Entry should have data"
    assert isinstance(entry.data, str), "data should be a string (raw response from provider)"


@pytest.mark.asyncio
async def test_async_raw_response_non_streaming(async_client: AsyncTensorZeroGateway):
    """Test that include_raw_response returns raw_response in non-streaming response."""
    input_data = {
        "messages": [{"role": "user", "content": [Text(type="text", text="Hello")]}],
    }

    result = await async_client.inference(
        model_name="gpt-4o-mini-2024-07-18",
        input=input_data,
        episode_id=uuid7(),
        include_raw_response=True,
    )

    assert isinstance(result, ChatInferenceResponse), "Response should be ChatInferenceResponse"
    # raw_response is at response level
    assert result.raw_response is not None, "Response should have raw_response when requested"
    assert isinstance(result.raw_response, list), "raw_response should be a list"
    assert len(result.raw_response) > 0, "raw_response should have at least one entry"

    # Verify structure of first entry
    entry = result.raw_response[0]
    assert_raw_response_entry_structure(entry)


@pytest.mark.asyncio
async def test_async_raw_response_not_requested(async_client: AsyncTensorZeroGateway):
    """Test that raw_response is not present when include_raw_response is False."""
    input_data = {
        "messages": [{"role": "user", "content": [Text(type="text", text="Hello")]}],
    }

    result = await async_client.inference(
        model_name="gpt-4o-mini-2024-07-18",
        input=input_data,
        episode_id=uuid7(),
        include_raw_response=False,
    )

    assert isinstance(result, ChatInferenceResponse), "Response should be ChatInferenceResponse"
    # raw_response should be None when not requested
    assert result.raw_response is None, "raw_response should be None when not requested"


@pytest.mark.asyncio
async def test_async_raw_response_streaming(async_client: AsyncTensorZeroGateway):
    """Test that include_raw_response returns raw_chunk in streaming response."""
    input_data = {
        "messages": [{"role": "user", "content": [Text(type="text", text="Hello")]}],
    }

    stream = await async_client.inference(
        model_name="gpt-4o-mini-2024-07-18",
        input=input_data,
        episode_id=uuid7(),
        stream=True,
        include_raw_response=True,
    )
    assert isinstance(stream, AsyncIterator)

    found_raw_chunk = False
    async for chunk in stream:
        # Check if this chunk has raw_chunk (the raw response data for the current chunk)
        if chunk.raw_chunk is not None:
            found_raw_chunk = True
            assert isinstance(chunk.raw_chunk, str), "raw_chunk should be a string"

        # For single inference, raw_response (array of previous inferences) should be None
        # because there are no previous model inferences
        # Note: raw_response in streaming is for previous model inferences (e.g., in best-of-n)

    assert found_raw_chunk, "Streaming response should include raw_chunk in at least one chunk"


@pytest.mark.asyncio
async def test_async_raw_response_streaming_not_requested(async_client: AsyncTensorZeroGateway):
    """Test that raw_chunk is not present when include_raw_response is False in streaming."""
    input_data = {
        "messages": [{"role": "user", "content": [Text(type="text", text="Hello")]}],
    }

    stream = await async_client.inference(
        model_name="gpt-4o-mini-2024-07-18",
        input=input_data,
        episode_id=uuid7(),
        stream=True,
        include_raw_response=False,
    )
    assert isinstance(stream, AsyncIterator)

    async for chunk in stream:
        # raw_chunk should be None when not requested
        assert chunk.raw_chunk is None, "raw_chunk should be None when not requested"


def test_sync_raw_response_non_streaming(sync_client: TensorZeroGateway):
    """Test that include_raw_response returns raw_response in sync non-streaming response."""
    input_data = {
        "messages": [{"role": "user", "content": [Text(type="text", text="Hello")]}],
    }

    result = sync_client.inference(
        model_name="gpt-4o-mini-2024-07-18",
        input=input_data,
        episode_id=uuid7(),
        include_raw_response=True,
    )

    assert isinstance(result, ChatInferenceResponse), "Response should be ChatInferenceResponse"
    # raw_response is at response level
    assert result.raw_response is not None, "Response should have raw_response when requested"
    assert isinstance(result.raw_response, list), "raw_response should be a list"
    assert len(result.raw_response) > 0, "raw_response should have at least one entry"

    # Verify structure of first entry
    entry = result.raw_response[0]
    assert_raw_response_entry_structure(entry)


def test_sync_raw_response_streaming(sync_client: TensorZeroGateway):
    """Test that include_raw_response returns raw_chunk in sync streaming response."""
    input_data = {
        "messages": [{"role": "user", "content": [Text(type="text", text="Hello")]}],
    }

    stream = sync_client.inference(
        model_name="gpt-4o-mini-2024-07-18",
        input=input_data,
        episode_id=uuid7(),
        stream=True,
        include_raw_response=True,
    )
    assert isinstance(stream, Iterator)

    found_raw_chunk = False
    for chunk in stream:
        # Check if this chunk has raw_chunk (the raw response data for the current chunk)
        if chunk.raw_chunk is not None:
            found_raw_chunk = True
            assert isinstance(chunk.raw_chunk, str), "raw_chunk should be a string"

    assert found_raw_chunk, "Streaming response should include raw_chunk in at least one chunk"


def test_sync_raw_response_streaming_not_requested(sync_client: TensorZeroGateway):
    """Test that raw_chunk is not present when include_raw_response is False in sync streaming."""
    input_data = {
        "messages": [{"role": "user", "content": [Text(type="text", text="Hello")]}],
    }

    stream = sync_client.inference(
        model_name="gpt-4o-mini-2024-07-18",
        input=input_data,
        episode_id=uuid7(),
        stream=True,
        include_raw_response=False,
    )
    assert isinstance(stream, Iterator)

    for chunk in stream:
        # raw_chunk should be None when not requested
        assert chunk.raw_chunk is None, "raw_chunk should be None when not requested"
