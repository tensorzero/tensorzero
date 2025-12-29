"""
Tests for include_raw_usage parameter in the TensorZero Python SDK.

These tests verify that raw provider-specific usage data is correctly returned
when include_raw_usage is set to True.
"""

import pytest
from tensorzero import (
    AsyncTensorZeroGateway,
    ChatInferenceResponse,
    TensorZeroGateway,
    Text,
)
from uuid_utils import uuid7


@pytest.mark.asyncio
async def test_async_raw_usage_non_streaming(async_client: AsyncTensorZeroGateway):
    """Test that include_raw_usage returns raw_usage in non-streaming response."""
    input_data = {
        "system": {"assistant_name": "Alfred Pennyworth"},
        "messages": [{"role": "user", "content": [Text(type="text", text="Hello")]}],
    }

    result = await async_client.inference(
        function_name="basic_test",
        input=input_data,
        episode_id=uuid7(),
        include_raw_usage=True,
    )

    assert isinstance(result, ChatInferenceResponse), "Response should be ChatInferenceResponse"
    assert result.usage is not None, "Response should have usage"
    assert result.usage.raw_usage is not None, "usage should have raw_usage when requested"
    assert isinstance(result.usage.raw_usage, list), "raw_usage should be a list"
    assert len(result.usage.raw_usage) > 0, "raw_usage should have at least one entry"

    # Verify structure of first entry
    entry = result.usage.raw_usage[0]
    assert entry.model_inference_id is not None, "Entry should have model_inference_id"
    assert entry.provider_type is not None, "Entry should have provider_type"
    assert entry.api_type is not None, "Entry should have api_type"


@pytest.mark.asyncio
async def test_async_raw_usage_not_requested(async_client: AsyncTensorZeroGateway):
    """Test that raw_usage is not present when include_raw_usage is False."""
    input_data = {
        "system": {"assistant_name": "Alfred Pennyworth"},
        "messages": [{"role": "user", "content": [Text(type="text", text="Hello")]}],
    }

    result = await async_client.inference(
        function_name="basic_test",
        input=input_data,
        episode_id=uuid7(),
        include_raw_usage=False,
    )

    assert isinstance(result, ChatInferenceResponse), "Response should be ChatInferenceResponse"
    assert result.usage is not None, "Response should have usage"
    assert result.usage.raw_usage is None, "raw_usage should be None when not requested"


@pytest.mark.asyncio
async def test_async_raw_usage_streaming(async_client: AsyncTensorZeroGateway):
    """Test that include_raw_usage returns raw_usage in streaming response."""
    input_data = {
        "system": {"assistant_name": "Alfred Pennyworth"},
        "messages": [{"role": "user", "content": [Text(type="text", text="Hello")]}],
    }

    stream = await async_client.inference(
        function_name="basic_test",
        input=input_data,
        episode_id=uuid7(),
        stream=True,
        include_raw_usage=True,
    )

    found_raw_usage = False
    async for chunk in stream:
        # Check if this chunk has usage with raw_usage
        if chunk.usage is not None and chunk.usage.raw_usage is not None:
            found_raw_usage = True
            assert isinstance(chunk.usage.raw_usage, list), "raw_usage should be a list"
            assert len(chunk.usage.raw_usage) > 0, "raw_usage should have at least one entry"

            entry = chunk.usage.raw_usage[0]
            assert entry.model_inference_id is not None, "Entry should have model_inference_id"
            assert entry.provider_type is not None, "Entry should have provider_type"
            assert entry.api_type is not None, "Entry should have api_type"

    assert found_raw_usage, "Streaming response should include raw_usage in final chunk"


def test_sync_raw_usage_non_streaming(sync_client: TensorZeroGateway):
    """Test that include_raw_usage returns raw_usage in sync non-streaming response."""
    input_data = {
        "system": {"assistant_name": "Alfred Pennyworth"},
        "messages": [{"role": "user", "content": [Text(type="text", text="Hello")]}],
    }

    result = sync_client.inference(
        function_name="basic_test",
        input=input_data,
        episode_id=uuid7(),
        include_raw_usage=True,
    )

    assert isinstance(result, ChatInferenceResponse), "Response should be ChatInferenceResponse"
    assert result.usage is not None, "Response should have usage"
    assert result.usage.raw_usage is not None, "usage should have raw_usage when requested"
    assert isinstance(result.usage.raw_usage, list), "raw_usage should be a list"
    assert len(result.usage.raw_usage) > 0, "raw_usage should have at least one entry"

    # Verify structure of first entry
    entry = result.usage.raw_usage[0]
    assert entry.model_inference_id is not None, "Entry should have model_inference_id"
    assert entry.provider_type is not None, "Entry should have provider_type"
    assert entry.api_type is not None, "Entry should have api_type"


def test_sync_raw_usage_streaming(sync_client: TensorZeroGateway):
    """Test that include_raw_usage returns raw_usage in sync streaming response."""
    input_data = {
        "system": {"assistant_name": "Alfred Pennyworth"},
        "messages": [{"role": "user", "content": [Text(type="text", text="Hello")]}],
    }

    stream = sync_client.inference(
        function_name="basic_test",
        input=input_data,
        episode_id=uuid7(),
        stream=True,
        include_raw_usage=True,
    )

    found_raw_usage = False
    for chunk in stream:
        # Check if this chunk has usage with raw_usage
        if chunk.usage is not None and chunk.usage.raw_usage is not None:
            found_raw_usage = True
            assert isinstance(chunk.usage.raw_usage, list), "raw_usage should be a list"
            assert len(chunk.usage.raw_usage) > 0, "raw_usage should have at least one entry"

            entry = chunk.usage.raw_usage[0]
            assert entry.model_inference_id is not None, "Entry should have model_inference_id"
            assert entry.provider_type is not None, "Entry should have provider_type"
            assert entry.api_type is not None, "Entry should have api_type"

    assert found_raw_usage, "Streaming response should include raw_usage in final chunk"
