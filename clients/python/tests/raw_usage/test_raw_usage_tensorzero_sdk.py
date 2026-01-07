"""
Tests for include_raw_usage parameter in the TensorZero Python SDK.

These tests verify that raw provider-specific usage data is correctly returned
when include_raw_usage is set to True.
"""

from typing import AsyncIterator, Iterator

import pytest
from tensorzero import (
    AsyncTensorZeroGateway,
    ChatInferenceResponse,
    RawUsageEntry,
    TensorZeroGateway,
    Text,
)
from uuid_utils import uuid7


def assert_openai_chat_usage_details(entry: RawUsageEntry) -> None:
    data = entry.data
    assert data is not None, "raw_usage entry should include data for chat completions"
    assert isinstance(data, dict), "raw_usage entry data should be a dict for chat completions"
    assert "total_tokens" in data, "raw_usage should include `total_tokens` for chat completions"
    prompt_details = data.get("prompt_tokens_details")
    assert isinstance(prompt_details, dict), "raw_usage should include `prompt_tokens_details` for chat completions"
    assert "cached_tokens" in prompt_details, (
        "raw_usage should include `prompt_tokens_details.cached_tokens` for chat completions"
    )
    completion_details = data.get("completion_tokens_details")
    assert isinstance(completion_details, dict), (
        "raw_usage should include `completion_tokens_details` for chat completions"
    )
    assert "reasoning_tokens" in completion_details, (
        "raw_usage should include `completion_tokens_details.reasoning_tokens` for chat completions"
    )


@pytest.mark.asyncio
async def test_async_raw_usage_non_streaming(async_client: AsyncTensorZeroGateway):
    """Test that include_raw_usage returns raw_usage in non-streaming response."""
    input_data = {
        "messages": [{"role": "user", "content": [Text(type="text", text="Hello")]}],
    }

    result = await async_client.inference(
        model_name="gpt-4o-mini-2024-07-18",
        input=input_data,
        episode_id=uuid7(),
        include_raw_usage=True,
    )

    assert isinstance(result, ChatInferenceResponse), "Response should be ChatInferenceResponse"
    assert result.usage is not None, "Response should have usage"
    # raw_usage is now at response level (sibling to usage)
    assert result.raw_usage is not None, "Response should have raw_usage when requested"
    assert isinstance(result.raw_usage, list), "raw_usage should be a list"
    assert len(result.raw_usage) > 0, "raw_usage should have at least one entry"

    # Verify structure of first entry
    entry = result.raw_usage[0]
    assert entry.model_inference_id is not None, "Entry should have model_inference_id"
    assert entry.provider_type is not None, "Entry should have provider_type"
    assert entry.api_type is not None, "Entry should have api_type"
    assert_openai_chat_usage_details(entry)


@pytest.mark.asyncio
async def test_async_raw_usage_not_requested(async_client: AsyncTensorZeroGateway):
    """Test that raw_usage is not present when include_raw_usage is False."""
    input_data = {
        "messages": [{"role": "user", "content": [Text(type="text", text="Hello")]}],
    }

    result = await async_client.inference(
        model_name="gpt-4o-mini-2024-07-18",
        input=input_data,
        episode_id=uuid7(),
        include_raw_usage=False,
    )

    assert isinstance(result, ChatInferenceResponse), "Response should be ChatInferenceResponse"
    assert result.usage is not None, "Response should have usage"
    # raw_usage is now at response level (sibling to usage)
    assert result.raw_usage is None, "raw_usage should be None when not requested"


@pytest.mark.asyncio
async def test_async_raw_usage_streaming(async_client: AsyncTensorZeroGateway):
    """Test that include_raw_usage returns raw_usage in streaming response."""
    input_data = {
        "messages": [{"role": "user", "content": [Text(type="text", text="Hello")]}],
    }

    stream = await async_client.inference(
        model_name="gpt-4o-mini-2024-07-18",
        input=input_data,
        episode_id=uuid7(),
        stream=True,
        include_raw_usage=True,
    )
    assert isinstance(stream, AsyncIterator)

    found_raw_usage = False
    async for chunk in stream:
        # Check if this chunk has raw_usage at chunk level (sibling to usage)
        if chunk.raw_usage is not None:
            found_raw_usage = True
            assert isinstance(chunk.raw_usage, list), "raw_usage should be a list"
            assert len(chunk.raw_usage) > 0, "raw_usage should have at least one entry"

            entry = chunk.raw_usage[0]
            assert entry.model_inference_id is not None, "Entry should have model_inference_id"
            assert entry.provider_type is not None, "Entry should have provider_type"
            assert entry.api_type is not None, "Entry should have api_type"
            assert_openai_chat_usage_details(entry)

    assert found_raw_usage, "Streaming response should include raw_usage in final chunk"


def test_sync_raw_usage_non_streaming(sync_client: TensorZeroGateway):
    """Test that include_raw_usage returns raw_usage in sync non-streaming response."""
    input_data = {
        "messages": [{"role": "user", "content": [Text(type="text", text="Hello")]}],
    }

    result = sync_client.inference(
        model_name="gpt-4o-mini-2024-07-18",
        input=input_data,
        episode_id=uuid7(),
        include_raw_usage=True,
    )

    assert isinstance(result, ChatInferenceResponse), "Response should be ChatInferenceResponse"
    assert result.usage is not None, "Response should have usage"
    # raw_usage is now at response level (sibling to usage)
    assert result.raw_usage is not None, "Response should have raw_usage when requested"
    assert isinstance(result.raw_usage, list), "raw_usage should be a list"
    assert len(result.raw_usage) > 0, "raw_usage should have at least one entry"

    # Verify structure of first entry
    entry = result.raw_usage[0]
    assert entry.model_inference_id is not None, "Entry should have model_inference_id"
    assert entry.provider_type is not None, "Entry should have provider_type"
    assert entry.api_type is not None, "Entry should have api_type"
    assert_openai_chat_usage_details(entry)


def test_sync_raw_usage_streaming(sync_client: TensorZeroGateway):
    """Test that include_raw_usage returns raw_usage in sync streaming response."""
    input_data = {
        "messages": [{"role": "user", "content": [Text(type="text", text="Hello")]}],
    }

    stream = sync_client.inference(
        model_name="gpt-4o-mini-2024-07-18",
        input=input_data,
        episode_id=uuid7(),
        stream=True,
        include_raw_usage=True,
    )
    assert isinstance(stream, Iterator)

    found_raw_usage = False
    for chunk in stream:
        # Check if this chunk has raw_usage at chunk level (sibling to usage)
        if chunk.raw_usage is not None:
            found_raw_usage = True
            assert isinstance(chunk.raw_usage, list), "raw_usage should be a list"
            assert len(chunk.raw_usage) > 0, "raw_usage should have at least one entry"

            entry = chunk.raw_usage[0]
            assert entry.model_inference_id is not None, "Entry should have model_inference_id"
            assert entry.provider_type is not None, "Entry should have provider_type"
            assert entry.api_type is not None, "Entry should have api_type"
            assert_openai_chat_usage_details(entry)

    assert found_raw_usage, "Streaming response should include raw_usage in final chunk"
