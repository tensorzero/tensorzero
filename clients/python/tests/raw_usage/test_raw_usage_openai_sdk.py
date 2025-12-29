"""
Tests for tensorzero::include_raw_usage parameter using the OpenAI Python SDK.

These tests verify that raw provider-specific usage data is correctly returned
when tensorzero::include_raw_usage is set to True via the OpenAI-compatible API.
"""

from typing import List

import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from uuid_utils.compat import uuid7


@pytest.mark.asyncio
async def test_async_raw_usage_non_streaming(async_openai_client: AsyncOpenAI):
    """Test that tensorzero::include_raw_usage returns tensorzero_raw_usage in non-streaming response."""
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "tensorzero::template",
                    "name": "system",
                    "arguments": {"assistant_name": "Alfred Pennyworth"},
                }
            ],
        },
        {"role": "user", "content": "Hello"},
    ]

    result = await async_openai_client.chat.completions.create(
        extra_body={
            "tensorzero::episode_id": str(uuid7()),
            "tensorzero::include_raw_usage": True,
        },
        messages=messages,  # type: ignore
        model="tensorzero::function_name::basic_test",
    )

    assert result.usage is not None, "Response should have usage"
    assert hasattr(result.usage, "tensorzero_raw_usage"), "usage should have tensorzero_raw_usage when requested"
    assert result.usage.tensorzero_raw_usage is not None, "tensorzero_raw_usage should not be None"  # type: ignore
    assert isinstance(result.usage.tensorzero_raw_usage, list), "tensorzero_raw_usage should be a list"  # type: ignore
    assert len(result.usage.tensorzero_raw_usage) > 0, "tensorzero_raw_usage should have at least one entry"  # type: ignore

    # Verify structure of first entry
    entry = result.usage.tensorzero_raw_usage[0]  # type: ignore
    assert "model_inference_id" in entry, "Entry should have model_inference_id"
    assert "provider_type" in entry, "Entry should have provider_type"
    assert "api_type" in entry, "Entry should have api_type"


@pytest.mark.asyncio
async def test_async_raw_usage_not_requested(async_openai_client: AsyncOpenAI):
    """Test that tensorzero_raw_usage is not present when tensorzero::include_raw_usage is False."""
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "tensorzero::template",
                    "name": "system",
                    "arguments": {"assistant_name": "Alfred Pennyworth"},
                }
            ],
        },
        {"role": "user", "content": "Hello"},
    ]

    result = await async_openai_client.chat.completions.create(
        extra_body={
            "tensorzero::episode_id": str(uuid7()),
            "tensorzero::include_raw_usage": False,
        },
        messages=messages,  # type: ignore
        model="tensorzero::function_name::basic_test",
    )

    assert result.usage is not None, "Response should have usage"
    # tensorzero_raw_usage should not be present or should be None
    raw_usage = getattr(result.usage, "tensorzero_raw_usage", None)
    assert raw_usage is None, "tensorzero_raw_usage should be None when not requested"


@pytest.mark.asyncio
async def test_async_raw_usage_streaming(async_openai_client: AsyncOpenAI):
    """Test that tensorzero::include_raw_usage returns tensorzero_raw_usage in streaming response."""
    messages: List[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": [
                {
                    "type": "tensorzero::template",
                    "name": "system",
                    "arguments": {"assistant_name": "Alfred Pennyworth"},
                }
            ],
        },
        {"role": "user", "content": "Hello"},
    ]

    # Note: tensorzero::include_raw_usage automatically enables include_usage for streaming
    stream = await async_openai_client.chat.completions.create(
        extra_body={
            "tensorzero::episode_id": str(uuid7()),
            "tensorzero::include_raw_usage": True,
        },
        messages=messages,
        model="tensorzero::function_name::basic_test",
        stream=True,
    )

    found_raw_usage = False
    async for chunk in stream:
        # Check if this chunk has usage with tensorzero_raw_usage
        if chunk.usage is not None:
            raw_usage = getattr(chunk.usage, "tensorzero_raw_usage", None)
            if raw_usage is not None:
                found_raw_usage = True
                assert isinstance(raw_usage, list), "tensorzero_raw_usage should be a list"
                assert len(raw_usage) > 0, "tensorzero_raw_usage should have at least one entry"  # type: ignore

                entry = raw_usage[0]  # type: ignore
                assert "model_inference_id" in entry, "Entry should have model_inference_id"
                assert "provider_type" in entry, "Entry should have provider_type"
                assert "api_type" in entry, "Entry should have api_type"

    assert found_raw_usage, "Streaming response should include tensorzero_raw_usage in final chunk"
