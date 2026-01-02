"""
Tests for tensorzero::include_raw_usage parameter using the OpenAI Python SDK.

These tests verify that raw provider-specific usage data is correctly returned
when tensorzero::include_raw_usage is set to True via the OpenAI-compatible API.
"""

from typing import Any, List, cast

import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from uuid_utils.compat import uuid7


def assert_openai_chat_usage_details(entry: dict[str, Any]) -> None:
    data: dict[str, Any] | None = entry.get("data")
    assert data is not None, "raw_usage entry should include data for chat completions"
    assert isinstance(data, dict), "raw_usage entry data should be a dict for chat completions"
    assert "total_tokens" in data, "raw_usage should include `total_tokens` for chat completions"
    prompt_details: dict[str, Any] | None = data.get("prompt_tokens_details")
    assert isinstance(prompt_details, dict), "raw_usage should include `prompt_tokens_details` for chat completions"
    assert "cached_tokens" in prompt_details, (
        "raw_usage should include `prompt_tokens_details.cached_tokens` for chat completions"
    )
    completion_details: dict[str, Any] | None = data.get("completion_tokens_details")
    assert isinstance(completion_details, dict), (
        "raw_usage should include `completion_tokens_details` for chat completions"
    )
    assert "reasoning_tokens" in completion_details, (
        "raw_usage should include `completion_tokens_details.reasoning_tokens` for chat completions"
    )


@pytest.mark.asyncio
async def test_async_raw_usage_non_streaming(async_openai_client: AsyncOpenAI):
    """Test that tensorzero::include_raw_usage returns tensorzero_raw_usage in non-streaming response."""
    messages: List[ChatCompletionMessageParam] = [
        {"role": "user", "content": "Hello"},
    ]

    result = await async_openai_client.chat.completions.create(
        extra_body={
            "tensorzero::episode_id": str(uuid7()),
            "tensorzero::include_raw_usage": True,
        },
        messages=messages,
        model="tensorzero::model_name::gpt-4o-mini-2024-07-18",
    )

    assert result.usage is not None, "Response should have usage"
    # tensorzero_raw_usage is now at response level (sibling to usage)
    assert hasattr(result, "tensorzero_raw_usage"), "Response should have tensorzero_raw_usage when requested"
    assert result.tensorzero_raw_usage is not None, "tensorzero_raw_usage should not be None"  # type: ignore
    assert isinstance(result.tensorzero_raw_usage, list), "tensorzero_raw_usage should be a list"  # type: ignore
    assert len(result.tensorzero_raw_usage) > 0, "tensorzero_raw_usage should have at least one entry"  # type: ignore

    # Verify structure of first entry
    entry = cast(dict[str, Any], result.tensorzero_raw_usage[0])  # type: ignore[attr-defined]
    assert "model_inference_id" in entry, "Entry should have model_inference_id"
    assert "provider_type" in entry, "Entry should have provider_type"
    assert "api_type" in entry, "Entry should have api_type"
    assert_openai_chat_usage_details(entry)


@pytest.mark.asyncio
async def test_async_raw_usage_not_requested(async_openai_client: AsyncOpenAI):
    """Test that tensorzero_raw_usage is not present when tensorzero::include_raw_usage is False."""
    messages: List[ChatCompletionMessageParam] = [
        {"role": "user", "content": "Hello"},
    ]

    result = await async_openai_client.chat.completions.create(
        extra_body={
            "tensorzero::episode_id": str(uuid7()),
            "tensorzero::include_raw_usage": False,
        },
        messages=messages,
        model="tensorzero::model_name::gpt-4o-mini-2024-07-18",
    )

    assert result.usage is not None, "Response should have usage"
    # tensorzero_raw_usage is at response level and should not be present when not requested
    raw_usage = getattr(result, "tensorzero_raw_usage", None)
    assert raw_usage is None, "tensorzero_raw_usage should be None when not requested"


@pytest.mark.asyncio
async def test_async_raw_usage_streaming(async_openai_client: AsyncOpenAI):
    """Test that tensorzero::include_raw_usage returns tensorzero_raw_usage in streaming response."""
    messages: List[ChatCompletionMessageParam] = [
        {"role": "user", "content": "Hello"},
    ]

    # Note: tensorzero::include_raw_usage automatically enables include_usage for streaming
    stream = await async_openai_client.chat.completions.create(
        extra_body={
            "tensorzero::episode_id": str(uuid7()),
            "tensorzero::include_raw_usage": True,
        },
        messages=messages,
        model="tensorzero::model_name::gpt-4o-mini-2024-07-18",
        stream=True,
    )

    found_raw_usage = False
    async for chunk in stream:
        # Check if this chunk has tensorzero_raw_usage at chunk level (sibling to usage)
        raw_usage = getattr(chunk, "tensorzero_raw_usage", None)
        if raw_usage is not None:
            found_raw_usage = True
            assert isinstance(raw_usage, list), "tensorzero_raw_usage should be a list"
            assert len(raw_usage) > 0, "tensorzero_raw_usage should have at least one entry"  # type: ignore

            entry = cast(dict[str, Any], raw_usage[0])
            assert "model_inference_id" in entry, "Entry should have model_inference_id"
            assert "provider_type" in entry, "Entry should have provider_type"
            assert "api_type" in entry, "Entry should have api_type"
            assert_openai_chat_usage_details(entry)

    assert found_raw_usage, "Streaming response should include tensorzero_raw_usage in final chunk"
