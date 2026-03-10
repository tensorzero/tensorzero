"""
Tests for tensorzero::include_raw_response parameter using the OpenAI Python SDK.

These tests verify that raw provider-specific response data is correctly returned
when tensorzero::include_raw_response is set to True via the OpenAI-compatible API.
"""

from typing import Any, List, cast

import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from uuid_utils.compat import uuid7


def assert_raw_response_entry_structure(entry: dict[str, Any]) -> None:
    """Verify the structure of a raw_response entry."""
    assert "model_inference_id" in entry, "Entry should have model_inference_id"
    assert "provider_type" in entry, "Entry should have provider_type"
    assert isinstance(entry["provider_type"], str), "provider_type should be a string"
    assert "api_type" in entry, "Entry should have api_type"
    assert entry["api_type"] in ("chat_completions", "responses", "embeddings"), (
        f"api_type should be 'chat_completions', 'responses', or 'embeddings', got {entry['api_type']}"
    )
    assert "data" in entry, "Entry should have data"
    assert isinstance(entry["data"], str), "data should be a string (raw response from provider)"


@pytest.mark.asyncio
async def test_async_raw_response_non_streaming(async_openai_client: AsyncOpenAI):
    """Test that tensorzero::include_raw_response returns tensorzero_raw_response in non-streaming response."""
    messages: List[ChatCompletionMessageParam] = [
        {"role": "user", "content": "Hello"},
    ]

    result = await async_openai_client.chat.completions.create(
        extra_body={
            "tensorzero::episode_id": str(uuid7()),
            "tensorzero::include_raw_response": True,
        },
        messages=messages,
        model="tensorzero::model_name::gpt-4o-mini-2024-07-18",
    )

    # tensorzero_raw_response is at response level
    assert hasattr(result, "tensorzero_raw_response"), "Response should have tensorzero_raw_response when requested"
    assert result.tensorzero_raw_response is not None, "tensorzero_raw_response should not be None"  # type: ignore
    assert isinstance(result.tensorzero_raw_response, list), "tensorzero_raw_response should be a list"  # type: ignore
    assert len(result.tensorzero_raw_response) > 0, "tensorzero_raw_response should have at least one entry"  # type: ignore

    # Verify structure of first entry
    entry = cast(dict[str, Any], result.tensorzero_raw_response[0])  # type: ignore[attr-defined]
    assert_raw_response_entry_structure(entry)


@pytest.mark.asyncio
async def test_async_raw_response_not_requested(async_openai_client: AsyncOpenAI):
    """Test that tensorzero_raw_response is not present when tensorzero::include_raw_response is False."""
    messages: List[ChatCompletionMessageParam] = [
        {"role": "user", "content": "Hello"},
    ]

    result = await async_openai_client.chat.completions.create(
        extra_body={
            "tensorzero::episode_id": str(uuid7()),
            "tensorzero::include_raw_response": False,
        },
        messages=messages,
        model="tensorzero::model_name::gpt-4o-mini-2024-07-18",
    )

    # tensorzero_raw_response is at response level and should not be present when not requested
    raw_response = getattr(result, "tensorzero_raw_response", None)
    assert raw_response is None, "tensorzero_raw_response should be None when not requested"


@pytest.mark.asyncio
async def test_async_raw_response_streaming(async_openai_client: AsyncOpenAI):
    """Test that tensorzero::include_raw_response returns tensorzero_raw_chunk in streaming response."""
    messages: List[ChatCompletionMessageParam] = [
        {"role": "user", "content": "Hello"},
    ]

    stream = await async_openai_client.chat.completions.create(
        extra_body={
            "tensorzero::episode_id": str(uuid7()),
            "tensorzero::include_raw_response": True,
        },
        messages=messages,
        model="tensorzero::model_name::gpt-4o-mini-2024-07-18",
        stream=True,
    )

    found_raw_chunk = False
    async for chunk in stream:
        # Check if this chunk has tensorzero_raw_chunk (raw response data for current chunk)
        raw_chunk = getattr(chunk, "tensorzero_raw_chunk", None)
        if raw_chunk is not None:
            found_raw_chunk = True
            assert isinstance(raw_chunk, str), "tensorzero_raw_chunk should be a string"

        # For single inference streaming, tensorzero_raw_response (array of previous inferences) should be None
        # because there are no previous model inferences in a simple chat completion

    assert found_raw_chunk, "Streaming response should include tensorzero_raw_chunk in at least one chunk"


@pytest.mark.asyncio
async def test_async_raw_response_streaming_not_requested(async_openai_client: AsyncOpenAI):
    """Test that tensorzero_raw_chunk is not present when tensorzero::include_raw_response is False."""
    messages: List[ChatCompletionMessageParam] = [
        {"role": "user", "content": "Hello"},
    ]

    stream = await async_openai_client.chat.completions.create(
        extra_body={
            "tensorzero::episode_id": str(uuid7()),
            "tensorzero::include_raw_response": False,
        },
        messages=messages,
        model="tensorzero::model_name::gpt-4o-mini-2024-07-18",
        stream=True,
    )

    async for chunk in stream:
        # tensorzero_raw_chunk should not be present when not requested
        raw_chunk = getattr(chunk, "tensorzero_raw_chunk", None)
        assert raw_chunk is None, "tensorzero_raw_chunk should not be present when not requested"
