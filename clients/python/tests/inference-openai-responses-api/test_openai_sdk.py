# type: ignore
"""
Test for OpenAI Responses API integration

This test verifies that TensorZero can use OpenAI's Responses API
through the OpenAI-compatible endpoint.
"""

import pytest
from uuid_utils.compat import uuid7


@pytest.mark.asyncio
async def test_basic_responses_api(async_openai_client):
    """Test basic inference using OpenAI Responses API."""
    messages = [
        {
            "role": "user",
            "content": "Tell me a fun fact.",
        }
    ]

    result = await async_openai_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": str(uuid7())},
        messages=messages,
        model="tensorzero::model_name::responses-gpt-5-mini",
    )

    # Verify we got a response
    assert result.choices[0].message.content is not None
    assert len(result.choices[0].message.content) > 0
    assert result.choices[0].finish_reason == "stop"

    # Verify usage information
    assert result.usage is not None
    assert result.usage.prompt_tokens > 0
    assert result.usage.completion_tokens > 0
    assert result.usage.total_tokens > 0
