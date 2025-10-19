# type: ignore
"""
Tests for OpenAI Responses API integration
"""

import pytest
from uuid_utils.compat import uuid7


@pytest.mark.asyncio
async def test_openai_responses_basic_inference(async_openai_client):
    response = await async_openai_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": str(uuid7())},
        messages=[{"role": "user", "content": "What is 2+2?"}],
        model="tensorzero::model_name::responses-gpt-5-mini",
    )

    assert response.choices[0].message.content is not None
    assert len(response.choices[0].message.content) > 0
    assert "4" in response.choices[0].message.content

    assert response.usage is not None
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    # TODO (#4041): Check `finish_reason` when we improve handling of `incomplete_details.reason`.
    # assert response.choices[0].finish_reason == "stop"
