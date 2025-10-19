# type: ignore
"""
Tests for OpenAI Responses API integration
"""

import pytest
from tensorzero import (
    AsyncTensorZeroGateway,
    ChatInferenceResponse,
    Text,
)


@pytest.mark.asyncio
async def test_openai_responses_basic_inference(async_client: AsyncTensorZeroGateway):
    response = await async_client.inference(
        model_name="responses-gpt-5",
        input={
            "messages": [{"role": "user", "content": "What is 2+2?"}],
        },
    )

    assert isinstance(response, ChatInferenceResponse)

    print(response)

    assert len(response.content) > 0

    # Extract the text content block because the response might include reasoning and more
    text_content_block = [cb for cb in response.content if cb.type == "text"]
    assert text_content_block[0].type == "text"
    assert isinstance(text_content_block[0], Text)
    assert "4" in text_content_block[0].text

    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    # TODO (#4041): Check `finish_reason` when we improve handling of `incomplete_details.reason`.
    # assert response.finish_reason == FinishReason.STOP
