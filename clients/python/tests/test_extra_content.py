# type: ignore
"""
Tests for `tensorzero_extra_content` round-trip support.

These tests verify that extra content blocks (Thought, Unknown) can be:
1. Received from the API in responses
2. Sent back to the API in follow-up requests (round-trip)
"""

import pytest
from uuid_utils.compat import uuid7


@pytest.mark.asyncio
async def test_extra_content_roundtrip_non_streaming(async_openai_client):
    """Test that extra content can be round-tripped in non-streaming mode."""
    episode_id = str(uuid7())

    # Step 1: Make inference request with a model that returns Thought content
    # The dummy::reasoner model returns [Thought, Text] content
    result = await async_openai_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": episode_id},
        messages=[{"role": "user", "content": "Hello"}],
        model="tensorzero::model_name::dummy::reasoner",
        stream=False,
    )

    # Step 2: Verify response has extra content
    message = result.choices[0].message
    assert message.content is not None, "Response should have text content"

    # Check for extra content (using getattr since it's an extension field)
    extra_content = getattr(message, "tensorzero_extra_content", None)
    assert extra_content is not None, "Response should have tensorzero_extra_content"
    assert len(extra_content) > 0, "Extra content should have at least one block"

    # Verify structure of the thought block
    thought_block = extra_content[0]
    assert thought_block.get("type") == "thought", "First block should be a thought"
    assert "insert_index" in thought_block, "Thought block should have insert_index"
    assert "text" in thought_block, "Thought block should have text field"

    # Step 3: Round-trip - send the extra content back as an assistant message
    roundtrip_result = await async_openai_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": episode_id},
        messages=[
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": message.content,
                "tensorzero_extra_content": extra_content,
            },
            {"role": "user", "content": "Continue"},
        ],
        model="tensorzero::model_name::dummy::echo",
        stream=False,
    )

    # Verify round-trip succeeded
    assert roundtrip_result.choices[0].message is not None, "Round-trip should succeed"


@pytest.mark.asyncio
async def test_extra_content_roundtrip_streaming(async_openai_client):
    """Test that extra content can be round-tripped in streaming mode."""
    episode_id = str(uuid7())

    # Step 1: Make streaming inference request
    stream = await async_openai_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": episode_id},
        messages=[{"role": "user", "content": "Hello"}],
        model="tensorzero::model_name::dummy::reasoner",
        stream=True,
    )

    # Step 2: Collect chunks and extract extra content
    chunks = []
    extra_content_chunks = []
    content_text = ""

    async for chunk in stream:
        chunks.append(chunk)
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta:
            # Collect text content
            if delta.content:
                content_text += delta.content

            # Collect extra content chunks
            extra_content = getattr(delta, "tensorzero_extra_content", None)
            if extra_content:
                extra_content_chunks.extend(extra_content)

    # Step 3: Verify we received extra content in streaming
    assert len(extra_content_chunks) > 0, "Streaming should include extra content chunks"

    # Reconstruct extra content for round-trip (filter for chunks with insert_index)
    reconstructed_extra_content = [chunk for chunk in extra_content_chunks if "insert_index" in chunk]

    # Step 4: Round-trip if we have valid content
    if reconstructed_extra_content and content_text:
        roundtrip_result = await async_openai_client.chat.completions.create(
            extra_body={"tensorzero::episode_id": episode_id},
            messages=[
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": content_text,
                    "tensorzero_extra_content": reconstructed_extra_content,
                },
                {"role": "user", "content": "Continue"},
            ],
            model="tensorzero::model_name::dummy::echo",
            stream=False,
        )

        assert roundtrip_result.choices[0].message is not None, "Streaming round-trip should succeed"
