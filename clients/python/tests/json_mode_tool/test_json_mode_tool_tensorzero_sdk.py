# type: ignore
"""
Tests for json_mode="tool" using the TensorZero Python SDK

These tests verify that chat functions with json_mode="tool" properly convert
tool calls to text responses, both in streaming and non-streaming modes.
"""

import json

import pytest
from tensorzero import AsyncTensorZeroGateway, ChatInferenceResponse


@pytest.mark.asyncio
async def test_chat_json_mode_tool_non_streaming():
    """
    Test chat function with json_mode="tool" in non-streaming mode.

    Verifies that:
    - Chat function with NO tools configured accepts json_mode="tool"
    - Response is TEXT (not tool_call)
    - JSON is valid and matches output_schema
    """
    client = AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000",
        verbose_errors=True,
        async_setup=False,
    )
    assert isinstance(client, AsyncTensorZeroGateway)

    output_schema = {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "confidence": {"type": "number"},
        },
        "required": ["sentiment", "confidence"],
        "additionalProperties": False,
    }

    response = await client.inference(
        function_name="test_chat_json_mode_tool_openai",
        input={"messages": [{"role": "user", "content": "Analyze sentiment"}]},
        params={"chat_completion": {"json_mode": "tool"}},
        output_schema=output_schema,
        stream=False,
    )

    # Verify we got a chat response (not streaming)
    assert isinstance(response, ChatInferenceResponse)

    # Verify response has content
    assert len(response.content) > 0

    # Extract the text content
    content_block = response.content[0]
    assert hasattr(content_block, "text"), "Expected text content, not tool_call"
    text_content = content_block.text

    # Verify the text is valid JSON
    parsed_json = json.loads(text_content)

    # Verify schema structure
    assert "sentiment" in parsed_json, "Should have 'sentiment' field"
    assert "confidence" in parsed_json, "Should have 'confidence' field"

    # Verify the values from dummy provider
    assert parsed_json["sentiment"] == "positive"
    assert parsed_json["confidence"] == 0.95

    await client.close()


@pytest.mark.asyncio
async def test_chat_json_mode_tool_streaming():
    """
    Test chat function with json_mode="tool" in streaming mode.

    Verifies that:
    - Chat function with NO tools configured accepts json_mode="tool"
    - Chunks are TEXT chunks (not tool_call chunks)
    - Accumulated JSON is valid and matches output_schema
    """
    client = AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000",
        verbose_errors=True,
        async_setup=False,
    )
    assert isinstance(client, AsyncTensorZeroGateway)

    output_schema = {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "confidence": {"type": "number"},
        },
        "required": ["sentiment", "confidence"],
        "additionalProperties": False,
    }

    stream = await client.inference(
        function_name="test_chat_json_mode_tool_openai",
        input={"messages": [{"role": "user", "content": "Analyze sentiment"}]},
        params={"chat_completion": {"json_mode": "tool"}},
        output_schema=output_schema,
        stream=True,
    )

    # Accumulate text from chunks
    accumulated_text = ""
    chunk_count = 0

    async for chunk in stream:
        chunk_count += 1

        # Verify we're getting chat chunks
        assert hasattr(chunk, "content"), "Expected chat chunk with content"

        # Verify chunks are text chunks (not tool_call)
        for content_block in chunk.content:
            assert hasattr(content_block, "text"), f"Expected text chunk, got {type(content_block)}"
            accumulated_text += content_block.text

    # Verify we got at least one chunk
    assert chunk_count > 0, "Should have received at least one chunk"

    # Verify the accumulated text is not empty
    assert len(accumulated_text) > 0, "Should have accumulated some text"

    # Verify the accumulated text is valid JSON
    parsed_json = json.loads(accumulated_text)

    # Verify schema structure
    assert "sentiment" in parsed_json, "Should have 'sentiment' field"
    assert "confidence" in parsed_json, "Should have 'confidence' field"

    # Verify the values from dummy provider
    assert parsed_json["sentiment"] == "positive"
    assert parsed_json["confidence"] == 0.95

    await client.close()
