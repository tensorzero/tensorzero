# type: ignore
"""
Tests for json_mode="tool" using the OpenAI Python SDK with TensorZero

These tests verify that chat functions with json_mode="tool" properly convert
tool calls to text responses when using the OpenAI-compatible API, both in
streaming and non-streaming modes.
"""

import json

import pytest


@pytest.mark.asyncio
async def test_chat_json_mode_tool_non_streaming_openai(async_openai_client):
    """
    Test chat function with json_mode="tool" in non-streaming mode using OpenAI SDK.

    Verifies that:
    - Chat function with NO tools configured accepts json_mode="tool"
    - Response is TEXT (not tool_call)
    - JSON is valid and matches output_schema
    """
    output_schema = {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "confidence": {"type": "number"},
        },
        "required": ["sentiment", "confidence"],
        "additionalProperties": False,
    }

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "sentiment_analysis",
            "description": "Sentiment analysis schema",
            "schema": output_schema,
            "strict": True,
        },
    }

    response = await async_openai_client.chat.completions.create(
        model="tensorzero::function_name::test_chat_json_mode_tool_openai",
        messages=[{"role": "user", "content": "Analyze sentiment"}],
        response_format=response_format,
        extra_body={
            "tensorzero::params": {
                "chat_completion": {
                    "json_mode": "tool",
                }
            }
        },
        stream=False,
    )

    # Verify we got a response
    assert response.choices is not None
    assert len(response.choices) > 0

    # Extract the text content
    message = response.choices[0].message
    assert message.content is not None, "Expected text content, not tool_call"

    # Verify no tool_calls (should be text response)
    assert message.tool_calls is None or len(message.tool_calls) == 0

    # Verify the text is valid JSON
    parsed_json = json.loads(message.content)

    # Verify schema structure
    assert "sentiment" in parsed_json, "Should have 'sentiment' field"
    assert "confidence" in parsed_json, "Should have 'confidence' field"

    # Verify the values from dummy provider
    assert parsed_json["sentiment"] == "positive"
    assert parsed_json["confidence"] == 0.95


@pytest.mark.asyncio
async def test_chat_json_mode_tool_streaming_openai(async_openai_client):
    """
    Test chat function with json_mode="tool" in streaming mode using OpenAI SDK.

    Verifies that:
    - Chat function with NO tools configured accepts json_mode="tool"
    - Chunks are TEXT chunks (not tool_call chunks)
    - Accumulated JSON is valid and matches output_schema
    """
    output_schema = {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "confidence": {"type": "number"},
        },
        "required": ["sentiment", "confidence"],
        "additionalProperties": False,
    }

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "sentiment_analysis",
            "description": "Sentiment analysis schema",
            "schema": output_schema,
            "strict": True,
        },
    }

    stream = await async_openai_client.chat.completions.create(
        model="tensorzero::function_name::test_chat_json_mode_tool_openai",
        messages=[{"role": "user", "content": "Analyze sentiment"}],
        response_format=response_format,
        extra_body={
            "tensorzero::params": {
                "chat_completion": {
                    "json_mode": "tool",
                }
            }
        },
        stream=True,
    )

    # Accumulate text from chunks
    accumulated_text = ""
    chunk_count = 0

    async for chunk in stream:
        chunk_count += 1

        # Verify we're getting chat chunks
        assert chunk.choices is not None, "Expected chunk with choices"

        # Verify chunks are text chunks (not tool_call)
        for choice in chunk.choices:
            if choice.delta.content is not None:
                accumulated_text += choice.delta.content
            # Verify no tool_calls in delta
            if choice.delta.tool_calls is not None:
                assert len(choice.delta.tool_calls) == 0, "Expected text chunk, not tool_call chunk"

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
