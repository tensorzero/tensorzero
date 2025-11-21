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
        model="tensorzero::model_name::gpt-5-mini-responses",
    )

    # The response should contain content
    assert response.choices[0].message.content is not None
    assert len(response.choices[0].message.content) > 0

    # Extract the text content because the response might include reasoning and more
    # In OpenAI API, content is a single string, not separate blocks like TensorZero SDK
    assert "4" in response.choices[0].message.content

    assert response.usage is not None
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    # TODO (#4041): Check `finish_reason` when we improve handling of `incomplete_details.reason`.
    # assert response.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_openai_responses_basic_inference_streaming(async_openai_client):
    stream = await async_openai_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": str(uuid7())},
        messages=[{"role": "user", "content": "What is 2+2?"}],
        model="tensorzero::model_name::gpt-5-mini-responses",
        stream=True,
        stream_options={"include_usage": True},
    )

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) > 0

    # Verify consistency across chunks
    previous_inference_id = None
    previous_episode_id = None
    text_chunks = []
    for i, chunk in enumerate(chunks):
        if previous_inference_id is not None:
            assert chunk.id == previous_inference_id
        if previous_episode_id is not None:
            assert chunk.episode_id == previous_episode_id
        previous_inference_id = chunk.id
        previous_episode_id = chunk.episode_id

        # Collect text chunks (all chunks except the final usage-only chunk)
        if chunk.choices and chunk.choices[0].delta.content:
            text_chunks.append(chunk.choices[0].delta.content)

    # Should have received text content with "4" in it
    assert len(text_chunks) > 0
    full_text = "".join(text_chunks)
    assert "4" in full_text

    # Last chunk should have usage
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.prompt_tokens > 0
    assert chunks[-1].usage.completion_tokens > 0
    # TODO (#4041): Check `finish_reason` when we improve handling of `incomplete_details.reason`.


@pytest.mark.asyncio
async def test_openai_responses_web_search(async_openai_client):
    """Test OpenAI Responses API with built-in web search tool"""
    response = await async_openai_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": str(uuid7())},
        messages=[
            {
                "role": "user",
                "content": "What is the current population of Japan?",
            }
        ],
        model="tensorzero::model_name::gpt-5-mini-responses-web-search",
    )

    # The response should contain content
    assert response.choices[0].message.content is not None
    assert len(response.choices[0].message.content) > 0

    # Check that web search actually happened by looking for citations in markdown format
    assert "](" in response.choices[0].message.content, (
        f"Expected text to contain citations in markdown format [text](url), but found none. Text length: {len(response.choices[0].message.content)}"
    )

    # TODO (#4042): Check for web_search_call content blocks when we expose them in the OpenAI API
    # The TensorZero SDK returns web_search_call content blocks, but the OpenAI API doesn't expose them yet

    assert response.usage is not None
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0


@pytest.mark.asyncio
async def test_openai_responses_web_search_streaming(async_openai_client):
    """Test OpenAI Responses API with built-in web search tool (streaming)"""
    stream = await async_openai_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": str(uuid7())},
        messages=[
            {
                "role": "user",
                "content": "What is the current population of Japan?",
            }
        ],
        model="tensorzero::model_name::gpt-5-mini-responses-web-search",
        stream=True,
        stream_options={"include_usage": True},
    )

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) > 0

    # Verify consistency across chunks and collect text
    previous_inference_id = None
    previous_episode_id = None
    text_chunks = []
    for chunk in chunks:
        if previous_inference_id is not None:
            assert chunk.id == previous_inference_id
        if previous_episode_id is not None:
            assert chunk.episode_id == previous_episode_id
        previous_inference_id = chunk.id
        previous_episode_id = chunk.episode_id

        # Collect text chunks
        if chunk.choices and chunk.choices[0].delta.content:
            text_chunks.append(chunk.choices[0].delta.content)

    # Last chunk should have usage
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.prompt_tokens > 0
    assert chunks[-1].usage.completion_tokens > 0

    # Check that web search actually happened by looking for citations in markdown format
    full_text = "".join(text_chunks)
    assert "](" in full_text, (
        f"Expected concatenated text to contain citations in markdown format [text](url), but found none. Text length: {len(full_text)}"
    )

    # TODO (#4044): check for unknown web search events when we start returning them


@pytest.mark.asyncio
async def test_openai_responses_tool_call(async_openai_client):
    """Test OpenAI Responses API with tool calls passed at inference time"""
    response = await async_openai_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": str(uuid7())},
        messages=[
            {
                "role": "user",
                "content": "What's the temperature in Tokyo in Celsius?",
            }
        ],
        model="tensorzero::model_name::gpt-5-mini-responses",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_temperature",
                    "description": "Get the current temperature in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": 'The location to get the temperature for (e.g. "New York")',
                            },
                            "units": {
                                "type": "string",
                                "description": 'The units to get the temperature in (must be "fahrenheit" or "celsius")',
                                "enum": ["fahrenheit", "celsius"],
                            },
                        },
                        "required": ["location"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
    )

    # The response should contain content (tool calls)

    # Find the tool call
    assert response.choices[0].message.tool_calls is not None
    assert len(response.choices[0].message.tool_calls) > 0

    tool_call = response.choices[0].message.tool_calls[0]
    assert tool_call.function.name == "get_temperature"
    assert tool_call.function.arguments is not None
    assert "location" in tool_call.function.arguments

    assert response.usage is not None
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0


@pytest.mark.asyncio
async def test_openai_responses_tool_call_streaming(async_openai_client):
    """Test OpenAI Responses API with tool calls (streaming)"""
    stream = await async_openai_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": str(uuid7())},
        messages=[
            {
                "role": "user",
                "content": "What's the temperature in Tokyo in Celsius?",
            }
        ],
        model="tensorzero::model_name::gpt-5-mini-responses",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_temperature",
                    "description": "Get the current temperature in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": 'The location to get the temperature for (e.g. "New York")',
                            },
                            "units": {
                                "type": "string",
                                "description": 'The units to get the temperature in (must be "fahrenheit" or "celsius")',
                                "enum": ["fahrenheit", "celsius"],
                            },
                        },
                        "required": ["location"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        stream=True,
        stream_options={"include_usage": True},
    )

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) > 0

    # Verify consistency across chunks
    previous_inference_id = None
    previous_episode_id = None
    tool_call_name = ""
    for chunk in chunks:
        if previous_inference_id is not None:
            assert chunk.id == previous_inference_id
        if previous_episode_id is not None:
            assert chunk.episode_id == previous_episode_id
        previous_inference_id = chunk.id
        previous_episode_id = chunk.episode_id

        # Check for tool call chunks and get the tool name
        if chunk.choices and chunk.choices[0].delta.tool_calls:
            for tool_call_delta in chunk.choices[0].delta.tool_calls:
                if tool_call_delta.function and tool_call_delta.function.name:
                    tool_call_name += tool_call_delta.function.name

    # Last chunk should have usage
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.prompt_tokens > 0
    assert chunks[-1].usage.completion_tokens > 0

    # Should have received a tool call for get_temperature
    assert tool_call_name == "get_temperature"


@pytest.mark.asyncio
async def test_openai_responses_web_search_dynamic_provider_tools(async_openai_client):
    """Test OpenAI Responses API with dynamically configured provider tools (web search)"""
    response = await async_openai_client.chat.completions.create(
        extra_body={
            "tensorzero::episode_id": str(uuid7()),
            "tensorzero::provider_tools": [{"tool": {"type": "web_search"}}],
        },
        messages=[
            {
                "role": "user",
                "content": "What is the current population of Japan?",
            }
        ],
        model="tensorzero::model_name::gpt-5-mini-responses",
    )

    # The response should contain content
    assert response.choices[0].message.content is not None
    assert len(response.choices[0].message.content) > 0

    # Check that web search actually happened by looking for citations in markdown format
    assert "](" in response.choices[0].message.content, (
        f"Expected text to contain citations in markdown format [text](url), but found none. Text length: {len(response.choices[0].message.content)}"
    )

    # TODO (#4042): Check for web_search_call content blocks when we expose them in the OpenAI API
    # The TensorZero SDK returns web_search_call content blocks, but the OpenAI API doesn't expose them yet

    assert response.usage is not None
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0


# Note:
# The OpenAI SDK doesn't expose reasoning through chat completions, so there's no way to test that.
# Use the TensorZero SDK to retrieve reasoning.


@pytest.mark.asyncio
async def test_openai_responses_shorthand(async_openai_client):
    """Test OpenAI Responses API using shorthand model name format"""
    response = await async_openai_client.chat.completions.create(
        extra_body={"tensorzero::episode_id": str(uuid7())},
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        model="tensorzero::model_name::openai::responses::gpt-5-codex",
    )

    # The response should contain content
    assert response.choices[0].message.content is not None
    assert len(response.choices[0].message.content) > 0

    # Check that the response mentions Paris
    assert "Paris" in response.choices[0].message.content, "Content should mention Paris"

    assert response.usage is not None
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
