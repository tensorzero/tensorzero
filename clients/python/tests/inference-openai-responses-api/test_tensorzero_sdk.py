# type: ignore
"""
Tests for OpenAI Responses API integration
"""

import typing as t

import pytest
from tensorzero import (
    AsyncTensorZeroGateway,
    ChatInferenceResponse,
    InferenceChunk,
    Text,
    TextChunk,
    ThoughtChunk,
    ToolCall,
)
from tensorzero.types import (
    ChatChunk,
    Thought,
    ThoughtSummaryBlock,
    ToolCallChunk,
    UnknownContentBlock,
)


@pytest.mark.asyncio
async def test_openai_responses_basic_inference(async_client: AsyncTensorZeroGateway):
    response = await async_client.inference(
        model_name="gpt-5-mini-responses",
        input={
            "messages": [{"role": "user", "content": "What is 2+2?"}],
        },
    )

    assert isinstance(response, ChatInferenceResponse)

    assert len(response.content) > 0

    # Extract the text content block because the response might include reasoning and more
    text_content_block = [cb for cb in response.content if cb.type == "text"]
    assert len(text_content_block) > 0
    assert text_content_block[0].type == "text"
    assert isinstance(text_content_block[0], Text)
    assert "4" in text_content_block[0].text

    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    # TODO (#4041): Check `finish_reason` when we improve handling of `incomplete_details.reason`.
    # assert response.finish_reason == FinishReason.STOP


@pytest.mark.asyncio
async def test_openai_responses_basic_inference_streaming(
    async_client: AsyncTensorZeroGateway,
):
    stream = await async_client.inference(
        model_name="gpt-5-mini-responses",
        input={
            "messages": [{"role": "user", "content": "What is 2+2?"}],
        },
        stream=True,
    )
    assert isinstance(stream, t.AsyncIterator)

    chunks: t.List[InferenceChunk] = []
    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) > 0

    # Verify consistency across chunks
    previous_inference_id = None
    previous_episode_id = None
    text_chunks = []
    for chunk in chunks:
        assert isinstance(chunk, ChatChunk)
        if previous_inference_id is not None:
            assert chunk.inference_id == previous_inference_id
        if previous_episode_id is not None:
            assert chunk.episode_id == previous_episode_id
        previous_inference_id = chunk.inference_id
        previous_episode_id = chunk.episode_id

        # Collect text chunks
        for content_block in chunk.content:
            if content_block.type == "text":
                assert isinstance(content_block, TextChunk)
                text_chunks.append(content_block.text)

    # Should have received text content with "4" in it
    assert len(text_chunks) > 0
    full_text = "".join(text_chunks)
    assert "4" in full_text

    # Last chunk should have usage
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.input_tokens > 0
    assert chunks[-1].usage.output_tokens > 0
    # TODO (#4041): Check `finish_reason` when we improve handling of `incomplete_details.reason`.


@pytest.mark.asyncio
async def test_openai_responses_web_search(async_client: AsyncTensorZeroGateway):
    """Test OpenAI Responses API with built-in web search tool"""
    response = await async_client.inference(
        model_name="gpt-5-mini-responses-web-search",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "What is the current population of Japan?",
                }
            ],
        },
    )

    assert isinstance(response, ChatInferenceResponse)

    # The response should contain content
    assert len(response.content) > 0

    # Check that web search actually happened by looking for web_search_call content blocks
    web_search_blocks = [
        cb
        for cb in response.content
        if cb.type == "unknown" and isinstance(cb, UnknownContentBlock) and cb.data.get("type") == "web_search_call"
    ]
    assert len(web_search_blocks) > 0, "Expected web_search_call content blocks"

    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0


@pytest.mark.asyncio
async def test_openai_responses_web_search_streaming(
    async_client: AsyncTensorZeroGateway,
):
    """Test OpenAI Responses API with built-in web search tool (streaming)"""
    stream = await async_client.inference(
        model_name="gpt-5-mini-responses-web-search",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "What is the current population of Japan?",
                }
            ],
        },
        stream=True,
    )
    assert isinstance(stream, t.AsyncIterator)

    chunks: t.List[InferenceChunk] = []
    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) > 0

    # Verify consistency across chunks and collect text
    previous_inference_id = None
    previous_episode_id = None
    text_chunks = []
    for chunk in chunks:
        assert isinstance(chunk, ChatChunk)
        if previous_inference_id is not None:
            assert chunk.inference_id == previous_inference_id
        if previous_episode_id is not None:
            assert chunk.episode_id == previous_episode_id
        previous_inference_id = chunk.inference_id
        previous_episode_id = chunk.episode_id

        # Collect text chunks
        for content_block in chunk.content:
            if content_block.type == "text":
                assert isinstance(content_block, TextChunk)
                text_chunks.append(content_block.text)

    # Last chunk should have usage
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.input_tokens > 0
    assert chunks[-1].usage.output_tokens > 0

    # Check that web search actually happened by looking for citations in markdown format
    full_text = "".join(text_chunks)
    assert "](" in full_text, (
        f"Expected concatenated text to contain citations in markdown format [text](url), but found none. Text length: {len(full_text)}"
    )

    # TODO (#4044): check for unknown web search events when we start returning them


@pytest.mark.asyncio
async def test_openai_responses_tool_call(async_client: AsyncTensorZeroGateway):
    """Test OpenAI Responses API with tool calls passed at inference time"""
    response = await async_client.inference(
        model_name="gpt-5-mini-responses",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "What's the temperature in Tokyo in Celsius?",
                }
            ],
        },
        additional_tools=[
            {
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
            }
        ],
    )

    assert isinstance(response, ChatInferenceResponse)

    # The response should contain a tool call
    assert len(response.content) > 0

    # Find the tool call
    tool_calls = [cb for cb in response.content if cb.type == "tool_call"]
    assert len(tool_calls) > 0

    tool_call = tool_calls[0]
    assert isinstance(tool_call, ToolCall)
    assert tool_call.name == "get_temperature"
    assert tool_call.arguments is not None
    assert "location" in tool_call.arguments

    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0


@pytest.mark.asyncio
async def test_openai_responses_tool_call_streaming(
    async_client: AsyncTensorZeroGateway,
):
    """Test OpenAI Responses API with tool calls (streaming)"""
    stream = await async_client.inference(
        model_name="gpt-5-mini-responses",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "What's the temperature in Tokyo in Celsius?",
                }
            ],
        },
        additional_tools=[
            {
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
            }
        ],
        stream=True,
    )
    assert isinstance(stream, t.AsyncIterator)

    chunks: t.List[InferenceChunk] = []
    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) > 0

    # Verify consistency across chunks
    previous_inference_id = None
    previous_episode_id = None
    tool_call_name = ""
    for chunk in chunks:
        assert isinstance(chunk, ChatChunk)
        if previous_inference_id is not None:
            assert chunk.inference_id == previous_inference_id
        if previous_episode_id is not None:
            assert chunk.episode_id == previous_episode_id
        previous_inference_id = chunk.inference_id
        previous_episode_id = chunk.episode_id

        # Check for tool call chunks and get the tool name
        for content_block in chunk.content:
            if content_block.type == "tool_call":
                assert isinstance(content_block, ToolCallChunk)
                if content_block.raw_name is not None:
                    tool_call_name += content_block.raw_name

    # Last chunk should have usage
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.input_tokens > 0
    assert chunks[-1].usage.output_tokens > 0

    # Should have received a tool call for get_temperature
    assert tool_call_name == "get_temperature"


@pytest.mark.asyncio
async def test_openai_responses_reasoning(async_client: AsyncTensorZeroGateway):
    """Test OpenAI Responses API with encrypted reasoning (thought blocks)"""
    response = await async_client.inference(
        model_name="gpt-5-mini-responses",
        input={
            "messages": [{"role": "user", "content": "How many letters are in the word potato?"}],
        },
        extra_body=[
            {
                "variant_name": "gpt-5-mini-responses",
                "pointer": "/reasoning",
                "value": {"effort": "low", "summary": "auto"},
            }
        ],
    )

    assert isinstance(response, ChatInferenceResponse)

    # The response should contain content blocks
    assert len(response.content) > 0

    # Check for encrypted thought blocks
    thought_blocks = [cb for cb in response.content if cb.type == "thought"]

    # We expect at least one thought block when reasoning is enabled
    assert len(thought_blocks) > 0, "Expected thought content blocks when reasoning is enabled"

    # Verify thought content blocks exist
    for thought in thought_blocks:
        assert isinstance(thought, Thought)
        assert thought.type == "thought"

    # Check that at least one thought has a summary
    thought_with_summary = [t for t in thought_blocks if t.summary is not None]
    assert len(thought_with_summary) > 0, "Expected at least one thought block to have a summary"

    # Verify the summary structure
    for thought in thought_with_summary:
        assert isinstance(thought.summary, list)
        assert len(thought.summary) > 0
        for summary_block in thought.summary:
            assert isinstance(summary_block, ThoughtSummaryBlock)
            assert isinstance(summary_block.text, str)
            assert len(summary_block.text) > 0

    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0


@pytest.mark.asyncio
async def test_openai_responses_reasoning_streaming(
    async_client: AsyncTensorZeroGateway,
):
    """Test OpenAI Responses API with encrypted reasoning (streaming)"""
    stream = await async_client.inference(
        model_name="gpt-5-mini-responses",
        input={
            "messages": [{"role": "user", "content": "How many letters are in the word potato?"}],
        },
        extra_body=[
            {
                "variant_name": "gpt-5-mini-responses",
                "pointer": "/reasoning",
                "value": {"effort": "low", "summary": "auto"},
            }
        ],
        stream=True,
    )
    assert isinstance(stream, t.AsyncIterator)

    chunks: t.List[InferenceChunk] = []
    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) > 0

    # Verify consistency across chunks
    previous_inference_id = None
    previous_episode_id = None
    has_thought = False
    for chunk in chunks:
        assert isinstance(chunk, ChatChunk)
        if previous_inference_id is not None:
            assert chunk.inference_id == previous_inference_id
        if previous_episode_id is not None:
            assert chunk.episode_id == previous_episode_id
        previous_inference_id = chunk.inference_id
        previous_episode_id = chunk.episode_id

        # Check for thought chunks
        for content_block in chunk.content:
            if content_block.type == "thought":
                assert isinstance(content_block, ThoughtChunk)
                has_thought = True

    # Last chunk should have usage
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.input_tokens > 0
    assert chunks[-1].usage.output_tokens > 0

    # Should have received thought chunks when reasoning is enabled
    assert has_thought, "Expected thought content blocks when reasoning is enabled"

    # Note: Checking streaming summary chunks would require aggregating chunks across
    # multiple messages, which is complex. The summary is fully tested in the
    # non-streaming test above.


@pytest.mark.asyncio
async def test_openai_responses_web_search_dynamic_provider_tools(
    async_client: AsyncTensorZeroGateway,
):
    """Test OpenAI Responses API with dynamically configured provider tools (web search)"""
    response = await async_client.inference(
        model_name="gpt-5-mini-responses",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "What is the current population of Japan?",
                }
            ],
        },
        provider_tools=[{"tool": {"type": "web_search"}}],
    )

    assert isinstance(response, ChatInferenceResponse)

    # The response should contain content
    assert len(response.content) > 0
    # Check that web search actually happened by looking for web_search_call content blocks
    web_search_blocks = [
        cb
        for cb in response.content
        if cb.type == "unknown" and isinstance(cb, UnknownContentBlock) and cb.data.get("type") == "web_search_call"
    ]
    assert len(web_search_blocks) > 0, "Expected web_search_call content blocks"


@pytest.mark.asyncio
async def test_openai_responses_shorthand(async_client: AsyncTensorZeroGateway):
    """Test OpenAI Responses API using shorthand model name format"""
    response = await async_client.inference(
        model_name="openai::responses::gpt-5-codex",
        input={
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
        },
    )

    assert isinstance(response, ChatInferenceResponse)

    # The response should contain content
    assert len(response.content) > 0

    # Extract the text content block
    text_content_blocks = [cb for cb in response.content if cb.type == "text"]
    assert len(text_content_blocks) > 0
    assert isinstance(text_content_blocks[0], Text)

    # Check that the response mentions Paris
    assert "Paris" in text_content_blocks[0].text, "Content should mention Paris"

    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
