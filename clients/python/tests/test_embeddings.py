# type: ignore
"""
Tests for the TensorZero embeddings API using the OpenAI Python client

These tests cover the embeddings functionality of the TensorZero OpenAI-compatible interface.

To run:
```
pytest tests/test_embeddings.py
```
or
```
uv run pytest tests/test_embeddings.py
```
"""

import asyncio
import random

import pytest


@pytest.mark.asyncio
async def test_basic_embeddings(async_openai_client):
    """Test basic embeddings generation with a single input"""
    result = await async_openai_client.embeddings.create(
        input="Hello, world!",
        model="tensorzero::embedding_model_name::text-embedding-3-small",
    )

    # Verify the response structure
    assert result.model == "tensorzero::embedding_model_name::text-embedding-3-small"
    assert len(result.data) == 1
    assert result.data[0].index == 0
    assert result.data[0].object == "embedding"
    assert len(result.data[0].embedding) > 0  # Should have embedding vector
    assert result.usage.prompt_tokens > 0
    assert result.usage.total_tokens > 0


@pytest.mark.asyncio
async def test_basic_embeddings_shorthand(async_openai_client):
    """Test basic embeddings generation with a single input"""
    result = await async_openai_client.embeddings.create(
        input="Hello, world!",
        model="tensorzero::embedding_model_name::openai::text-embedding-3-large",
    )

    # Verify the response structure
    assert result.model == "tensorzero::embedding_model_name::openai::text-embedding-3-large"
    assert len(result.data) == 1
    assert result.data[0].index == 0
    assert result.data[0].object == "embedding"
    assert len(result.data[0].embedding) > 0  # Should have embedding vector
    assert result.usage.prompt_tokens > 0
    assert result.usage.total_tokens > 0


@pytest.mark.asyncio
async def test_batch_embeddings(async_openai_client):
    """Test embeddings generation with multiple inputs"""
    inputs = [
        "Hello, world!",
        "How are you today?",
        "This is a test of batch embeddings.",
    ]

    result = await async_openai_client.embeddings.create(
        input=inputs,
        model="tensorzero::embedding_model_name::text-embedding-3-small",
    )

    # Verify the response structure
    assert result.model == "tensorzero::embedding_model_name::text-embedding-3-small"
    assert len(result.data) == len(inputs)

    for i, embedding_data in enumerate(result.data):
        assert embedding_data.index == i
        assert embedding_data.object == "embedding"
        assert len(embedding_data.embedding) > 0

    assert result.usage.prompt_tokens > 0
    assert result.usage.total_tokens > 0


@pytest.mark.asyncio
async def test_embeddings_with_dimensions(async_openai_client):
    """Test embeddings with specified dimensions"""
    result = await async_openai_client.embeddings.create(
        input="Test with specific dimensions",
        model="tensorzero::embedding_model_name::text-embedding-3-small",
        dimensions=512,
    )

    # Verify the response structure
    assert result.model == "tensorzero::embedding_model_name::text-embedding-3-small"
    assert len(result.data) == 1
    # Should match requested dimensions
    assert len(result.data[0].embedding) == 512


@pytest.mark.asyncio
async def test_embeddings_with_encoding_format_float(async_openai_client):
    """Test embeddings with different encoding formats"""
    result = await async_openai_client.embeddings.create(
        input="Test encoding format",
        model="tensorzero::embedding_model_name::text-embedding-3-small",
        encoding_format="float",
    )

    # Verify the response structure
    assert result.model == "tensorzero::embedding_model_name::text-embedding-3-small"
    assert len(result.data) == 1
    assert isinstance(result.data[0].embedding[0], float)


@pytest.mark.asyncio
async def test_embeddings_with_encoding_format_base64(async_openai_client):
    """Test embeddings with different encoding formats"""
    result = await async_openai_client.embeddings.create(
        input="Test encoding format",
        model="tensorzero::embedding_model_name::text-embedding-3-small",
        encoding_format="base64",
    )

    # Verify the response structure
    assert result.model == "tensorzero::embedding_model_name::text-embedding-3-small"
    assert len(result.data) == 1
    assert isinstance(result.data[0].embedding, str)


@pytest.mark.asyncio
async def test_embeddings_with_user_parameter(async_openai_client):
    """Test embeddings with user parameter for tracking"""
    user_id = "test_user_123"
    result = await async_openai_client.embeddings.create(
        input="Test with user parameter",
        model="tensorzero::embedding_model_name::text-embedding-3-small",
        user=user_id,
    )

    # Verify the response structure
    assert result.model == "tensorzero::embedding_model_name::text-embedding-3-small"
    assert len(result.data) == 1
    assert len(result.data[0].embedding) > 0


@pytest.mark.asyncio
async def test_embeddings_invalid_model_error(async_openai_client):
    """Test that invalid model name raises appropriate error"""
    with pytest.raises(Exception) as exc_info:
        await async_openai_client.embeddings.create(
            input="Test invalid model",
            model="tensorzero::embedding_model_name::nonexistent_model",
        )

    # Should get a 404 error for unknown model
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_embeddings_large_batch(async_openai_client):
    """Test embeddings with a larger batch of inputs"""
    # Create a batch of 10 different inputs
    inputs = [f"This is test input number {i + 1}" for i in range(10)]

    result = await async_openai_client.embeddings.create(
        input=inputs,
        model="tensorzero::embedding_model_name::text-embedding-3-small",
    )

    # Verify the response structure
    assert result.model == "tensorzero::embedding_model_name::text-embedding-3-small"
    assert len(result.data) == 10

    # Verify each embedding
    for i, embedding_data in enumerate(result.data):
        assert embedding_data.index == i
        assert embedding_data.object == "embedding"
        assert len(embedding_data.embedding) > 0

    assert result.usage.prompt_tokens > 0
    assert result.usage.total_tokens > 0


@pytest.mark.asyncio
async def test_embeddings_consistency(async_openai_client):
    """Test that the same input produces consistent embeddings"""
    input_text = "This is a consistency test"

    # Generate embeddings twice with the same input
    result1 = await async_openai_client.embeddings.create(
        input=input_text,
        model="tensorzero::embedding_model_name::text-embedding-3-small",
    )

    result2 = await async_openai_client.embeddings.create(
        input=input_text,
        model="tensorzero::embedding_model_name::text-embedding-3-small",
    )

    # Both should have the same model and structure
    assert result1.model == result2.model
    assert len(result1.data) == len(result2.data) == 1
    assert len(result1.data[0].embedding) == len(result2.data[0].embedding)

    # The embeddings should be identical for the same input
    # (assuming deterministic behavior or proper caching)
    embedding1 = result1.data[0].embedding
    embedding2 = result2.data[0].embedding

    # Check that embeddings are similar (allowing for small numerical differences)
    for i in range(min(10, len(embedding1))):  # Check first 10 dimensions
        assert abs(embedding1[i] - embedding2[i]) < 0.01, f"Embeddings differ significantly at index {i}"


@pytest.mark.asyncio
async def test_embeddings_cache_with_float_encoding(async_openai_client):
    """Test that caching works correctly with float encoding format"""
    # Use a unique input to ensure we're not hitting existing cache
    input_text = f"Cache test with float encoding - {random.randint(0, 1000000)}"

    # First request with float encoding and cache enabled
    result1 = await async_openai_client.embeddings.create(
        input=input_text,
        model="tensorzero::embedding_model_name::text-embedding-3-small",
        encoding_format="float",
        extra_body={"tensorzero::cache_options": {"enabled": "on"}},
    )

    # Verify first response has non-zero usage (not from cache)
    assert result1.usage.prompt_tokens > 0
    assert result1.usage.total_tokens > 0
    assert isinstance(result1.data[0].embedding[0], float)  # float encoded

    # Wait briefly for cache write to complete
    await asyncio.sleep(2)

    # Second request with same parameters - should hit cache
    result2 = await async_openai_client.embeddings.create(
        input=input_text,
        model="tensorzero::embedding_model_name::text-embedding-3-small",
        encoding_format="float",
        extra_body={"tensorzero::cache_options": {"enabled": "on"}},
    )

    # Verify second response has zero usage (from cache)
    assert result2.usage.prompt_tokens == 0
    assert result2.usage.total_tokens == 0
    assert isinstance(result2.data[0].embedding[0], float)  # float encoded
    assert result1.data[0].embedding == result2.data[0].embedding  # Same embedding


@pytest.mark.asyncio
async def test_embeddings_cache_with_base64_encoding(async_openai_client):
    """Test that caching works correctly with base64 encoding format"""
    # Use a unique input to ensure we're not hitting existing cache
    input_text = f"Cache test with base64 encoding - {random.randint(0, 1000000)}"

    # First request with base64 encoding and cache enabled
    result1 = await async_openai_client.embeddings.create(
        input=input_text,
        model="tensorzero::embedding_model_name::text-embedding-3-small",
        encoding_format="base64",
        extra_body={"tensorzero::cache_options": {"enabled": "on"}},
    )

    # Verify first response has non-zero usage (not from cache)
    assert result1.usage.prompt_tokens > 0
    assert result1.usage.total_tokens > 0
    assert isinstance(result1.data[0].embedding, str)  # base64 encoded

    # Wait briefly for cache write to complete
    await asyncio.sleep(2)

    # Second request with same parameters - should hit cache
    result2 = await async_openai_client.embeddings.create(
        input=input_text,
        model="tensorzero::embedding_model_name::text-embedding-3-small",
        encoding_format="base64",
        extra_body={"tensorzero::cache_options": {"enabled": "on"}},
    )

    # Verify second response has zero usage (from cache)
    assert result2.usage.prompt_tokens == 0
    assert result2.usage.total_tokens == 0
    assert isinstance(result2.data[0].embedding, str)  # base64 encoded
    assert result1.data[0].embedding == result2.data[0].embedding  # Same embedding
