"""
Tests for OTLP traces extra headers using the TensorZero SDK

These tests verify that custom OTLP headers are correctly sent to Tempo
when using the native TensorZero Python SDK.
"""

import asyncio
import time

import pytest
from tensorzero import AsyncTensorZeroGateway, TensorZeroGateway
from tensorzero.types import ChatInferenceResponse
from uuid_utils.compat import uuid7

from .helpers import verify_otlp_header_in_tempo


@pytest.mark.tempo
def test_otlp_traces_extra_headers_tempo():
    """Test that OTLP headers are actually sent to Tempo (requires Tempo running and HTTP gateway)."""
    # Only use HTTP gateway for this test (embedded doesn't send to external Tempo)
    client = TensorZeroGateway.build_http(
        gateway_url="http://localhost:3000",
        verbose_errors=True,
    )

    # Use a unique header value to identify this specific trace
    test_value = f"python-test-{uuid7()}"

    result = client.inference(
        function_name="basic_test",
        variant_name="openai",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "What is 2+2?"}],
        },
        otlp_traces_extra_headers={
            "x-dummy-tensorzero": test_value,
        },
    )

    assert isinstance(result, ChatInferenceResponse)
    inference_id = str(result.inference_id)

    # Wait for trace to be exported to Tempo (same as Rust e2e tests)
    time.sleep(25)

    # Verify the custom header appears in the Tempo trace
    verify_otlp_header_in_tempo(inference_id, test_value, "POST /inference")

    client.close()


@pytest.mark.tempo
@pytest.mark.asyncio
async def test_async_otlp_traces_extra_headers_tempo():
    """Test that OTLP headers are actually sent to Tempo with async client (requires Tempo running and HTTP gateway)."""
    # Only use HTTP gateway for this test (embedded doesn't send to external Tempo)
    client = AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000",
        verbose_errors=True,
        async_setup=False,
    )
    assert isinstance(client, AsyncTensorZeroGateway)

    # Use a unique header value to identify this specific trace
    test_value = f"python-async-test-{uuid7()}"

    result = await client.inference(
        function_name="basic_test",
        variant_name="openai",
        input={
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "What is 3+3?"}],
        },
        otlp_traces_extra_headers={
            "x-dummy-tensorzero": test_value,
        },
    )

    assert isinstance(result, ChatInferenceResponse)
    inference_id = str(result.inference_id)

    # Wait for trace to be exported to Tempo (same as Rust e2e tests)
    await asyncio.sleep(25)

    # Verify the custom header appears in the Tempo trace
    verify_otlp_header_in_tempo(inference_id, test_value, "POST /inference")

    await client.close()
