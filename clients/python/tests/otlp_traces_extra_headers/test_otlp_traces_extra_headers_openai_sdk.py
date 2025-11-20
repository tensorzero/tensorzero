"""
Tests for OTLP traces extra headers using the OpenAI SDK

These tests verify that custom OTLP headers can be sent via the OpenAI SDK's
extra_headers parameter to the TensorZero OpenAI-compatible endpoint and are
correctly exported to Tempo.
"""

import asyncio
import time

import pytest
from openai import AsyncOpenAI, OpenAI
from uuid_utils.compat import uuid7

from .helpers import verify_otlp_header_in_tempo


@pytest.mark.tempo
@pytest.mark.asyncio
async def test_async_openai_compatible_otlp_traces_extra_headers_tempo():
    """Test that OTLP headers are sent to Tempo via OpenAI-compatible endpoint with async client."""
    # Use HTTP gateway directly (not patched client)
    async with AsyncOpenAI(api_key="not-used", base_url="http://localhost:3000/openai/v1") as client:
        # Use a unique header value to identify this specific trace
        test_value = f"openai-async-test-{uuid7()}"

        result = await client.chat.completions.create(
            model="tensorzero::function_name::basic_test",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            # type: ignore
                            "tensorzero::arguments": {"assistant_name": "Alfred Pennyworth"},
                        }
                    ],
                },
                {"role": "user", "content": "What is 3+3?"},
            ],
            extra_headers={
                "tensorzero-otlp-traces-extra-header-x-dummy-tensorzero": test_value,
            },
            extra_body={
                "tensorzero::variant_name": "openai",
            },
        )

        inference_id = result.id

        # Wait for trace to be exported to Tempo (same as other Tempo tests)
        await asyncio.sleep(25)

        # Verify the custom header appears in the Tempo trace
        verify_otlp_header_in_tempo(inference_id, test_value, "POST /openai/v1/chat/completions")


@pytest.mark.tempo
def test_sync_openai_compatible_otlp_traces_extra_headers_tempo():
    """Test that OTLP headers are sent to Tempo via OpenAI-compatible endpoint with sync client."""
    # Use HTTP gateway directly
    client = OpenAI(api_key="not-used", base_url="http://localhost:3000/openai/v1")

    # Use a unique header value to identify this specific trace
    test_value = f"openai-sync-test-{uuid7()}"

    result = client.chat.completions.create(
        model="tensorzero::function_name::basic_test",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        # type: ignore
                        "tensorzero::arguments": {"assistant_name": "Alfred Pennyworth"},
                    }
                ],
            },
            {"role": "user", "content": "What is 2+2?"},
        ],
        extra_headers={
            "tensorzero-otlp-traces-extra-header-x-dummy-tensorzero": test_value,
        },
        extra_body={
            "tensorzero::variant_name": "openai",
        },
    )

    inference_id = result.id

    # Wait for trace to be exported to Tempo (same as other Tempo tests)
    time.sleep(25)

    # Verify the custom header appears in the Tempo trace
    verify_otlp_header_in_tempo(inference_id, test_value, "POST /openai/v1/chat/completions")

    client.close()
