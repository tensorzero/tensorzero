"""
Tests for OTLP traces extra headers using the OpenAI SDK

These tests verify that custom OTLP headers can be sent via the OpenAI SDK's
extra_headers parameter to the TensorZero OpenAI-compatible endpoint and are
correctly exported to Tempo.
"""

import asyncio
import os
import time

import pytest
import requests
from openai import AsyncOpenAI, OpenAI
from uuid_utils.compat import uuid7


@pytest.mark.tempo
@pytest.mark.asyncio
async def test_async_openai_compatible_otlp_traces_extra_headers_tempo():
    """Test that OTLP headers are sent to Tempo via OpenAI-compatible endpoint with async client."""
    # Use HTTP gateway directly (not patched client)
    async with AsyncOpenAI(api_key="donotuse", base_url="http://localhost:3000/openai/v1") as client:
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

        # Query Tempo for the trace
        tempo_url = os.environ.get("TENSORZERO_TEMPO_URL", "http://localhost:3200")
        start_time = int(time.time()) - 60  # Look back 60 seconds
        end_time = int(time.time())

        search_url = f"{tempo_url}/api/search?tags=inference_id={inference_id}&start={start_time}&end={end_time}"
        search_response = requests.get(search_url, timeout=10)
        assert search_response.status_code == 200, f"Failed to search Tempo: {search_response.text}"

        tempo_traces = search_response.json()
        assert len(tempo_traces.get("traces", [])) > 0, f"No traces found for inference_id {inference_id}"

        trace_id = tempo_traces["traces"][0]["traceID"]

        # Get trace details
        trace_url = f"{tempo_url}/api/traces/{trace_id}"
        trace_response = requests.get(trace_url, timeout=10)
        assert trace_response.status_code == 200, f"Failed to get trace: {trace_response.text}"

        trace_data = trace_response.json()

        # Find the parent span (POST /openai/v1/chat/completions) and check for our custom header
        found_header = False
        for batch in trace_data.get("batches", []):
            for scope_span in batch.get("scopeSpans", []):
                for span in scope_span.get("spans", []):
                    if span.get("name") == "POST /openai/v1/chat/completions":
                        # Check span attributes for our custom header value
                        for attr in span.get("attributes", []):
                            if attr.get("key") == "tensorzero.custom_key":
                                attr_value = attr.get("value", {}).get("stringValue")
                                if attr_value == test_value:
                                    found_header = True
                                    break

        assert found_header, f"Custom OTLP header value '{test_value}' not found in Tempo trace"


@pytest.mark.tempo
def test_sync_openai_compatible_otlp_traces_extra_headers_tempo():
    """Test that OTLP headers are sent to Tempo via OpenAI-compatible endpoint with sync client."""
    # Use HTTP gateway directly
    client = OpenAI(api_key="donotuse", base_url="http://localhost:3000/openai/v1")

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

    # Query Tempo for the trace
    tempo_url = os.environ.get("TENSORZERO_TEMPO_URL", "http://localhost:3200")
    start_time = int(time.time()) - 60  # Look back 60 seconds
    end_time = int(time.time())

    search_url = f"{tempo_url}/api/search?tags=inference_id={inference_id}&start={start_time}&end={end_time}"
    search_response = requests.get(search_url, timeout=10)
    assert search_response.status_code == 200, f"Failed to search Tempo: {search_response.text}"

    tempo_traces = search_response.json()
    assert len(tempo_traces.get("traces", [])) > 0, f"No traces found for inference_id {inference_id}"

    trace_id = tempo_traces["traces"][0]["traceID"]

    # Get trace details
    trace_url = f"{tempo_url}/api/traces/{trace_id}"
    trace_response = requests.get(trace_url, timeout=10)
    assert trace_response.status_code == 200, f"Failed to get trace: {trace_response.text}"

    trace_data = trace_response.json()

    # Find the parent span (POST /openai/v1/chat/completions) and check for our custom header
    found_header = False
    for batch in trace_data.get("batches", []):
        for scope_span in batch.get("scopeSpans", []):
            for span in scope_span.get("spans", []):
                if span.get("name") == "POST /openai/v1/chat/completions":
                    # Check span attributes for our custom header value
                    for attr in span.get("attributes", []):
                        if attr.get("key") == "tensorzero.custom_key":
                            attr_value = attr.get("value", {}).get("stringValue")
                            if attr_value == test_value:
                                found_header = True
                                break

    assert found_header, f"Custom OTLP header value '{test_value}' not found in Tempo trace"

    client.close()
