"""
Tests for OTLP traces extra headers using the TensorZero SDK

These tests verify that custom OTLP headers are correctly sent to Tempo
when using the native TensorZero Python SDK.
"""

import asyncio
import os
import time

import pytest
import requests
from tensorzero import AsyncTensorZeroGateway, TensorZeroGateway
from tensorzero.types import ChatInferenceResponse
from uuid_utils.compat import uuid7


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

    # Find the parent span (POST /inference) and check for our custom header in attributes
    found_header = False
    for batch in trace_data.get("batches", []):
        for scope_span in batch.get("scopeSpans", []):
            for span in scope_span.get("spans", []):
                if span.get("name") == "POST /inference":
                    # Check span attributes for our custom header value
                    for attr in span.get("attributes", []):
                        if attr.get("key") == "tensorzero.custom_key":
                            attr_value = attr.get("value", {}).get("stringValue")
                            if attr_value == test_value:
                                found_header = True
                                break

    assert found_header, f"Custom OTLP header value '{test_value}' not found in Tempo trace"

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

    # Find the parent span (POST /inference) and check for our custom header in attributes
    found_header = False
    for batch in trace_data.get("batches", []):
        for scope_span in batch.get("scopeSpans", []):
            for span in scope_span.get("spans", []):
                if span.get("name") == "POST /inference":
                    # Check span attributes for our custom header value
                    for attr in span.get("attributes", []):
                        if attr.get("key") == "tensorzero.custom_key":
                            attr_value = attr.get("value", {}).get("stringValue")
                            if attr_value == test_value:
                                found_header = True
                                break

    assert found_header, f"Custom OTLP header value '{test_value}' not found in Tempo trace"

    await client.close()
