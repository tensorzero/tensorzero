"""
Tests for W3C `traceparent` propagation via the TensorZero Python SDK.

These tests verify that a `traceparent` header passed to `inference()` is
attached to the request to the gateway, and that the resulting trace in Tempo
is rooted at the trace_id the caller supplied.
"""

import asyncio
import base64
import os
import secrets
import time

import pytest
import requests
from tensorzero import AsyncTensorZeroGateway, TensorZeroGateway
from tensorzero.types import ChatInferenceResponse

GATEWAY_URL = os.environ.get("TENSORZERO_GATEWAY_URL", "http://localhost:3000")
TEMPO_URL = os.environ.get("TENSORZERO_TEMPO_URL", "http://localhost:3200")


def _random_traceparent() -> tuple[str, str, str]:
    """Build a W3C traceparent with fresh ids. Returns (header, trace_id, span_id)."""
    trace_id = secrets.token_hex(16)
    span_id = secrets.token_hex(8)
    header = f"00-{trace_id}-{span_id}-01"
    return header, trace_id, span_id


def _assert_trace_rooted_at(inference_id: str, expected_trace_id: str) -> None:
    """Look up the inference in Tempo by inference_id tag and confirm the
    `POST /inference` span's traceId matches `expected_trace_id`."""
    start_time = int(time.time()) - 60
    end_time = int(time.time())

    search_url = f"{TEMPO_URL}/api/search?tags=inference_id={inference_id}&start={start_time}&end={end_time}"
    search_response = requests.get(search_url, timeout=10)
    assert search_response.status_code == 200, f"Failed to search Tempo: {search_response.text}"
    traces = search_response.json().get("traces", [])
    assert traces, f"No traces found for inference_id {inference_id}"

    trace_id = traces[0]["traceID"]
    trace_response = requests.get(f"{TEMPO_URL}/api/traces/{trace_id}", timeout=10)
    assert trace_response.status_code == 200, f"Failed to get trace: {trace_response.text}"

    trace_data = trace_response.json()
    found = False
    for batch in trace_data.get("batches", []):
        for scope_span in batch.get("scopeSpans", []):
            for span in scope_span.get("spans", []):
                if span.get("name") != "POST /inference":
                    continue
                # Tempo returns trace_id as base64-encoded binary.
                decoded = base64.b64decode(span["traceId"]).hex()
                assert decoded == expected_trace_id, f"Expected trace_id {expected_trace_id}, got {decoded}"
                found = True
    assert found, "No `POST /inference` span found in the trace"


@pytest.mark.tempo
def test_traceparent_sync():
    header, trace_id, _ = _random_traceparent()

    with TensorZeroGateway.build_http(
        gateway_url=GATEWAY_URL,
        verbose_errors=True,
    ) as client:
        result = client.inference(
            function_name="basic_test",
            variant_name="openai",
            input={
                "system": {"assistant_name": "Alfred Pennyworth"},
                "messages": [{"role": "user", "content": "What is 2+2?"}],
            },
            gateway_http_headers={"traceparent": header},
        )

    assert isinstance(result, ChatInferenceResponse)
    time.sleep(25)
    _assert_trace_rooted_at(str(result.inference_id), trace_id)


@pytest.mark.tempo
@pytest.mark.asyncio
async def test_traceparent_async():
    header, trace_id, _ = _random_traceparent()

    client = AsyncTensorZeroGateway.build_http(
        gateway_url=GATEWAY_URL,
        verbose_errors=True,
        async_setup=False,
    )
    assert isinstance(client, AsyncTensorZeroGateway)
    try:
        result = await client.inference(
            function_name="basic_test",
            variant_name="openai",
            input={
                "system": {"assistant_name": "Alfred Pennyworth"},
                "messages": [{"role": "user", "content": "What is 3+3?"}],
            },
            gateway_http_headers={"traceparent": header},
        )
    finally:
        await client.close()

    assert isinstance(result, ChatInferenceResponse)
    await asyncio.sleep(25)
    _assert_trace_rooted_at(str(result.inference_id), trace_id)


def test_invalid_header_name_raises():
    """Invalid header names should raise at the Python boundary."""
    client = TensorZeroGateway.build_http(
        gateway_url=GATEWAY_URL,
        verbose_errors=True,
    )
    with pytest.raises(Exception):
        client.inference(
            function_name="basic_test",
            variant_name="openai",
            input={
                "system": {"assistant_name": "Alfred Pennyworth"},
                "messages": [{"role": "user", "content": "hi"}],
            },
            gateway_http_headers={"bad header name with spaces": "value"},
        )
    client.close()
