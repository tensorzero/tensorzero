"""Shared test utilities for OTLP traces extra headers tests."""

import os
import time

import requests


def verify_otlp_header_in_tempo(
    inference_id: str,
    test_value: str,
    span_name: str,
) -> None:
    """
    Verify that a custom OTLP header value appears in a Tempo trace.

    Args:
        inference_id: The inference ID to search for
        test_value: The expected header value to find
        span_name: The span name to search within (e.g., "POST /inference")

    Raises:
        AssertionError: If the trace or header value is not found
    """
    tempo_url = os.environ.get("TENSORZERO_TEMPO_URL", "http://localhost:3200")
    start_time = int(time.time()) - 60  # Look back 60 seconds
    end_time = int(time.time())

    # Search for trace
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

    # Find the span and check for our custom header
    found_header = False
    for batch in trace_data.get("batches", []):
        for scope_span in batch.get("scopeSpans", []):
            for span in scope_span.get("spans", []):
                if span.get("name") == span_name:
                    for attr in span.get("attributes", []):
                        if attr.get("key") == "tensorzero.custom_key":
                            attr_value = attr.get("value", {}).get("stringValue")
                            if attr_value == test_value:
                                found_header = True
                                break

    assert found_header, f"Custom OTLP header value '{test_value}' not found in Tempo trace"
