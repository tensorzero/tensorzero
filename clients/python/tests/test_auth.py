"""
Tests for API key authentication in the TensorZero Python SDK.

These tests verify that:
1. Explicit api_key parameter sends Bearer token
2. TENSORZERO_API_KEY environment variable is used as fallback
3. API key and timeout can be used together
4. No auth header is sent when neither api_key nor env var is set
"""

from typing import Dict, cast

import pytest
from pytest import MonkeyPatch
from pytest_httpserver import HTTPServer
from tensorzero import AsyncTensorZeroGateway, TensorZeroGateway
from werkzeug import Request, Response

# Minimal valid inference response
MOCK_INFERENCE_RESPONSE = {
    "inference_id": "test-inference-id-123",
    "episode_id": "test-episode-id-456",
    "variant_name": "test-variant",
    "content": [{"type": "text", "text": "This is a test response"}],
}


def test_explicit_api_key_sync(httpserver: HTTPServer) -> None:
    """Test that explicit api_key parameter sends Bearer token (sync client)"""
    api_key = "sk-t0-test-key-123"

    # Set up mock server to capture the request
    captured_headers: Dict[str, str] = {}

    def handler(request: Request) -> Response:
        for key, value in request.headers.items():
            captured_headers[key] = value
        return Response(
            response=str(MOCK_INFERENCE_RESPONSE).replace("'", '"'),
            status=200,
            content_type="application/json",
        )

    httpserver.expect_request("/inference", method="POST").respond_with_handler(handler)

    # Create client with explicit api_key
    with TensorZeroGateway.build_http(
        gateway_url=httpserver.url_for("/"),
        api_key=api_key,
    ) as client:
        # Make inference request
        try:
            client.inference(
                function_name="test_function",
                input={"messages": [{"role": "user", "content": "test"}]},
            )
        except Exception:
            # This can fail; we only care that the HTTP header was sent
            pass

    # Verify the Authorization header was sent with correct Bearer token
    assert "Authorization" in captured_headers
    assert captured_headers["Authorization"] == f"Bearer {api_key}"


@pytest.mark.asyncio
async def test_explicit_api_key_async(httpserver: HTTPServer) -> None:
    """Test that explicit api_key parameter sends Bearer token (async client)"""
    api_key = "sk-t0-test-key-async"

    # Set up mock server to capture the request
    captured_headers: Dict[str, str] = {}

    def handler(request: Request) -> Response:
        for key, value in request.headers.items():
            captured_headers[key] = value
        return Response(
            response=str(MOCK_INFERENCE_RESPONSE).replace("'", '"'),
            status=200,
            content_type="application/json",
        )

    httpserver.expect_request("/inference", method="POST").respond_with_handler(handler)

    # Create async client with explicit api_key
    # When async_setup=False, it returns AsyncTensorZeroGateway directly
    # TODO: cast() is a workaround for https://github.com/tensorzero/tensorzero/issues/2074
    client = cast(
        AsyncTensorZeroGateway,
        AsyncTensorZeroGateway.build_http(
            gateway_url=httpserver.url_for("/"),
            api_key=api_key,
            async_setup=False,
        ),
    )

    async with client:
        # Make inference request
        try:
            await client.inference(
                function_name="test_function",
                input={"messages": [{"role": "user", "content": "test"}]},
            )
        except Exception:
            # This can fail; we only care that the HTTP header was sent
            pass

    # Verify the Authorization header was sent
    assert "Authorization" in captured_headers
    assert captured_headers["Authorization"] == f"Bearer {api_key}"


def test_env_var_api_key_sync(httpserver: HTTPServer, monkeypatch: MonkeyPatch) -> None:
    """Test that TENSORZERO_API_KEY environment variable is used (sync)"""
    api_key = "sk-t0-env-key-123"

    # Set environment variable
    monkeypatch.setenv("TENSORZERO_API_KEY", api_key)

    # Set up mock server to capture the request
    captured_headers: Dict[str, str] = {}

    def handler(request: Request) -> Response:
        for key, value in request.headers.items():
            captured_headers[key] = value
        return Response(
            response=str(MOCK_INFERENCE_RESPONSE).replace("'", '"'),
            status=200,
            content_type="application/json",
        )

    httpserver.expect_request("/inference", method="POST").respond_with_handler(handler)

    # Create client WITHOUT explicit api_key (should use env var)
    with TensorZeroGateway.build_http(
        gateway_url=httpserver.url_for("/"),
    ) as client:
        # Make inference request
        try:
            client.inference(
                function_name="test_function",
                input={"messages": [{"role": "user", "content": "test"}]},
            )
        except Exception:
            # This can fail; we only care that the HTTP header was sent
            pass

    # Verify the Authorization header was sent from env var
    assert "Authorization" in captured_headers
    assert captured_headers["Authorization"] == f"Bearer {api_key}"


@pytest.mark.asyncio
async def test_env_var_api_key_async(httpserver: HTTPServer, monkeypatch: MonkeyPatch) -> None:
    """Test that TENSORZERO_API_KEY environment variable is used (async)"""
    api_key = "sk-t0-env-key-async"

    # Set environment variable
    monkeypatch.setenv("TENSORZERO_API_KEY", api_key)

    # Set up mock server to capture the request
    captured_headers: Dict[str, str] = {}

    def handler(request: Request) -> Response:
        for key, value in request.headers.items():
            captured_headers[key] = value
        return Response(
            response=str(MOCK_INFERENCE_RESPONSE).replace("'", '"'),
            status=200,
            content_type="application/json",
        )

    httpserver.expect_request("/inference", method="POST").respond_with_handler(handler)

    # Create async client WITHOUT explicit api_key
    # TODO: cast() is a workaround for https://github.com/tensorzero/tensorzero/issues/2074
    client = cast(
        AsyncTensorZeroGateway,
        AsyncTensorZeroGateway.build_http(
            gateway_url=httpserver.url_for("/"),
            async_setup=False,
        ),
    )

    async with client:
        try:
            await client.inference(
                function_name="test_function",
                input={"messages": [{"role": "user", "content": "test"}]},
            )
        except Exception:
            # This can fail; we only care that the HTTP header was sent
            pass

    # Verify the Authorization header was sent from env var
    assert "Authorization" in captured_headers
    assert captured_headers["Authorization"] == f"Bearer {api_key}"


def test_api_key_with_timeout(httpserver: HTTPServer) -> None:
    """Test that api_key and timeout can be used together"""
    api_key = "sk-t0-timeout-test"

    # Set up mock server to capture the request
    captured_headers: Dict[str, str] = {}

    def handler(request: Request) -> Response:
        for key, value in request.headers.items():
            captured_headers[key] = value
        return Response(
            response=str(MOCK_INFERENCE_RESPONSE).replace("'", '"'),
            status=200,
            content_type="application/json",
        )

    httpserver.expect_request("/inference", method="POST").respond_with_handler(handler)

    # Create client with BOTH api_key AND timeout
    with TensorZeroGateway.build_http(
        gateway_url=httpserver.url_for("/"),
        api_key=api_key,
        timeout=30.0,  # This should work together with api_key
    ) as client:
        try:
            client.inference(
                function_name="test_function",
                input={"messages": [{"role": "user", "content": "test"}]},
            )
        except Exception:
            # This can fail; we only care that the HTTP header was sent
            pass

    # Verify the Authorization header was sent
    # (proving that timeout didn't break api_key functionality)
    assert "Authorization" in captured_headers
    assert captured_headers["Authorization"] == f"Bearer {api_key}"


def test_no_auth_when_no_key(httpserver: HTTPServer, monkeypatch: MonkeyPatch) -> None:
    """Test that no Authorization header is sent when neither api_key nor env var is set"""

    # Ensure environment variable is NOT set
    monkeypatch.delenv("TENSORZERO_API_KEY", raising=False)

    # Set up mock server to capture the request
    captured_headers: Dict[str, str] = {}

    def handler(request: Request) -> Response:
        for key, value in request.headers.items():
            captured_headers[key] = value
        return Response(
            response=str(MOCK_INFERENCE_RESPONSE).replace("'", '"'),
            status=200,
            content_type="application/json",
        )

    httpserver.expect_request("/inference", method="POST").respond_with_handler(handler)

    # Create client WITHOUT api_key and WITHOUT env var
    with TensorZeroGateway.build_http(
        gateway_url=httpserver.url_for("/"),
    ) as client:
        try:
            client.inference(
                function_name="test_function",
                input={"messages": [{"role": "user", "content": "test"}]},
            )
        except Exception:
            # This can fail; we only care that the HTTP header was not sent
            pass

    # Verify NO Authorization header was sent
    assert "Authorization" not in captured_headers
