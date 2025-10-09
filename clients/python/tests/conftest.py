import inspect
import os
from datetime import datetime, timezone
from enum import Enum
from typing import List

import pytest
import pytest_asyncio
from openai import AsyncOpenAI
from pytest import FixtureRequest
from tensorzero import (
    AsyncTensorZeroGateway,
    JsonInferenceOutput,
    RenderedSample,
    StoredInference,
    TensorZeroGateway,
    Text,
    Tool,
    ToolParams,
    patch_openai_client,
)
from tensorzero.util import uuid7

TEST_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../tensorzero-core/tests/e2e/tensorzero.toml",
)

CLICKHOUSE_URL = "http://chuser:chpassword@localhost:8123/tensorzero-python-e2e"


class ClientType(Enum):
    HttpGateway = 0
    EmbeddedGateway = 1


@pytest.fixture
def embedded_sync_client():
    with TensorZeroGateway.build_embedded(
        config_file=TEST_CONFIG_FILE,
        clickhouse_url=CLICKHOUSE_URL,
    ) as client:
        yield client


@pytest_asyncio.fixture
async def embedded_async_client():
    client_fut = AsyncTensorZeroGateway.build_embedded(
        config_file=TEST_CONFIG_FILE,
        clickhouse_url=CLICKHOUSE_URL,
    )
    assert inspect.isawaitable(client_fut)
    async with await client_fut as client:
        yield client


# Shared fixtures for both HTTP and embedded clients
@pytest_asyncio.fixture(params=[ClientType.HttpGateway, ClientType.EmbeddedGateway])
async def async_client(request: FixtureRequest):
    if request.param == ClientType.HttpGateway:
        client_fut = AsyncTensorZeroGateway.build_http(
            gateway_url="http://localhost:3000",
            verbose_errors=True,
        )
        assert inspect.isawaitable(client_fut)
        async with await client_fut as client:
            yield client
    else:
        client_fut = AsyncTensorZeroGateway.build_embedded(
            config_file=TEST_CONFIG_FILE,
            clickhouse_url=CLICKHOUSE_URL,
        )
        assert inspect.isawaitable(client_fut)
        async with await client_fut as client:
            yield client


@pytest.fixture(params=[ClientType.HttpGateway, ClientType.EmbeddedGateway])
def sync_client(request: FixtureRequest):
    if request.param == ClientType.HttpGateway:
        with TensorZeroGateway.build_http(
            gateway_url="http://localhost:3000",
            verbose_errors=True,
        ) as client:
            yield client
    else:
        with TensorZeroGateway.build_embedded(
            config_file=TEST_CONFIG_FILE,
            clickhouse_url=CLICKHOUSE_URL,
        ) as client:
            yield client


@pytest.fixture
def mixed_rendered_samples(
    embedded_sync_client: TensorZeroGateway,
) -> List[RenderedSample]:
    chat_inference = StoredInference(
        type="chat",
        function_name="basic_test",
        variant_name="default",
        input={
            "system": {"assistant_name": "foo"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "thought", "text": "hmmm"},
                        {"type": "text", "value": "bar"},
                    ],
                },
            ],
        },
        output=[Text(text="Hello world")],
        episode_id=uuid7(),
        inference_id=uuid7(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        tool_params=ToolParams(
            tools_available=[
                Tool(
                    name="test",
                    description="test",
                    parameters={
                        "type": "object",
                        "properties": {"foo": {"type": "string", "description": "bar"}},
                        "required": ["foo"],
                    },
                    strict=False,
                )
            ],
            tool_choice="auto",
            parallel_tool_calls=False,
        ),
        output_schema=None,
        dispreferred_outputs=[],
        tags={"test_key": "test_value"},
    )
    json_inference = StoredInference(
        type="json",
        function_name="json_success",
        variant_name="dummy",
        input={
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": {"country": "Japan"}}],
                },
            ],
        },
        output=JsonInferenceOutput(
            parsed={"answer": "Tokyo"}, raw='{"answer": "Tokyo"}'
        ),
        episode_id=uuid7(),
        inference_id=uuid7(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        output_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        },
        tool_params=None,
        dispreferred_outputs=[],
        tags={"test_key": "test_value"},
    )
    sample_list = [chat_inference] * 10 + [json_inference] * 10
    return embedded_sync_client.experimental_render_samples(
        stored_samples=sample_list,
        variants={"basic_test": "test", "json_success": "test"},
    )


@pytest.fixture
def chat_function_rendered_samples(
    embedded_sync_client: TensorZeroGateway,
) -> List[RenderedSample]:
    """Fixture for optimization tests - chat function samples without tools."""
    chat_inference = StoredInference(
        type="chat",
        function_name="basic_test",
        variant_name="default",
        input={
            "system": {"assistant_name": "foo"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "value": "What is the capital of France?"}
                    ],
                },
            ],
        },
        output=[Text(text="The capital of France is Paris.")],
        episode_id=uuid7(),
        inference_id=uuid7(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        tool_params=ToolParams(
            tools_available=[],  # No tools for DICL compatibility
            tool_choice="none",
            parallel_tool_calls=False,
        ),
        output_schema=None,
        dispreferred_outputs=[],
        tags={"test_key": "test_value"},
    )
    # Create 20 samples from the same function
    sample_list = [chat_inference] * 20
    return embedded_sync_client.experimental_render_samples(
        stored_samples=sample_list,
        variants={"basic_test": "test"},
    )


@pytest.fixture
def json_function_rendered_samples(
    embedded_sync_client: TensorZeroGateway,
) -> List[RenderedSample]:
    """Fixture for optimization tests - JSON function samples."""
    json_inference = StoredInference(
        type="json",
        function_name="json_success",
        variant_name="dummy",
        input={
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": {"country": "Japan"}}],
                },
            ],
        },
        output=JsonInferenceOutput(
            parsed={"answer": "Tokyo"}, raw='{"answer": "Tokyo"}'
        ),
        episode_id=uuid7(),
        inference_id=uuid7(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        output_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        },
        tool_params=None,  # JSON functions don't have tool_params
        dispreferred_outputs=[],
        tags={"test_key": "test_value"},
    )
    # Create 20 samples from the same function
    sample_list = [json_inference] * 20
    return embedded_sync_client.experimental_render_samples(
        stored_samples=sample_list,
        variants={"json_success": "test"},
    )


class OpenAIClientType(Enum):
    HttpGateway = 0
    PatchedClient = 1


# Shared fixtures for both HTTP and embedded clients
@pytest_asyncio.fixture(
    params=[OpenAIClientType.HttpGateway, OpenAIClientType.PatchedClient]
)
async def async_openai_client(request: FixtureRequest):
    if request.param == OpenAIClientType.HttpGateway:
        async with AsyncOpenAI(
            api_key="donotuse", base_url="http://localhost:3000/openai/v1"
        ) as client:
            yield client
    else:
        async with AsyncOpenAI(api_key="donotuse") as client:
            await patch_openai_client(  # type: ignore[reportGeneralTypeIssues]
                client,
                config_file=TEST_CONFIG_FILE,
                clickhouse_url=CLICKHOUSE_URL,
                async_setup=True,
            )
            yield client
