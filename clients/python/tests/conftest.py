import inspect
import os
from enum import Enum
from typing import List

import pytest
import pytest_asyncio
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
)
from tensorzero.util import uuid7

TEST_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../tensorzero-core/tests/e2e/tensorzero.toml",
)


class ClientType(Enum):
    HttpGateway = 0
    EmbeddedGateway = 1


@pytest.fixture
def embedded_sync_client():
    with TensorZeroGateway.build_embedded(
        config_file=TEST_CONFIG_FILE,
        clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero-python-e2e",
    ) as client:
        yield client


@pytest_asyncio.fixture
async def embedded_async_client():
    client_fut = AsyncTensorZeroGateway.build_embedded(
        config_file=TEST_CONFIG_FILE,
        clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero-python-e2e",
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
            clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero-python-e2e",
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
            clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero-python-e2e",
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
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "value": "Hello world"},
                    ],
                },
            ],
        },
        output=[Text(text="Hello world")],
        episode_id=uuid7(),
        inference_id=uuid7(),
        tool_params=ToolParams(
            tools_available=[
                Tool(
                    name="test",
                    description="test",
                    parameters={"foo": "bar"},
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
