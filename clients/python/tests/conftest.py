# pyright: reportDeprecated=false
import inspect
import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast

import pytest
import pytest_asyncio
from openai import AsyncOpenAI
from pytest import FixtureRequest
from tensorzero import (
    AsyncTensorZeroGateway,
    ChatCompletionInferenceParams,
    ChatDatapointInsert,
    ContentBlockChatOutputText,
    FunctionTool,
    InferenceParams,
    JsonDatapointInsert,
    JsonInferenceOutput,
    RenderedSample,
    StoredInferenceChat,
    StoredInferenceJson,
    StoredInput,
    StoredInputMessage,
    StoredInputMessageContentTemplate,
    StoredInputMessageContentText,
    StoredInputMessageContentThought,
    TensorZeroGateway,
    patch_openai_client,
)
from tensorzero.util import uuid7


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "mock: tests that require the mock provider API")


TEST_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../tensorzero-core/tests/e2e/config/tensorzero.*.toml",
)

CLICKHOUSE_URL = "http://chuser:chpassword@localhost:8123/tensorzero_e2e_tests"


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
    chat_inference = StoredInferenceChat(
        function_name="basic_test",
        variant_name="default",
        input=StoredInput(
            system={"assistant_name": "foo"},
            messages=[
                StoredInputMessage(
                    role="user",
                    content=[
                        StoredInputMessageContentThought(type="thought", text="hmmm"),
                        StoredInputMessageContentText(type="text", text="bar"),
                    ],
                )
            ],
        ),
        output=[ContentBlockChatOutputText(text="Hello world")],
        episode_id=str(uuid7()),
        inference_id=str(uuid7()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        inference_params=InferenceParams(chat_completion=ChatCompletionInferenceParams()),
        additional_tools=[
            FunctionTool(
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
        dispreferred_outputs=[],
        tags={"test_key": "test_value"},
    )
    json_inference = StoredInferenceJson(
        function_name="json_success",
        variant_name="dummy",
        input=StoredInput(
            system={"assistant_name": "Dr. Mehta"},
            messages=[
                StoredInputMessage(
                    role="user",
                    content=[StoredInputMessageContentText(type="text", text='{"country": "Japan"}')],
                )
            ],
        ),
        output=JsonInferenceOutput(parsed={"answer": "Tokyo"}, raw='{"answer": "Tokyo"}'),
        episode_id=str(uuid7()),
        inference_id=str(uuid7()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        inference_params=InferenceParams(chat_completion=ChatCompletionInferenceParams()),
        extra_body=[],
        output_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        },
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
    chat_inference = StoredInferenceChat(
        function_name="basic_test",
        variant_name="default",
        input=StoredInput(
            system={"assistant_name": "foo"},
            messages=[
                StoredInputMessage(
                    role="user",
                    content=[StoredInputMessageContentText(type="text", text="What is the capital of France?")],
                )
            ],
        ),
        output=[ContentBlockChatOutputText(text="The capital of France is Paris.")],
        episode_id=str(uuid7()),
        inference_id=str(uuid7()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        inference_params=InferenceParams(chat_completion=ChatCompletionInferenceParams()),
        tool_choice="none",
        parallel_tool_calls=False,
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
    json_inference = StoredInferenceJson(
        function_name="json_success",
        variant_name="dummy",
        input=StoredInput(
            system={"assistant_name": "Dr. Mehta"},
            messages=[
                StoredInputMessage(
                    role="user",
                    content=[
                        StoredInputMessageContentTemplate(type="template", name="user", arguments={"country": "Japan"})
                    ],
                )
            ],
        ),
        output=JsonInferenceOutput(parsed={"answer": "Tokyo"}, raw='{"answer": "Tokyo"}'),
        episode_id=str(uuid7()),
        inference_id=str(uuid7()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        inference_params=InferenceParams(chat_completion=ChatCompletionInferenceParams()),
        extra_body=[],
        output_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        },
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
@pytest_asyncio.fixture(params=[OpenAIClientType.HttpGateway, OpenAIClientType.PatchedClient])
async def async_openai_client(request: FixtureRequest):
    if request.param == OpenAIClientType.HttpGateway:
        async with AsyncOpenAI(api_key="donotuse", base_url="http://localhost:3000/openai/v1") as client:
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


def _load_json_datapoints_from_fixture(fixture_path: Path, dataset_filter: str) -> List[JsonDatapointInsert]:
    """Load JSON datapoints from a JSONL fixture file."""
    datapoints: List[JsonDatapointInsert] = []
    with open(fixture_path) as f:
        for line in f:
            if not line.strip():
                continue
            data: Dict[str, Any] = json.loads(line)
            # Only load datapoints for the specified dataset
            if data.get("dataset_name") != dataset_filter:
                continue

            # Parse the JSON strings in the fixture
            input_data: Any = json.loads(data["input"])

            # Handle output - it may be in {"raw": "...", "parsed": {...}} format
            output_data: Optional[Any] = None
            if data.get("output"):
                parsed_output: Any = json.loads(data["output"])
                # If output has "parsed" field, extract it; otherwise use as-is
                if isinstance(parsed_output, dict) and "parsed" in parsed_output:
                    output_data = cast(Any, parsed_output["parsed"])
                else:
                    output_data = cast(Any, parsed_output)

            output_schema: Optional[Any] = json.loads(data["output_schema"]) if data.get("output_schema") else None

            datapoints.append(
                JsonDatapointInsert(
                    function_name=data["function_name"],
                    input=input_data,
                    output=output_data,
                    output_schema=output_schema,
                    tags=data.get("tags"),
                )
            )
    return datapoints


def _load_chat_datapoints_from_fixture(fixture_path: Path, dataset_filter: str) -> List[ChatDatapointInsert]:
    """Load Chat datapoints from a JSONL fixture file."""
    datapoints: List[ChatDatapointInsert] = []
    with open(fixture_path) as f:
        for line in f:
            if not line.strip():
                continue
            data: Dict[str, Any] = json.loads(line)
            # Only load datapoints for the specified dataset
            if data.get("dataset_name") != dataset_filter:
                continue

            # Parse the JSON strings in the fixture
            input_data: Any = json.loads(data["input"])
            output_data: Optional[Any] = json.loads(data["output"]) if data.get("output") else None

            datapoints.append(
                ChatDatapointInsert(
                    function_name=data["function_name"],
                    input=input_data,
                    output=output_data,
                    tags=data.get("tags"),
                )
            )
    return datapoints


@pytest.fixture
def evaluation_datasets(
    embedded_sync_client: TensorZeroGateway,
) -> Iterator[Dict[str, str]]:
    """
    Seed datasets needed for evaluation tests.

    Returns a mapping from original dataset names to unique test dataset names.
    This ensures test isolation and prevents conflicts between concurrent test runs.
    """
    fixtures_dir = Path(__file__).resolve().parents[3] / "tensorzero-core/fixtures/datasets"

    # Create unique dataset names for this test run
    dataset_mapping = {
        "extract_entities_0.8": f"extract_entities_0.8_{uuid7()}",
        "good-haikus-no-output": f"good-haikus-no-output_{uuid7()}",
    }

    # Load and insert JSON datapoints (for entity_extraction evaluation)
    json_fixture_path = fixtures_dir / "json_datapoint_fixture.jsonl"
    json_datapoints = _load_json_datapoints_from_fixture(json_fixture_path, "extract_entities_0.8")
    if json_datapoints:
        embedded_sync_client.create_datapoints_legacy(
            dataset_name=dataset_mapping["extract_entities_0.8"],
            datapoints=cast(
                Sequence[Union[ChatDatapointInsert, JsonDatapointInsert]],
                json_datapoints,
            ),
        )

    # Load and insert Chat datapoints (for haiku evaluation)
    chat_fixture_path = fixtures_dir / "chat_datapoint_fixture.jsonl"
    chat_datapoints = _load_chat_datapoints_from_fixture(chat_fixture_path, "good-haikus-no-output")
    if chat_datapoints:
        embedded_sync_client.create_datapoints_legacy(
            dataset_name=dataset_mapping["good-haikus-no-output"],
            datapoints=cast(
                Sequence[Union[ChatDatapointInsert, JsonDatapointInsert]],
                chat_datapoints,
            ),
        )

    yield dataset_mapping

    # Cleanup is optional - datasets will be isolated by unique names
    # and ClickHouse test database can be cleaned between full test runs
