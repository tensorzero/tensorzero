"""
Tests for datapoint rendering functionality in the TensorZero client.

These tests cover:
- Rendering datapoints via experimental_render_samples

To run:
```
pytest test_datapoints.py
```
or
```
uv run pytest test_datapoints.py
```
"""

import json

import pytest
from tensorzero import (
    AsyncTensorZeroGateway,
    ContentBlockChatOutputText,
    CreateDatapointRequestChat,
    CreateDatapointRequestJson,
    Input,
    InputMessage,
    InputMessageContentTemplate,
    InputMessageContentText,
    JsonDatapointOutputUpdate,
    ListDatapointsRequest,
    TensorZeroGateway,
    Text,
)
from uuid_utils import uuid7


def test_sync_render_datapoints(embedded_sync_client: TensorZeroGateway):
    """Test rendering datapoints using experimental_render_samples."""
    dataset_name = f"test_render_{uuid7()}"

    # Insert some datapoints
    requests = [
        CreateDatapointRequestChat(
            function_name="basic_test",
            input=Input(
                system={"assistant_name": "TestBot"},
                messages=[
                    InputMessage(
                        role="user",
                        content=[InputMessageContentText(text="Hello, world!")],
                    )
                ],
            ),
            output=[ContentBlockChatOutputText(text="Hello! How can I help you today?")],
        ),
        CreateDatapointRequestJson(
            function_name="json_success",
            input=Input(
                system={"assistant_name": "JsonBot"},
                messages=[
                    InputMessage(
                        role="user",
                        content=[InputMessageContentTemplate(name="user", arguments={"country": "France"})],
                    )
                ],
            ),
            output=JsonDatapointOutputUpdate(raw=json.dumps({"answer": "Paris"})),
            output_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            },
        ),
    ]

    response = embedded_sync_client.create_datapoints(dataset_name=dataset_name, requests=requests)
    datapoint_ids = response.ids
    assert len(datapoint_ids) == 2

    # List the inserted datapoints
    listed_response = embedded_sync_client.list_datapoints(
        dataset_name=dataset_name,
        request=ListDatapointsRequest(limit=10),
    )
    listed_datapoints = listed_response.datapoints
    assert len(listed_datapoints) == 2

    # Render the datapoints using experimental_render_samples
    rendered_samples = embedded_sync_client.experimental_render_samples(
        stored_samples=listed_datapoints,
        variants={"basic_test": "test", "json_success": "test"},
    )

    assert len(rendered_samples) == 2

    # Verify the chat datapoint was rendered correctly
    chat_sample = next(rs for rs in rendered_samples if rs.function_name == "basic_test")
    assert chat_sample.episode_id is None
    assert chat_sample.inference_id is None
    assert chat_sample.input.system == "You are a helpful and friendly assistant named TestBot"
    assert len(chat_sample.input.messages) == 1
    assert chat_sample.input.messages[0].role == "user"
    assert len(chat_sample.input.messages[0].content) == 1
    assert isinstance(chat_sample.input.messages[0].content[0], Text)
    assert chat_sample.input.messages[0].content[0].text == "Hello, world!"

    # Verify the json datapoint was rendered correctly
    json_sample = next(rs for rs in rendered_samples if rs.function_name == "json_success")
    assert json_sample.episode_id is None
    assert json_sample.inference_id is None
    assert json_sample.input.system is not None
    assert "JsonBot" in json_sample.input.system
    assert len(json_sample.input.messages) == 1
    assert json_sample.input.messages[0].role == "user"
    assert len(json_sample.input.messages[0].content) == 1
    assert isinstance(json_sample.input.messages[0].content[0], Text)
    assert json_sample.input.messages[0].content[0].text == "What is the name of the capital city of France?"

    # Clean up
    embedded_sync_client.delete_datapoints(dataset_name=dataset_name, ids=datapoint_ids)


@pytest.mark.asyncio
async def test_async_render_datapoints(
    embedded_async_client: AsyncTensorZeroGateway,
):
    """Test rendering datapoints using experimental_render_samples (async version)."""
    dataset_name = f"test_render_async_{uuid7()}"

    # Insert some datapoints
    requests = [
        CreateDatapointRequestChat(
            function_name="basic_test",
            input=Input(
                system={"assistant_name": "AsyncBot"},
                messages=[
                    InputMessage(
                        role="user",
                        content=[InputMessageContentText(text="What's the weather like?")],
                    )
                ],
            ),
            output=[
                ContentBlockChatOutputText(
                    text="I don't have access to current weather data.",
                )
            ],
        ),
        CreateDatapointRequestJson(
            function_name="json_success",
            input=Input(
                system={"assistant_name": "DataBot"},
                messages=[
                    InputMessage(
                        role="user",
                        content=[InputMessageContentTemplate(name="user", arguments={"country": "Italy"})],
                    )
                ],
            ),
            output=JsonDatapointOutputUpdate(raw=json.dumps({"answer": "Rome"})),
            output_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            },
        ),
    ]

    response = await embedded_async_client.create_datapoints(dataset_name=dataset_name, requests=requests)
    datapoint_ids = response.ids
    assert len(datapoint_ids) == 2

    # List the inserted datapoints
    listed_response = await embedded_async_client.list_datapoints(
        dataset_name=dataset_name,
        request=ListDatapointsRequest(limit=10),
    )
    listed_datapoints = listed_response.datapoints
    assert len(listed_datapoints) == 2

    # Render the datapoints using experimental_render_samples
    rendered_samples = await embedded_async_client.experimental_render_samples(
        stored_samples=listed_datapoints,
        variants={"basic_test": "test", "json_success": "test"},
    )

    assert len(rendered_samples) == 2

    # Verify the chat datapoint was rendered correctly
    chat_sample = next(rs for rs in rendered_samples if rs.function_name == "basic_test")
    assert chat_sample.episode_id is None
    assert chat_sample.inference_id is None
    assert chat_sample.input.system == "You are a helpful and friendly assistant named AsyncBot"
    assert len(chat_sample.input.messages) == 1
    assert chat_sample.input.messages[0].role == "user"
    assert len(chat_sample.input.messages[0].content) == 1
    assert isinstance(chat_sample.input.messages[0].content[0], Text)
    assert chat_sample.input.messages[0].content[0].text == "What's the weather like?"

    # Verify the json datapoint was rendered correctly
    json_sample = next(rs for rs in rendered_samples if rs.function_name == "json_success")
    assert json_sample.episode_id is None
    assert json_sample.inference_id is None
    assert json_sample.input.system is not None
    assert "DataBot" in json_sample.input.system
    assert len(json_sample.input.messages) == 1
    assert json_sample.input.messages[0].role == "user"
    assert len(json_sample.input.messages[0].content) == 1
    assert isinstance(json_sample.input.messages[0].content[0], Text)
    assert json_sample.input.messages[0].content[0].text == "What is the name of the capital city of Italy?"

    # Clean up
    await embedded_async_client.delete_datapoints(dataset_name=dataset_name, ids=datapoint_ids)


def test_sync_render_filtered_datapoints(
    embedded_sync_client: TensorZeroGateway,
):
    """Test rendering only specific datapoints by filtering function name."""
    dataset_name = f"test_render_filter_{uuid7()}"

    # Insert datapoints for different functions
    requests = [
        CreateDatapointRequestChat(
            function_name="basic_test",
            input=Input(
                system={"assistant_name": "FilterBot"},
                messages=[
                    InputMessage(
                        role="user",
                        content=[InputMessageContentText(text="Test message 1")],
                    )
                ],
            ),
        ),
        CreateDatapointRequestChat(
            function_name="basic_test",
            input=Input(
                system={"assistant_name": "FilterBot"},
                messages=[
                    InputMessage(
                        role="user",
                        content=[InputMessageContentText(text="Test message 2")],
                    )
                ],
            ),
        ),
        CreateDatapointRequestJson(
            function_name="json_success",
            input=Input(
                system={"assistant_name": "JsonFilter"},
                messages=[
                    InputMessage(
                        role="user",
                        content=[InputMessageContentTemplate(name="user", arguments={"country": "Spain"})],
                    ),
                ],
            ),
        ),
    ]

    response = embedded_sync_client.create_datapoints(dataset_name=dataset_name, requests=requests)
    datapoint_ids = response.ids
    assert len(datapoint_ids) == 3

    # List only the basic_test datapoints
    listed_response = embedded_sync_client.list_datapoints(
        dataset_name=dataset_name,
        request=ListDatapointsRequest(function_name="basic_test", limit=10),
    )
    chat_datapoints = listed_response.datapoints
    assert len(chat_datapoints) == 2

    # Render only the chat datapoints
    rendered_samples = embedded_sync_client.experimental_render_samples(
        stored_samples=chat_datapoints,
        variants={"basic_test": "test"},
    )

    assert len(rendered_samples) == 2
    assert all(rs.function_name == "basic_test" for rs in rendered_samples)
    assert all(rs.episode_id is None for rs in rendered_samples)
    assert all(rs.inference_id is None for rs in rendered_samples)
    for rs in rendered_samples:
        assert isinstance(rs.input.system, str)
        assert "FilterBot" in rs.input.system

    # Clean up
    embedded_sync_client.delete_datapoints(dataset_name=dataset_name, ids=datapoint_ids)
