# pyright: reportDeprecated=false
"""
Tests for datapoint and dataset handling functionality in the TensorZero client.

These tests cover:
- Bulk insertion of datapoints
- Retrieving individual datapoints
- Listing datapoints with filtering
- Rendering datapoints
- Deleting datapoints
- Dataset operations

To run:
```
pytest test_datapoints.py
```
or
```
uv run pytest test_datapoints.py
```
"""

from uuid import UUID

import pytest
from tensorzero import (
    AsyncTensorZeroGateway,
    ChatDatapoint,
    ChatDatapointInsert,
    InputMessageContentTemplate,
    InputMessageContentText,
    JsonDatapoint,
    JsonDatapointInsert,
    TensorZeroError,
    TensorZeroGateway,
    Text,
)
from uuid_utils import uuid7


def test_sync_insert_delete_datapoints(sync_client: TensorZeroGateway):
    dataset_name = f"test_{uuid7()}"
    datapoints = [
        ChatDatapointInsert(
            function_name="basic_test",
            input={
                "system": {"assistant_name": "foo"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "bar"}],
                    }
                ],
            },
            output=[{"type": "text", "text": "foobar"}],
            allowed_tools=None,
            additional_tools=None,
            tool_choice="auto",
            parallel_tool_calls=False,
            tags=None,
        ),
        ChatDatapointInsert(
            function_name="basic_test",
            input={
                "system": {"assistant_name": "Dummy"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "My synthetic input"}],
                    }
                ],
            },
            output=[
                {
                    "type": "tool_call",
                    "name": "get_temperature",
                    "id": "tool_call_id",
                    "arguments": {
                        "location": "New York",
                        "units": "fahrenheit",
                    },
                }
            ],
            additional_tools=[
                {
                    "description": "Get the current temperature in a given location",
                    "parameters": {
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": 'The location to get the temperature for (e.g. "New York")',
                            },
                            "units": {
                                "type": "string",
                                "description": 'The units to get the temperature in (must be "fahrenheit" or "celsius")',
                                "enum": ["fahrenheit", "celsius"],
                            },
                        },
                        "required": ["location"],
                        "additionalProperties": False,
                    },
                    "name": "get_temperature",
                    "strict": False,
                }
            ],
            tool_choice="auto",
            parallel_tool_calls=False,
            allowed_tools=None,
            tags=None,
        ),
        JsonDatapointInsert(
            function_name="json_success",
            input={
                "system": {"assistant_name": "foo"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "arguments": {"country": "US"}}],
                    }
                ],
            },
            output={"answer": "Hello"},
            output_schema=None,
            tags=None,
        ),
        JsonDatapointInsert(
            function_name="json_success",
            input={
                "system": {"assistant_name": "foo"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "arguments": {"country": "US"}}],
                    }
                ],
            },
            output={"response": "Hello"},
            output_schema={
                "type": "object",
                "properties": {"response": {"type": "string"}},
            },
            tags=None,
        ),
    ]
    datapoint_ids = sync_client.create_datapoints_legacy(dataset_name=dataset_name, datapoints=datapoints)
    assert len(datapoint_ids) == 4
    assert isinstance(datapoint_ids[0], UUID)
    assert isinstance(datapoint_ids[1], UUID)
    assert isinstance(datapoint_ids[2], UUID)
    assert isinstance(datapoint_ids[3], UUID)

    # List datapoints filtering by function name
    listed_datapoints = sync_client.list_datapoints_legacy(
        dataset_name=dataset_name,
        function_name="basic_test",
    )
    assert len(listed_datapoints) == 2
    assert all(isinstance(dp, ChatDatapoint) for dp in listed_datapoints)
    assert all(dp.function_name == "basic_test" for dp in listed_datapoints)

    sync_client.delete_datapoint(dataset_name=dataset_name, datapoint_id=datapoint_ids[0])
    sync_client.delete_datapoint(dataset_name=dataset_name, datapoint_id=datapoint_ids[1])
    sync_client.delete_datapoint(dataset_name=dataset_name, datapoint_id=datapoint_ids[2])
    sync_client.delete_datapoint(dataset_name=dataset_name, datapoint_id=datapoint_ids[3])


@pytest.mark.asyncio
async def test_async_insert_delete_datapoints(
    async_client: AsyncTensorZeroGateway,
):
    datapoints = [
        ChatDatapointInsert(
            function_name="basic_test",
            input={
                "system": {"assistant_name": "foo"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "bar"}],
                    }
                ],
            },
        ),
        ChatDatapointInsert(
            function_name="basic_test",
            input={
                "system": {"assistant_name": "Dummy"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "My synthetic input"}],
                    }
                ],
            },
            output=[
                {
                    "type": "tool_call",
                    "id": "tool_call_id",
                    "name": "get_temperature",
                    "arguments": {
                        "location": "New York",
                        "units": "fahrenheit",
                    },
                }
            ],
            additional_tools=[
                {
                    "description": "Get the current temperature in a given location",
                    "parameters": {
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": 'The location to get the temperature for (e.g. "New York")',
                            },
                            "units": {
                                "type": "string",
                                "description": 'The units to get the temperature in (must be "fahrenheit" or "celsius")',
                                "enum": ["fahrenheit", "celsius"],
                            },
                        },
                        "required": ["location"],
                        "additionalProperties": False,
                    },
                    "name": "get_temperature",
                    "strict": False,
                }
            ],
            tool_choice="auto",
            parallel_tool_calls=False,
            allowed_tools=None,
            tags=None,
        ),
        JsonDatapointInsert(
            function_name="json_success",
            input={
                "system": {"assistant_name": "foo"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "arguments": {"country": "US"}}],
                    }
                ],
            },
        ),
        JsonDatapointInsert(
            function_name="json_success",
            input={
                "system": {"assistant_name": "foo"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "arguments": {"country": "US"}}],
                    }
                ],
            },
            output={"response": "Hello"},
            output_schema={
                "type": "object",
                "properties": {"response": {"type": "string"}},
            },
            tags=None,
        ),
    ]
    dataset_name = f"test_{uuid7()}"
    datapoint_ids = await async_client.create_datapoints_legacy(dataset_name=dataset_name, datapoints=datapoints)
    assert len(datapoint_ids) == 4
    assert isinstance(datapoint_ids[0], UUID)
    assert isinstance(datapoint_ids[1], UUID)
    assert isinstance(datapoint_ids[2], UUID)
    assert isinstance(datapoint_ids[3], UUID)

    # Get a chat datapoint
    datapoint = await async_client.get_datapoint(dataset_name=dataset_name, datapoint_id=datapoint_ids[0])
    print(datapoint)
    assert isinstance(datapoint, ChatDatapoint)
    assert datapoint.function_name == "basic_test"
    assert datapoint.input.system == {"assistant_name": "foo"}
    assert datapoint.input.messages is not None and len(datapoint.input.messages) == 1
    assert datapoint.input.messages[0].role == "user"
    assert len(datapoint.input.messages[0].content) == 1
    assert datapoint.input.messages[0].content[0].type == "text"
    assert isinstance(datapoint.input.messages[0].content[0], InputMessageContentText)
    assert datapoint.input.messages[0].content[0].text == "bar"
    assert datapoint.output is None

    # Get a json datapoint
    datapoint = await async_client.get_datapoint(dataset_name=dataset_name, datapoint_id=datapoint_ids[2])
    assert isinstance(datapoint, JsonDatapoint)
    assert datapoint.function_name == "json_success"
    assert datapoint.input.system == {"assistant_name": "foo"}
    assert datapoint.input.messages is not None and len(datapoint.input.messages) == 1
    assert datapoint.input.messages[0].role == "user"
    assert len(datapoint.input.messages[0].content) == 1
    assert datapoint.input.messages[0].content[0].type == "template"
    assert isinstance(datapoint.input.messages[0].content[0], InputMessageContentTemplate)
    assert datapoint.input.messages[0].content[0].arguments == {"country": "US"}
    assert datapoint.output is None
    assert datapoint.is_custom

    # List datapoints
    listed_datapoints = await async_client.list_datapoints_legacy(
        dataset_name=dataset_name,
    )
    assert len(listed_datapoints) == 4
    # Assert that there are 2 chat and 2 json datapoints
    chat_datapoints = [dp for dp in listed_datapoints if isinstance(dp, ChatDatapoint)]  # pyright: ignore[reportUnnecessaryIsInstance]
    json_datapoints = [dp for dp in listed_datapoints if isinstance(dp, JsonDatapoint)]  # pyright: ignore[reportUnnecessaryIsInstance]
    assert len(chat_datapoints) == 2
    assert len(json_datapoints) == 2

    # List datapoints filtering by function name
    listed_datapoints = await async_client.list_datapoints_legacy(
        dataset_name=dataset_name,
        function_name="basic_test",
    )
    assert len(listed_datapoints) == 2
    assert all(isinstance(dp, ChatDatapoint) for dp in listed_datapoints)
    assert all(dp.is_custom for dp in listed_datapoints)
    assert all(dp.function_name == "basic_test" for dp in listed_datapoints)

    await async_client.delete_datapoint(dataset_name=dataset_name, datapoint_id=datapoint_ids[0])
    await async_client.delete_datapoint(dataset_name=dataset_name, datapoint_id=datapoint_ids[1])
    await async_client.delete_datapoint(dataset_name=dataset_name, datapoint_id=datapoint_ids[2])
    await async_client.delete_datapoint(dataset_name=dataset_name, datapoint_id=datapoint_ids[3])


@pytest.mark.asyncio
async def test_list_nonexistent_dataset(async_client: AsyncTensorZeroGateway):
    res = await async_client.list_datapoints_legacy(dataset_name="nonexistent_dataset")
    assert res == []


@pytest.mark.asyncio
async def test_get_nonexistent_datapoint(async_client: AsyncTensorZeroGateway):
    datapoint_id: UUID = uuid7()  # type: ignore
    with pytest.raises(TensorZeroError) as exc_info:
        await async_client.get_datapoint(dataset_name="nonexistent_dataset", datapoint_id=datapoint_id)
    assert "Datapoint not found for" in str(exc_info.value)
    assert "404" in str(exc_info.value)


def test_sync_render_datapoints(embedded_sync_client: TensorZeroGateway):
    """Test rendering datapoints using experimental_render_samples."""
    dataset_name = f"test_render_{uuid7()}"

    # Insert some datapoints
    datapoints = [
        ChatDatapointInsert(
            function_name="basic_test",
            input={
                "system": {"assistant_name": "TestBot"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Hello, world!"}],
                    }
                ],
            },
            output=[{"type": "text", "text": "Hello! How can I help you today?"}],
        ),
        JsonDatapointInsert(
            function_name="json_success",
            input={
                "system": {"assistant_name": "JsonBot"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "arguments": {"country": "France"}}],
                    }
                ],
            },
            output={"answer": "Paris"},
            output_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            },
        ),
    ]

    datapoint_ids = embedded_sync_client.create_datapoints_legacy(dataset_name=dataset_name, datapoints=datapoints)
    assert len(datapoint_ids) == 2

    # List the inserted datapoints
    listed_datapoints = embedded_sync_client.list_datapoints_legacy(dataset_name=dataset_name)
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
    for datapoint_id in datapoint_ids:
        embedded_sync_client.delete_datapoint(dataset_name=dataset_name, datapoint_id=datapoint_id)


@pytest.mark.asyncio
async def test_async_render_datapoints(
    embedded_async_client: AsyncTensorZeroGateway,
):
    """Test rendering datapoints using experimental_render_samples (async version)."""
    dataset_name = f"test_render_async_{uuid7()}"

    # Insert some datapoints
    datapoints = [
        ChatDatapointInsert(
            function_name="basic_test",
            input={
                "system": {"assistant_name": "AsyncBot"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "What's the weather like?"}],
                    }
                ],
            },
            output=[
                {
                    "type": "text",
                    "text": "I don't have access to current weather data.",
                }
            ],
        ),
        JsonDatapointInsert(
            function_name="json_success",
            input={
                "system": {"assistant_name": "DataBot"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "arguments": {"country": "Italy"}}],
                    }
                ],
            },
            output={"answer": "Rome"},
            output_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            },
        ),
    ]

    datapoint_ids = await embedded_async_client.create_datapoints_legacy(
        dataset_name=dataset_name, datapoints=datapoints
    )
    assert len(datapoint_ids) == 2

    # List the inserted datapoints
    listed_datapoints = await embedded_async_client.list_datapoints_legacy(dataset_name=dataset_name)
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
    for datapoint_id in datapoint_ids:
        await embedded_async_client.delete_datapoint(dataset_name=dataset_name, datapoint_id=datapoint_id)


def test_sync_render_filtered_datapoints(
    embedded_sync_client: TensorZeroGateway,
):
    """Test rendering only specific datapoints by filtering function name."""
    dataset_name = f"test_render_filter_{uuid7()}"

    # Insert datapoints for different functions
    datapoints = [
        ChatDatapointInsert(
            function_name="basic_test",
            input={
                "system": {"assistant_name": "FilterBot"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Test message 1"}],
                    }
                ],
            },
        ),
        ChatDatapointInsert(
            function_name="basic_test",
            input={
                "system": {"assistant_name": "FilterBot"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Test message 2"}],
                    }
                ],
            },
        ),
        JsonDatapointInsert(
            function_name="json_success",
            input={
                "system": {"assistant_name": "JsonFilter"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "arguments": {"country": "Spain"}}],
                    },
                ],
            },
        ),
    ]

    datapoint_ids = embedded_sync_client.create_datapoints_legacy(dataset_name=dataset_name, datapoints=datapoints)
    assert len(datapoint_ids) == 3

    # List only the basic_test datapoints
    chat_datapoints = embedded_sync_client.list_datapoints_legacy(dataset_name=dataset_name, function_name="basic_test")
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
    for datapoint_id in datapoint_ids:
        embedded_sync_client.delete_datapoint(dataset_name=dataset_name, datapoint_id=datapoint_id)


def test_sync_create_datapoints_legacy_deprecated(sync_client: TensorZeroGateway):
    dataset_name = f"test_{uuid7()}"
    datapoints = [
        ChatDatapointInsert(
            function_name="basic_test",
            input={
                "system": {"assistant_name": "foo"},
                "messages": [{"role": "user", "content": [{"type": "text", "text": "bar"}]}],
            },
            output=[{"type": "text", "text": "foobar"}],
        ),
    ]

    # Test that the deprecated function still works
    with pytest.warns(DeprecationWarning, match="Please use `create_datapoints` instead"):
        datapoint_ids = sync_client.create_datapoints_legacy(dataset_name=dataset_name, datapoints=datapoints)

    assert len(datapoint_ids) == 1
    assert isinstance(datapoint_ids[0], UUID)

    # Clean up
    sync_client.delete_datapoint(dataset_name=dataset_name, datapoint_id=datapoint_ids[0])


def test_sync_bulk_insert_datapoints_deprecated(sync_client: TensorZeroGateway):
    dataset_name = f"test_{uuid7()}"
    datapoints = [
        ChatDatapointInsert(
            function_name="basic_test",
            input={
                "system": {"assistant_name": "foo"},
                "messages": [{"role": "user", "content": [{"type": "text", "text": "bar"}]}],
            },
            output=[{"type": "text", "text": "foobar"}],
        ),
    ]

    # Test that the deprecated function still works
    with pytest.warns(DeprecationWarning, match="Please use `create_datapoints` instead"):
        datapoint_ids = sync_client.bulk_insert_datapoints(dataset_name=dataset_name, datapoints=datapoints)

    assert len(datapoint_ids) == 1
    assert isinstance(datapoint_ids[0], UUID)

    # Clean up
    sync_client.delete_datapoint(dataset_name=dataset_name, datapoint_id=datapoint_ids[0])


@pytest.mark.asyncio
async def test_async_bulk_insert_datapoints_deprecated(
    async_client: AsyncTensorZeroGateway,
):
    dataset_name = f"test_{uuid7()}"
    datapoints = [
        ChatDatapointInsert(
            function_name="basic_test",
            input={
                "system": {"assistant_name": "foo"},
                "messages": [{"role": "user", "content": [{"type": "text", "text": "bar"}]}],
            },
            output=[{"type": "text", "text": "foobar"}],
        ),
    ]

    # Test that the deprecated function still works
    with pytest.warns(DeprecationWarning, match="Please use `create_datapoints` instead"):
        datapoint_ids = await async_client.bulk_insert_datapoints(dataset_name=dataset_name, datapoints=datapoints)

    assert len(datapoint_ids) == 1
    assert isinstance(datapoint_ids[0], UUID)

    # Clean up
    await async_client.delete_datapoint(dataset_name=dataset_name, datapoint_id=datapoint_ids[0])


def test_sync_datapoints_with_name(sync_client: TensorZeroGateway):
    """Test that datapoints with name field are correctly stored and retrieved."""
    dataset_name = f"test_name_{uuid7()}"

    # Create datapoints with name field
    datapoints = [
        ChatDatapointInsert(
            function_name="basic_test",
            input={
                "system": {"assistant_name": "TestBot"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Hello"}],
                    }
                ],
            },
            output=[{"type": "text", "text": "Hi there!"}],
            name="greeting_example",
        ),
        JsonDatapointInsert(
            function_name="json_success",
            input={
                "system": {"assistant_name": "JsonBot"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "arguments": {"country": "Japan"}}],
                    }
                ],
            },
            output={"answer": "Tokyo"},
            name="tokyo_capital_query",
        ),
    ]

    # Insert datapoints
    datapoint_ids = sync_client.create_datapoints_legacy(dataset_name=dataset_name, datapoints=datapoints)
    assert len(datapoint_ids) == 2

    # Retrieve and verify chat datapoint with name
    chat_datapoint = sync_client.get_datapoint(dataset_name=dataset_name, datapoint_id=datapoint_ids[0])
    assert isinstance(chat_datapoint, ChatDatapoint)
    assert chat_datapoint.name == "greeting_example"
    assert chat_datapoint.function_name == "basic_test"

    # Retrieve and verify json datapoint with name
    json_datapoint = sync_client.get_datapoint(dataset_name=dataset_name, datapoint_id=datapoint_ids[1])
    assert isinstance(json_datapoint, JsonDatapoint)
    assert json_datapoint.name == "tokyo_capital_query"
    assert json_datapoint.function_name == "json_success"

    # Clean up
    for datapoint_id in datapoint_ids:
        sync_client.delete_datapoint(dataset_name=dataset_name, datapoint_id=datapoint_id)


@pytest.mark.asyncio
async def test_async_datapoints_with_name(async_client: AsyncTensorZeroGateway):
    """Test that datapoints with name field are correctly stored and retrieved (async version)."""
    dataset_name = f"test_name_async_{uuid7()}"

    # Create datapoints with name field
    datapoints = [
        ChatDatapointInsert(
            function_name="basic_test",
            input={
                "system": {"assistant_name": "AsyncBot"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Good morning"}],
                    }
                ],
            },
            output=[{"type": "text", "text": "Good morning to you!"}],
            name="morning_greeting",
        ),
        JsonDatapointInsert(
            function_name="json_success",
            input={
                "system": {"assistant_name": "AsyncJsonBot"},
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "arguments": {"country": "Germany"},
                            }
                        ],
                    }
                ],
            },
            output={"answer": "Berlin"},
            name="berlin_capital_query",
        ),
    ]

    # Insert datapoints
    datapoint_ids = await async_client.create_datapoints_legacy(dataset_name=dataset_name, datapoints=datapoints)
    assert len(datapoint_ids) == 2

    # Retrieve and verify chat datapoint with name
    chat_datapoint = await async_client.get_datapoint(dataset_name=dataset_name, datapoint_id=datapoint_ids[0])
    assert isinstance(chat_datapoint, ChatDatapoint)
    assert chat_datapoint.name == "morning_greeting"
    assert chat_datapoint.function_name == "basic_test"

    # Retrieve and verify json datapoint with name
    json_datapoint = await async_client.get_datapoint(dataset_name=dataset_name, datapoint_id=datapoint_ids[1])
    assert isinstance(json_datapoint, JsonDatapoint)
    assert json_datapoint.name == "berlin_capital_query"
    assert json_datapoint.function_name == "json_success"

    # List all datapoints and verify names are preserved
    all_datapoints = await async_client.list_datapoints_legacy(dataset_name=dataset_name)
    assert len(all_datapoints) == 2
    names = {dp.name for dp in all_datapoints}
    assert "morning_greeting" in names
    assert "berlin_capital_query" in names

    # Clean up
    for datapoint_id in datapoint_ids:
        await async_client.delete_datapoint(dataset_name=dataset_name, datapoint_id=datapoint_id)
