"""
Tests for datapoint and dataset handling functionality in the TensorZero client.

These tests cover:
- Bulk insertion of datapoints
- Retrieving individual datapoints
- Listing datapoints with filtering
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
    ChatInferenceDatapointInput,
    JsonDatapoint,
    JsonDatapointInsert,
    JsonInferenceDatapointInput,
    TensorZeroError,
    TensorZeroGateway,
    Text,
)
from uuid_utils import uuid7


def test_sync_bulk_insert_delete_datapoints(sync_client: TensorZeroGateway):
    dataset_name = f"test_{uuid7()}"
    datapoints = [
        ChatDatapointInsert(
            function_name="basic_test",
            input={
                "system": {"assistant_name": "foo"},
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "bar"}]}
                ],
            },
            output=[{"type": "text", "text": "foobar"}],
            allowed_tools=None,
            additional_tools=None,
            tool_choice="auto",
            parallel_tool_calls=False,
            tags=None,
        ),
        # Ensure deprecated ChatInferenceDatapointInput is still supported
        ChatInferenceDatapointInput(
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
                    "arguments": {"location": "New York", "units": "fahrenheit"},
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
        # Ensure deprecated JsonInferenceDatapointInput is still supported
        JsonInferenceDatapointInput(
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
    datapoint_ids = sync_client.bulk_insert_datapoints(
        dataset_name=dataset_name, datapoints=datapoints
    )
    assert len(datapoint_ids) == 4
    assert isinstance(datapoint_ids[0], UUID)
    assert isinstance(datapoint_ids[1], UUID)
    assert isinstance(datapoint_ids[2], UUID)
    assert isinstance(datapoint_ids[3], UUID)

    # List datapoints filtering by function name
    listed_datapoints = sync_client.list_datapoints(
        dataset_name=dataset_name,
        function_name="basic_test",
    )
    assert len(listed_datapoints) == 2
    assert all(isinstance(dp, ChatDatapoint) for dp in listed_datapoints)
    assert all(dp.function_name == "basic_test" for dp in listed_datapoints)

    sync_client.delete_datapoint(dataset_name="test", datapoint_id=datapoint_ids[0])
    sync_client.delete_datapoint(dataset_name="test", datapoint_id=datapoint_ids[1])
    sync_client.delete_datapoint(dataset_name="test", datapoint_id=datapoint_ids[2])
    sync_client.delete_datapoint(dataset_name="test", datapoint_id=datapoint_ids[3])


@pytest.mark.asyncio
async def test_async_bulk_insert_delete_datapoints(
    async_client: AsyncTensorZeroGateway,
):
    datapoints = [
        ChatDatapointInsert(
            function_name="basic_test",
            input={
                "system": {"assistant_name": "foo"},
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "bar"}]}
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
                    "name": "get_temperature",
                    "arguments": {"location": "New York", "units": "fahrenheit"},
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
    datapoint_ids = await async_client.bulk_insert_datapoints(
        dataset_name=dataset_name, datapoints=datapoints
    )
    assert len(datapoint_ids) == 4
    assert isinstance(datapoint_ids[0], UUID)
    assert isinstance(datapoint_ids[1], UUID)
    assert isinstance(datapoint_ids[2], UUID)
    assert isinstance(datapoint_ids[3], UUID)

    # Get a chat datapoint
    datapoint = await async_client.get_datapoint(
        dataset_name=dataset_name, datapoint_id=datapoint_ids[0]
    )
    print(datapoint)
    assert isinstance(datapoint, ChatDatapoint)
    assert datapoint.function_name == "basic_test"
    assert datapoint.input.system == {"assistant_name": "foo"}
    assert len(datapoint.input.messages) == 1
    assert datapoint.input.messages[0].role == "user"
    assert len(datapoint.input.messages[0].content) == 1
    assert datapoint.input.messages[0].content[0].type == "text"
    assert isinstance(datapoint.input.messages[0].content[0], Text)
    assert datapoint.input.messages[0].content[0].text == "bar"
    assert datapoint.output is None

    # Get a json datapoint
    datapoint = await async_client.get_datapoint(
        dataset_name=dataset_name, datapoint_id=datapoint_ids[2]
    )
    assert isinstance(datapoint, JsonDatapoint)
    assert datapoint.function_name == "json_success"
    assert datapoint.input.system == {"assistant_name": "foo"}
    assert len(datapoint.input.messages) == 1
    assert datapoint.input.messages[0].role == "user"
    assert len(datapoint.input.messages[0].content) == 1
    assert datapoint.input.messages[0].content[0].type == "text"
    assert isinstance(datapoint.input.messages[0].content[0], Text)
    assert datapoint.input.messages[0].content[0].arguments == {"country": "US"}
    assert datapoint.output is None

    # List datapoints
    listed_datapoints = await async_client.list_datapoints(
        dataset_name=dataset_name,
    )
    assert len(listed_datapoints) == 4
    # Assert that there are 2 chat and 2 json datapoints
    chat_datapoints = [dp for dp in listed_datapoints if isinstance(dp, ChatDatapoint)]
    json_datapoints = [dp for dp in listed_datapoints if isinstance(dp, JsonDatapoint)]
    assert len(chat_datapoints) == 2
    assert len(json_datapoints) == 2

    # List datapoints filtering by function name
    listed_datapoints = await async_client.list_datapoints(
        dataset_name=dataset_name,
        function_name="basic_test",
    )
    assert len(listed_datapoints) == 2
    assert all(isinstance(dp, ChatDatapoint) for dp in listed_datapoints)
    assert all(dp.function_name == "basic_test" for dp in listed_datapoints)

    await async_client.delete_datapoint(
        dataset_name=dataset_name, datapoint_id=datapoint_ids[0]
    )
    await async_client.delete_datapoint(
        dataset_name=dataset_name, datapoint_id=datapoint_ids[1]
    )
    await async_client.delete_datapoint(
        dataset_name=dataset_name, datapoint_id=datapoint_ids[2]
    )
    await async_client.delete_datapoint(
        dataset_name=dataset_name, datapoint_id=datapoint_ids[3]
    )


@pytest.mark.asyncio
async def test_list_nonexistent_dataset(async_client: AsyncTensorZeroGateway):
    res = await async_client.list_datapoints(dataset_name="nonexistent_dataset")
    assert res == []


@pytest.mark.asyncio
async def test_get_nonexistent_datapoint(async_client: AsyncTensorZeroGateway):
    datapoint_id: UUID = uuid7()  # type: ignore
    with pytest.raises(TensorZeroError) as exc_info:
        await async_client.get_datapoint(
            dataset_name="nonexistent_dataset", datapoint_id=datapoint_id
        )
    assert "Datapoint not found for" in str(exc_info.value)
    assert "404" in str(exc_info.value)
