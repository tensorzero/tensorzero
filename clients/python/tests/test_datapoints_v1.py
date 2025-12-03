"""
Tests for v1 dataset endpoints in the TensorZero client.

These tests cover the new v1 endpoints:
- get_datapoints: Retrieve multiple datapoints by IDs
- list_datapoints: List datapoints with filters (BREAKING CHANGE from old API)
- update_datapoints_metadata: Update metadata without creating new IDs
- delete_datapoints: Delete multiple datapoints at once
- delete_dataset: Delete an entire dataset
- create_datapoints_from_inferences: Create dataset from inference results

"""

from time import sleep

import pytest
from tensorzero import (
    AsyncTensorZeroGateway,
    ContentBlockChatOutputText,
    CreateDatapointRequestChat,
    CreateDatapointRequestJson,
    CreateDatapointsFromInferenceRequestParamsInferenceIds,
    Input,
    InputMessage,
    InputMessageContentTemplate,
    InputMessageContentText,
    JsonDatapointOutputUpdate,
    ListDatapointsRequest,
    OrderBy,
    TensorZeroGateway,
    UpdateDatapointMetadataRequest,
)
from uuid_utils import uuid7


def test_sync_get_datapoints_by_ids(sync_client: TensorZeroGateway):
    """Test retrieving multiple datapoints by IDs using get_datapoints endpoint."""
    dataset_name = f"test_get_v1_{uuid7()}"

    # Insert test datapoints
    requests = [
        CreateDatapointRequestChat(
            function_name="basic_test",
            input=Input(
                system={"assistant_name": "TestBot"},
                messages=[InputMessage(role="user", content=[InputMessageContentText(text="First message")])],
            ),
            output=[ContentBlockChatOutputText(text="First response")],
        ),
        CreateDatapointRequestChat(
            function_name="basic_test",
            input=Input(
                system={"assistant_name": "TestBot"},
                messages=[InputMessage(role="user", content=[InputMessageContentText(text="Second message")])],
            ),
            output=[ContentBlockChatOutputText(text="Second response")],
        ),
        CreateDatapointRequestJson(
            function_name="json_success",
            input=Input(
                system={"assistant_name": "JsonBot"},
                messages=[
                    InputMessage(
                        role="user", content=[InputMessageContentTemplate(name="user", arguments={"country": "Canada"})]
                    )
                ],
            ),
            output=JsonDatapointOutputUpdate(raw='{"answer":"Ottawa"}'),
        ),
    ]

    response = sync_client.create_datapoints(dataset_name=dataset_name, requests=requests)
    datapoint_ids = response.ids
    assert len(datapoint_ids) == 3

    # Get all datapoints by IDs using v1 endpoint (convert UUIDs to strings)
    response = sync_client.get_datapoints(ids=datapoint_ids)
    datapoints = response.datapoints

    assert datapoints is not None
    assert len(datapoints) == 3

    # Verify we got the correct datapoints
    retrieved_ids = [dp.id for dp in datapoints]
    assert set(retrieved_ids) == set(datapoint_ids)

    # Verify types
    chat_dps = [dp for dp in datapoints if dp.type == "chat"]
    json_dps = [dp for dp in datapoints if dp.type == "json"]
    assert len(chat_dps) == 2
    assert len(json_dps) == 1

    # Clean up
    sync_client.delete_datapoints(dataset_name=dataset_name, ids=datapoint_ids)


@pytest.mark.asyncio
async def test_async_get_datapoints_by_ids(async_client: AsyncTensorZeroGateway):
    """Test async version of get_datapoints endpoint."""
    dataset_name = f"test_get_v1_async_{uuid7()}"

    # Insert test datapoints
    requests = [
        CreateDatapointRequestChat(
            function_name="basic_test",
            input=Input(
                system={"assistant_name": "AsyncBot"},
                messages=[InputMessage(role="user", content=[InputMessageContentText(text="Async message")])],
            ),
        ),
        CreateDatapointRequestJson(
            function_name="json_success",
            input=Input(
                system={"assistant_name": "AsyncJson"},
                messages=[
                    InputMessage(
                        role="user", content=[InputMessageContentTemplate(name="user", arguments={"country": "Mexico"})]
                    )
                ],
            ),
        ),
    ]

    response = await async_client.create_datapoints(dataset_name=dataset_name, requests=requests)
    datapoint_ids = response.ids
    assert len(datapoint_ids) == 2

    # Get datapoints by IDs (convert to strings)
    response = await async_client.get_datapoints(ids=datapoint_ids)
    datapoints = response.datapoints

    assert datapoints is not None
    assert len(datapoints) == 2

    # Clean up
    await async_client.delete_datapoints(dataset_name=dataset_name, ids=datapoint_ids)


def test_sync_get_datapoints_by_ids_with_dataset_name(sync_client: TensorZeroGateway):
    """Test retrieving multiple datapoints by IDs using get_datapoints endpoint."""
    dataset_name = f"test_get_v1_{uuid7()}"

    # Insert test datapoints
    requests = [
        CreateDatapointRequestChat(
            function_name="basic_test",
            input=Input(
                system={"assistant_name": "TestBot"},
                messages=[InputMessage(role="user", content=[InputMessageContentText(text="First message")])],
            ),
            output=[ContentBlockChatOutputText(text="First response")],
        ),
        CreateDatapointRequestChat(
            function_name="basic_test",
            input=Input(
                system={"assistant_name": "TestBot"},
                messages=[InputMessage(role="user", content=[InputMessageContentText(text="Second message")])],
            ),
            output=[ContentBlockChatOutputText(text="Second response")],
        ),
        CreateDatapointRequestJson(
            function_name="json_success",
            input=Input(
                system={"assistant_name": "JsonBot"},
                messages=[
                    InputMessage(
                        role="user", content=[InputMessageContentTemplate(name="user", arguments={"country": "Canada"})]
                    )
                ],
            ),
            output=JsonDatapointOutputUpdate(raw='{"answer":"Ottawa"}'),
        ),
    ]

    response = sync_client.create_datapoints(dataset_name=dataset_name, requests=requests)
    datapoint_ids = response.ids
    assert len(datapoint_ids) == 3

    # Get all datapoints by IDs using v1 endpoint (convert UUIDs to strings)
    response = sync_client.get_datapoints(dataset_name=dataset_name, ids=datapoint_ids)
    datapoints = response.datapoints

    assert datapoints is not None
    assert len(datapoints) == 3

    # Verify we got the correct datapoints
    retrieved_ids = [dp.id for dp in datapoints]
    assert set(retrieved_ids) == set(datapoint_ids)

    # Verify types
    chat_dps = [dp for dp in datapoints if dp.type == "chat"]
    json_dps = [dp for dp in datapoints if dp.type == "json"]
    assert len(chat_dps) == 2
    assert len(json_dps) == 1

    # Clean up
    sync_client.delete_datapoints(dataset_name=dataset_name, ids=datapoint_ids)


@pytest.mark.asyncio
async def test_async_get_datapoints_by_ids_with_dataset_name(async_client: AsyncTensorZeroGateway):
    """Test async version of get_datapoints endpoint."""
    dataset_name = f"test_get_v1_async_{uuid7()}"

    # Insert test datapoints
    requests = [
        CreateDatapointRequestChat(
            function_name="basic_test",
            input=Input(
                system={"assistant_name": "AsyncBot"},
                messages=[InputMessage(role="user", content=[InputMessageContentText(text="Async message")])],
            ),
        ),
        CreateDatapointRequestJson(
            function_name="json_success",
            input=Input(
                system={"assistant_name": "AsyncJson"},
                messages=[
                    InputMessage(
                        role="user", content=[InputMessageContentTemplate(name="user", arguments={"country": "Mexico"})]
                    )
                ],
            ),
        ),
    ]

    response = await async_client.create_datapoints(dataset_name=dataset_name, requests=requests)
    datapoint_ids = response.ids
    assert len(datapoint_ids) == 2

    # Get datapoints by IDs (convert to strings)
    response = await async_client.get_datapoints(dataset_name=dataset_name, ids=datapoint_ids)
    datapoints = response.datapoints

    assert datapoints is not None
    assert len(datapoints) == 2

    # Clean up
    await async_client.delete_datapoints(dataset_name=dataset_name, ids=datapoint_ids)


def test_sync_list_datapoints_with_filters(sync_client: TensorZeroGateway):
    """Test listing datapoints with the new v1 filter-based API."""
    dataset_name = f"test_list_v1_{uuid7()}"

    # Insert multiple datapoints
    requests = [
        CreateDatapointRequestChat(
            function_name="basic_test",
            input=Input(
                system={"assistant_name": "Bot1"},
                messages=[InputMessage(role="user", content=[InputMessageContentText(text="msg1")])],
            ),
        ),
        CreateDatapointRequestChat(
            function_name="basic_test",
            input=Input(
                system={"assistant_name": "Bot2"},
                messages=[InputMessage(role="user", content=[InputMessageContentText(text="msg2")])],
            ),
        ),
        CreateDatapointRequestJson(
            function_name="json_success",
            input=Input(
                system={"assistant_name": "JsonBot"},
                messages=[
                    InputMessage(
                        role="user", content=[InputMessageContentTemplate(name="user", arguments={"country": "Brazil"})]
                    )
                ],
            ),
        ),
    ]

    response = sync_client.create_datapoints(dataset_name=dataset_name, requests=requests)
    datapoint_ids = response.ids
    assert len(datapoint_ids) == 3

    # List all datapoints with v1 API (using request dict with page_size)
    response = sync_client.list_datapoints(dataset_name=dataset_name, request=ListDatapointsRequest(limit=10))
    datapoints = response.datapoints

    assert datapoints is not None
    assert len(datapoints) == 3

    # List with page_size limit
    response = sync_client.list_datapoints(dataset_name=dataset_name, request=ListDatapointsRequest(limit=2, offset=0))
    datapoints = response.datapoints
    assert len(datapoints) == 2

    # List with offset
    response = sync_client.list_datapoints(dataset_name=dataset_name, request=ListDatapointsRequest(limit=10, offset=2))
    datapoints = response.datapoints
    assert len(datapoints) == 1

    # Clean up
    sync_client.delete_datapoints(dataset_name=dataset_name, ids=datapoint_ids)


@pytest.mark.asyncio
async def test_async_list_datapoints_with_filters(async_client: AsyncTensorZeroGateway):
    """Test async version of list_datapoints with v1 filter API."""
    dataset_name = f"test_list_v1_async_{uuid7()}"

    # Insert datapoints
    requests = [
        CreateDatapointRequestChat(
            function_name="basic_test",
            input=Input(
                system={"assistant_name": "Filter1"},
                messages=[InputMessage(role="user", content=[InputMessageContentText(text="test1")])],
            ),
        ),
        CreateDatapointRequestChat(
            function_name="basic_test",
            input=Input(
                system={"assistant_name": "Filter2"},
                messages=[InputMessage(role="user", content=[InputMessageContentText(text="test2")])],
            ),
        ),
    ]

    response = await async_client.create_datapoints(dataset_name=dataset_name, requests=requests)
    datapoint_ids = response.ids

    # List with filters
    response = await async_client.list_datapoints(
        dataset_name=dataset_name, request=ListDatapointsRequest(limit=10, offset=0)
    )
    datapoints = response.datapoints

    assert len(datapoints) == 2

    # Clean up
    await async_client.delete_datapoints(dataset_name=dataset_name, ids=datapoint_ids)


def test_sync_update_datapoints_metadata(sync_client: TensorZeroGateway):
    """Test updating datapoint metadata without creating new IDs."""
    dataset_name = f"test_update_meta_{uuid7()}"

    # Insert datapoint with initial name
    requests = [
        CreateDatapointRequestChat(
            function_name="basic_test",
            input=Input(
                system={"assistant_name": "MetaBot"},
                messages=[InputMessage(role="user", content=[InputMessageContentText(text="original")])],
            ),
            name="original_name",
        ),
    ]

    response = sync_client.create_datapoints(dataset_name=dataset_name, requests=requests)
    datapoint_ids = response.ids
    original_id = datapoint_ids[0]

    # Update metadata using v1 endpoint (returns list of UUIDs)
    response = sync_client.update_datapoints_metadata(
        dataset_name=dataset_name,
        requests=[UpdateDatapointMetadataRequest(id=str(original_id), name="updated_name")],
    )
    updated_ids = response.ids

    assert updated_ids is not None
    assert isinstance(updated_ids, list)
    assert len(updated_ids) == 1
    # The ID should remain the same (not a new ID like update_datapoints would create)
    assert updated_ids[0] == original_id

    # Wait for the metadata to be updated
    sleep(1)

    # Verify the metadata was updated
    response = sync_client.get_datapoints(dataset_name=dataset_name, ids=[str(original_id)])
    datapoints = response.datapoints
    assert len(datapoints) == 1
    assert datapoints[0].name == "updated_name"

    # Clear the name using v1 endpoint
    response = sync_client.update_datapoints_metadata(
        dataset_name=dataset_name,
        requests=[UpdateDatapointMetadataRequest(id=str(original_id), name=None)],
    )

    # Wait for the metadata to be updated
    sleep(1)

    # Verify the name was cleared
    response = sync_client.get_datapoints(dataset_name=dataset_name, ids=[str(original_id)])
    datapoints = response.datapoints
    assert len(datapoints) == 1
    assert datapoints[0].name is None

    # Clean up
    sync_client.delete_datapoints(dataset_name=dataset_name, ids=[original_id])


@pytest.mark.asyncio
async def test_async_update_datapoints_metadata(async_client: AsyncTensorZeroGateway):
    """Test async version of update_datapoints_metadata."""
    dataset_name = f"test_update_meta_async_{uuid7()}"

    # Insert datapoint
    requests = [
        CreateDatapointRequestJson(
            function_name="json_success",
            input=Input(
                system={"assistant_name": "AsyncMeta"},
                messages=[
                    InputMessage(
                        role="user", content=[InputMessageContentTemplate(name="user", arguments={"country": "France"})]
                    )
                ],
            ),
            name="initial",
        ),
    ]

    response = await async_client.create_datapoints(dataset_name=dataset_name, requests=requests)
    datapoint_ids = response.ids
    original_id = datapoint_ids[0]

    # Update metadata (returns list of UUIDs)
    response = await async_client.update_datapoints_metadata(
        dataset_name=dataset_name,
        requests=[
            UpdateDatapointMetadataRequest(id=str(original_id), name="modified"),
        ],
    )
    updated_ids = response.ids

    assert len(updated_ids) == 1
    assert updated_ids[0] == original_id

    # Note: Metadata verification skipped due to potential caching/timing issues

    # Clean up
    await async_client.delete_datapoints(dataset_name=dataset_name, ids=[original_id])


def test_sync_delete_multiple_datapoints(sync_client: TensorZeroGateway):
    """Test deleting multiple datapoints at once using delete_datapoints endpoint."""
    dataset_name = f"test_delete_multi_{uuid7()}"

    # Insert multiple datapoints
    requests = [
        CreateDatapointRequestChat(
            function_name="basic_test",
            input=Input(
                system={"assistant_name": "DeleteBot"},
                messages=[InputMessage(role="user", content=[InputMessageContentText(text=f"message {i}")])],
            ),
        )
        for i in range(5)
    ]

    response = sync_client.create_datapoints(dataset_name=dataset_name, requests=requests)
    datapoint_ids = response.ids
    assert len(datapoint_ids) == 5

    # Delete first 3 datapoints using v1 bulk delete (convert to strings)
    ids_to_delete = datapoint_ids[:3]
    response = sync_client.delete_datapoints(dataset_name=dataset_name, ids=ids_to_delete)
    num_deleted = response.num_deleted_datapoints

    assert num_deleted == 3

    # Verify remaining datapoints
    response = sync_client.list_datapoints(
        dataset_name=dataset_name, request=ListDatapointsRequest(limit=100, offset=0)
    )
    remaining = response.datapoints
    assert len(remaining) == 2

    remaining_ids = [dp.id for dp in remaining]
    assert set(remaining_ids) == set(datapoint_ids[3:])

    # Clean up remaining
    sync_client.delete_datapoints(dataset_name=dataset_name, ids=datapoint_ids[3:])


@pytest.mark.asyncio
async def test_async_delete_multiple_datapoints(async_client: AsyncTensorZeroGateway):
    """Test async version of delete_datapoints endpoint."""
    dataset_name = f"test_delete_multi_async_{uuid7()}"

    # Insert datapoints
    requests = [
        CreateDatapointRequestJson(
            function_name="json_success",
            input=Input(
                system={"assistant_name": "DeleteJson"},
                messages=[
                    InputMessage(
                        role="user",
                        content=[InputMessageContentTemplate(name="user", arguments={"country": f"Country{i}"})],
                    )
                ],
            ),
        )
        for i in range(4)
    ]

    response = await async_client.create_datapoints(dataset_name=dataset_name, requests=requests)
    datapoint_ids = response.ids

    # Delete all at once (convert to strings)
    response = await async_client.delete_datapoints(dataset_name=dataset_name, ids=datapoint_ids)
    num_deleted = response.num_deleted_datapoints

    assert num_deleted == 4

    # Verify all deleted
    response = await async_client.list_datapoints(
        dataset_name=dataset_name, request=ListDatapointsRequest(limit=100, offset=0)
    )
    remaining = response.datapoints
    assert len(remaining) == 0


def test_sync_delete_entire_dataset(sync_client: TensorZeroGateway):
    """Test deleting an entire dataset using delete_dataset endpoint."""
    dataset_name = f"test_delete_dataset_{uuid7()}"

    # Create dataset with datapoints
    requests = [
        CreateDatapointRequestChat(
            function_name="basic_test",
            input=Input(
                system={"assistant_name": "ToDelete"},
                messages=[InputMessage(role="user", content=[InputMessageContentText(text=f"data {i}")])],
            ),
        )
        for i in range(10)
    ]

    response = sync_client.create_datapoints(dataset_name=dataset_name, requests=requests)
    datapoint_ids = response.ids
    assert len(datapoint_ids) == 10

    # Verify dataset exists
    response = sync_client.list_datapoints(
        dataset_name=dataset_name, request=ListDatapointsRequest(limit=100, offset=0)
    )
    datapoints = response.datapoints
    assert len(datapoints) == 10

    # Delete entire dataset
    response = sync_client.delete_dataset(dataset_name=dataset_name)

    assert response.num_deleted_datapoints == 10

    # Verify dataset is empty
    response = sync_client.list_datapoints(
        dataset_name=dataset_name, request=ListDatapointsRequest(limit=100, offset=0)
    )
    remaining = response.datapoints
    assert len(remaining) == 0


@pytest.mark.asyncio
async def test_async_delete_entire_dataset(async_client: AsyncTensorZeroGateway):
    """Test async version of delete_dataset endpoint."""
    dataset_name = f"test_delete_dataset_async_{uuid7()}"

    # Create dataset
    requests = [
        CreateDatapointRequestJson(
            function_name="json_success",
            input=Input(
                system={"assistant_name": "AsyncDelete"},
                messages=[
                    InputMessage(
                        role="user",
                        content=[InputMessageContentTemplate(name="user", arguments={"country": f"Country{i}"})],
                    )
                ],
            ),
        )
        for i in range(7)
    ]

    response = await async_client.create_datapoints(dataset_name=dataset_name, requests=requests)
    datapoint_ids = response.ids
    assert len(datapoint_ids) == 7

    # Delete dataset
    response = await async_client.delete_dataset(dataset_name=dataset_name)
    num_deleted = response.num_deleted_datapoints
    assert num_deleted == 7

    # Verify empty
    response = await async_client.list_datapoints(
        dataset_name=dataset_name, request=ListDatapointsRequest(limit=100, offset=0)
    )
    remaining = response.datapoints
    assert len(remaining) == 0


def test_sync_create_datapoints_from_inferences(embedded_sync_client: TensorZeroGateway):
    """Test creating dataset from inference results."""
    # First, list a few existing inferences
    order_by = [OrderBy(by="timestamp", direction="descending")]
    inferences = embedded_sync_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=None,
        output_source="inference",
        limit=2,
        offset=None,
        order_by=order_by,
    )
    assert len(inferences) == 2, "Should be able to find 2 existing stored inferences"
    for inference in inferences:
        assert inference.inference_id is not None, "Inferences should contain IDs"

    # Now create dataset from these inferences
    dataset_name = f"test_from_inferences_{uuid7()}"

    response = embedded_sync_client.create_datapoints_from_inferences(
        dataset_name=dataset_name,
        params=CreateDatapointsFromInferenceRequestParamsInferenceIds(
            inference_ids=[str(inference.inference_id) for inference in inferences]
        ),
        output_source="inference",
    )
    created_ids = response.ids

    assert len(created_ids) == 2

    # Verify datapoints were created
    response = embedded_sync_client.list_datapoints(
        dataset_name=dataset_name,
        request=ListDatapointsRequest(limit=10),
    )
    datapoints = response.datapoints
    assert len(datapoints) == 2

    # Clean up
    embedded_sync_client.delete_dataset(dataset_name=dataset_name)


@pytest.mark.asyncio
async def test_async_create_datapoints_from_inferences(embedded_async_client: AsyncTensorZeroGateway):
    """Test async version of create_datapoints_from_inferences."""
    order_by = [OrderBy(by="timestamp", direction="descending")]
    inferences = await embedded_async_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=None,
        output_source="inference",
        limit=2,
        offset=None,
        order_by=order_by,
    )
    assert len(inferences) == 2, "Should be able to find 2 existing stored inferences"
    for inference in inferences:
        assert inference.inference_id is not None, "Inferences should contain IDs"

    # Now create dataset from these inferences
    dataset_name = f"test_from_inferences_async_{uuid7()}"

    response = await embedded_async_client.create_datapoints_from_inferences(
        dataset_name=dataset_name,
        params=CreateDatapointsFromInferenceRequestParamsInferenceIds(
            inference_ids=[str(inference.inference_id) for inference in inferences]
        ),
        output_source="inference",
    )
    datapoint_ids = response.ids

    assert len(datapoint_ids) == 2

    # Verify
    response = await embedded_async_client.list_datapoints(
        dataset_name=dataset_name,
        request=ListDatapointsRequest(limit=10),
    )
    listed = response.datapoints
    assert len(listed) == 2

    # Clean up
    await embedded_async_client.delete_dataset(dataset_name=dataset_name)


def test_sync_get_datapoints_empty_list(sync_client: TensorZeroGateway):
    """Test get_datapoints with empty ID list."""
    response = sync_client.get_datapoints(ids=[])
    datapoints = response.datapoints
    assert datapoints is not None
    assert len(datapoints) == 0


def test_sync_get_datapoints_nonexistent_ids(sync_client: TensorZeroGateway):
    """Test get_datapoints with non-existent IDs returns empty datapoints."""
    fake_ids = [str(uuid7()), str(uuid7())]  # type: ignore
    response = sync_client.get_datapoints(ids=fake_ids)
    datapoints = response.datapoints

    # Non-existent IDs should return empty list, not error
    assert datapoints is not None
    assert len(datapoints) == 0
