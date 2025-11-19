"""
Tests for v1 inference endpoints in the TensorZero client.

These tests cover the new v1 endpoints:
- get_inferences: Retrieve multiple inferences by IDs
- list_inferences: List inferences with filters, pagination, and sorting
"""

import json
from dataclasses import asdict

import pytest
from tensorzero import (
    AsyncTensorZeroGateway,
    InferenceFilterTag,
    InferenceResponse,
    ListInferencesRequest,
    TensorZeroGateway,
)
from tensorzero.generated_types import OrderByTimestamp


def _create_test_inference(client: TensorZeroGateway, tags: dict[str, str] | None = None) -> str:
    """Helper function to create a test inference and return its ID."""
    response = client.inference(
        function_name="basic_test",
        input={
            "system": {"assistant_name": "Assistant"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
        stream=False,
        tags=tags,
    )
    assert isinstance(response, InferenceResponse)
    return str(response.inference_id)


async def _create_test_inference_async(client: AsyncTensorZeroGateway, tags: dict[str, str] | None = None) -> str:
    """Helper function to create a test inference asynchronously and return its ID."""
    response = await client.inference(
        function_name="basic_test",
        input={
            "system": {"assistant_name": "Assistant"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
        stream=False,
        tags=tags,
    )
    assert isinstance(response, InferenceResponse)
    return str(response.inference_id)


def test_sync_get_inferences_by_ids(embedded_sync_client: TensorZeroGateway):
    """Test retrieving multiple inferences by IDs using get_inferences endpoint."""
    # Create some test inferences first
    _create_test_inference(embedded_sync_client)
    _create_test_inference(embedded_sync_client)
    _create_test_inference(embedded_sync_client)

    # First list some existing inferences
    request = ListInferencesRequest(
        function_name="basic_test",
        output_source="inference",
        limit=3,
        offset=0,
    )
    list_response = embedded_sync_client.list_inferences(request=request)

    assert list_response.inferences is not None
    assert len(list_response.inferences) > 0, "Expected at least some inferences to exist"

    # Get the IDs of some existing inferences
    inference_ids = [str(inf.inference_id) for inf in list_response.inferences]

    # Get inferences by IDs
    response = embedded_sync_client.get_inferences(ids=inference_ids, output_source="inference")

    assert response.inferences is not None
    assert len(response.inferences) == len(inference_ids)

    # Verify we got the correct inferences
    retrieved_ids = [str(inf.inference_id) for inf in response.inferences]
    assert set(retrieved_ids) == set(inference_ids)


@pytest.mark.asyncio
async def test_async_get_inferences_by_ids(embedded_async_client: AsyncTensorZeroGateway):
    """Test async version of get_inferences endpoint."""
    # Create some test inferences first
    await _create_test_inference_async(embedded_async_client)
    await _create_test_inference_async(embedded_async_client)

    # First list some existing inferences
    request = ListInferencesRequest(
        function_name="basic_test",
        output_source="inference",
        limit=2,
        offset=0,
    )
    list_response = await embedded_async_client.list_inferences(request=request)

    assert list_response.inferences is not None
    assert len(list_response.inferences) > 0, "Expected at least some inferences to exist"

    # Get the IDs of some existing inferences
    inference_ids = [str(inf.inference_id) for inf in list_response.inferences]

    # Get inferences by IDs
    response = await embedded_async_client.get_inferences(ids=inference_ids, output_source="inference")

    assert response.inferences is not None
    assert len(response.inferences) == len(inference_ids)


def test_sync_get_inferences_empty_ids(embedded_sync_client: TensorZeroGateway):
    """Test get_inferences with empty ID list."""
    response = embedded_sync_client.get_inferences(ids=[], output_source="inference")

    assert response.inferences is not None
    assert len(response.inferences) == 0


def test_sync_list_inferences_basic(embedded_sync_client: TensorZeroGateway):
    """Test basic listing of inferences."""
    # Create some test inferences first
    for _ in range(3):
        _create_test_inference(embedded_sync_client)

    # List inferences for the function
    request = ListInferencesRequest(
        function_name="basic_test",
        output_source="inference",
        limit=100,
        offset=0,
    )
    response = embedded_sync_client.list_inferences(request=request)

    assert response.inferences is not None
    assert len(response.inferences) > 0, "Expected at least some inferences"

    # Verify all returned inferences are from basic_test
    for inference in response.inferences:
        assert inference.function_name == "basic_test"


@pytest.mark.asyncio
async def test_async_list_inferences_basic(embedded_async_client: AsyncTensorZeroGateway):
    """Test async version of list_inferences."""
    # Create some test inferences first
    for _ in range(3):
        await _create_test_inference_async(embedded_async_client)

    # List inferences
    request = ListInferencesRequest(
        function_name="basic_test",
        output_source="inference",
        limit=50,
        offset=0,
    )
    response = await embedded_async_client.list_inferences(request=request)

    assert response.inferences is not None
    assert len(response.inferences) > 0


def test_sync_list_inferences_with_pagination(embedded_sync_client: TensorZeroGateway):
    """Test listing inferences with pagination."""
    # Create some test inferences first
    for _ in range(5):
        _create_test_inference(embedded_sync_client)

    # List all inferences with default pagination
    request = ListInferencesRequest(
        function_name="basic_test",
        output_source="inference",
        limit=100,
        offset=0,
    )
    response = embedded_sync_client.list_inferences(request=request)

    assert len(response.inferences) > 0, "Expected at least some inferences"
    total_count = len(response.inferences)

    # List with limit
    request = ListInferencesRequest(
        function_name="basic_test",
        output_source="inference",
        limit=2,
        offset=0,
    )
    response = embedded_sync_client.list_inferences(request=request)

    assert len(response.inferences) <= 2, "Limit should cap the results at 2"

    # List with offset (only if we have enough inferences)
    if total_count > 2:
        request = ListInferencesRequest(
            function_name="basic_test",
            output_source="inference",
            limit=100,
            offset=2,
        )
        response = embedded_sync_client.list_inferences(request=request)

        assert len(response.inferences) > 0, "Expected at least some inferences with offset"


def test_sync_list_inferences_by_variant(embedded_sync_client: TensorZeroGateway):
    """Test filtering inferences by variant name."""
    # Create some test inferences first
    for _ in range(3):
        _create_test_inference(embedded_sync_client)

    # First get existing inferences to find a variant name
    list_request = ListInferencesRequest(
        function_name="basic_test",
        output_source="inference",
        limit=1,
        offset=0,
    )
    list_response = embedded_sync_client.list_inferences(request=list_request)

    assert list_response.inferences is not None
    assert len(list_response.inferences) > 0, "Expected at least some inferences to exist"

    # Get the variant name from the first inference
    variant_name = list_response.inferences[0].variant_name

    # List inferences for specific variant
    request = ListInferencesRequest(
        function_name="basic_test",
        variant_name=variant_name,
        output_source="inference",
        limit=100,
        offset=0,
    )
    response = embedded_sync_client.list_inferences(request=request)

    assert response.inferences is not None
    assert len(response.inferences) > 0, "Expected at least some inferences with this variant"
    # Verify all returned inferences are from the specified variant
    for inference in response.inferences:
        assert inference.variant_name == variant_name


def test_sync_list_inferences_by_episode(embedded_sync_client: TensorZeroGateway):
    """Test filtering inferences by episode ID."""
    # Create some test inferences first
    for _ in range(3):
        _create_test_inference(embedded_sync_client)

    # First get an existing inference to extract an episode_id
    list_request = ListInferencesRequest(
        function_name="basic_test",
        output_source="inference",
        limit=100,
        offset=0,
    )
    list_response = embedded_sync_client.list_inferences(request=list_request)

    assert list_response.inferences is not None
    assert len(list_response.inferences) > 0, "Expected at least some inferences to exist"

    # Get an episode_id from one of the existing inferences
    episode_id = str(list_response.inferences[0].episode_id)

    # List inferences by episode ID
    request = ListInferencesRequest(
        episode_id=episode_id,
        output_source="inference",
        limit=100,
        offset=0,
    )
    response = embedded_sync_client.list_inferences(request=request)

    assert len(response.inferences) > 0, "Expected at least one inference with this episode_id"
    # Verify all inferences have the correct episode ID
    for inference in response.inferences:
        assert str(inference.episode_id) == episode_id


def test_sync_list_inferences_with_ordering(embedded_sync_client: TensorZeroGateway):
    """Test ordering inferences by timestamp."""
    # Create some test inferences first
    for _ in range(5):
        _create_test_inference(embedded_sync_client)

    # List inferences ordered by timestamp descending
    request = ListInferencesRequest(
        function_name="basic_test",
        output_source="inference",
        limit=10,
        offset=0,
        order_by=[OrderByTimestamp(direction="descending")],
    )
    response = embedded_sync_client.list_inferences(request=request)

    assert len(response.inferences) > 0

    # Verify timestamps are in descending order
    timestamps = [inf.timestamp for inf in response.inferences]
    for i in range(len(timestamps) - 1):
        assert timestamps[i] >= timestamps[i + 1], "Timestamps should be in descending order"


def test_sync_list_inferences_with_tag_filter(embedded_sync_client: TensorZeroGateway):
    """Test filtering inferences by tags."""
    # Create an inference with a specific tag
    _create_test_inference(embedded_sync_client, tags={"test_key": "test_value"})

    # First get existing inferences to find one with tags
    list_request = ListInferencesRequest(
        function_name="basic_test",
        output_source="inference",
        limit=100,
        offset=0,
    )
    list_response = embedded_sync_client.list_inferences(request=list_request)

    assert list_response.inferences is not None
    assert len(list_response.inferences) > 0, "Expected at least some inferences to exist"

    # Find an inference with tags
    inference_with_tags = None
    for inf in list_response.inferences:
        if inf.tags and len(inf.tags) > 0:
            inference_with_tags = inf
            break

    # If we found an inference with tags, test filtering by one of its tags
    if inference_with_tags is not None and inference_with_tags.tags:
        key, value = next(iter(inference_with_tags.tags.items()))

        # List inferences filtered by tag
        request = ListInferencesRequest(
            function_name="basic_test",
            output_source="inference",
            limit=100,
            offset=0,
            filter=InferenceFilterTag(key=key, value=value, comparison_operator="="),
        )
        response = embedded_sync_client.list_inferences(request=request)

        assert len(response.inferences) > 0, "Expected at least some inferences with this tag"
        # Verify all returned inferences have the tag
        for inference in response.inferences:
            assert inference.tags is not None
            assert key in inference.tags
            assert inference.tags[key] == value


@pytest.mark.asyncio
async def test_async_list_inferences_with_pagination(embedded_async_client: AsyncTensorZeroGateway):
    """Test async pagination for list_inferences."""
    # Create some test inferences first
    for _ in range(3):
        await _create_test_inference_async(embedded_async_client)

    # List with limit
    request = ListInferencesRequest(
        function_name="basic_test",
        output_source="inference",
        limit=2,
        offset=0,
    )
    response = await embedded_async_client.list_inferences(request=request)

    assert len(response.inferences) <= 2


def test_sync_list_inferences_no_function_filter(embedded_sync_client: TensorZeroGateway):
    """Test listing inferences without function name filter."""
    # Create some test inferences first
    for _ in range(3):
        _create_test_inference(embedded_sync_client)

    # List without function filter - should return inferences from any function
    request = ListInferencesRequest(
        output_source="inference",
        limit=10,
        offset=0,
    )
    response = embedded_sync_client.list_inferences(request=request)

    assert response.inferences is not None
    assert len(response.inferences) >= 1  # Should have at least some inferences


def test_sync_list_inferences_with_search_query(embedded_sync_client: TensorZeroGateway):
    """Test searching for inferences using search_query_experimental."""
    test_word = "hello"

    request = ListInferencesRequest(
        function_name="basic_test",
        output_source="inference",
        limit=2,
        offset=0,
        search_query_experimental=test_word,
    )
    response = embedded_sync_client.list_inferences(request=request)

    assert len(response.inferences) > 0
    for inference in response.inferences:
        assert test_word in json.dumps(asdict(inference)).lower()


@pytest.mark.asyncio
async def test_async_list_inferences_with_search_query(embedded_async_client: AsyncTensorZeroGateway):
    """Test async version of search_query_experimental."""
    test_word = "hello"

    request = ListInferencesRequest(
        function_name="basic_test",
        output_source="inference",
        limit=2,
        offset=0,
        search_query_experimental=test_word,
    )
    response = await embedded_async_client.list_inferences(request=request)

    assert len(response.inferences) > 0
    for inference in response.inferences:
        assert test_word in json.dumps(asdict(inference)).lower()
