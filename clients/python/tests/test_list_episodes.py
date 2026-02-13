import pytest
from tensorzero import (
    AsyncTensorZeroGateway,
    EpisodeByIdRow,
    ListEpisodesRequest,
    ListEpisodesResponse,
    TensorZeroGateway,
)
from tensorzero.generated_types import (
    InferenceFilterAnd,
    InferenceFilterBooleanMetric,
    InferenceFilterFloatMetric,
)


def test_list_episodes(embedded_sync_client: TensorZeroGateway):
    response = embedded_sync_client.list_episodes(request=ListEpisodesRequest(limit=5))
    assert isinstance(response, ListEpisodesResponse)
    assert len(response.episodes) > 0
    assert len(response.episodes) <= 5

    for episode in response.episodes:
        assert isinstance(episode, EpisodeByIdRow)
        assert isinstance(episode.episode_id, str)
        assert isinstance(episode.count, int)
        assert episode.count > 0
        assert isinstance(episode.start_time, str)
        assert isinstance(episode.end_time, str)
        assert isinstance(episode.last_inference_id, str)


def test_list_episodes_pagination(embedded_sync_client: TensorZeroGateway):
    # Get the first page
    first_page = embedded_sync_client.list_episodes(request=ListEpisodesRequest(limit=2))
    assert isinstance(first_page, ListEpisodesResponse)
    assert len(first_page.episodes) == 2

    # Get the next page using `before` cursor
    last_episode = first_page.episodes[-1]
    second_page = embedded_sync_client.list_episodes(
        request=ListEpisodesRequest(limit=2, before=last_episode.episode_id)
    )
    assert isinstance(second_page, ListEpisodesResponse)

    # Pages should not overlap
    first_page_ids = {e.episode_id for e in first_page.episodes}
    second_page_ids = {e.episode_id for e in second_page.episodes}
    assert first_page_ids.isdisjoint(second_page_ids), "Paginated pages should not overlap"


def test_list_episodes_with_function_name(embedded_sync_client: TensorZeroGateway):
    response = embedded_sync_client.list_episodes(request=ListEpisodesRequest(limit=10, function_name="basic_test"))
    assert isinstance(response, ListEpisodesResponse)
    assert len(response.episodes) > 0, "Expected at least some episodes for function `basic_test`"

    # Non-existent function should return no episodes
    response = embedded_sync_client.list_episodes(
        request=ListEpisodesRequest(limit=10, function_name="nonexistent_function")
    )
    assert isinstance(response, ListEpisodesResponse)
    assert len(response.episodes) == 0, "Expected no episodes for non-existent function"


def test_list_episodes_with_boolean_filter(embedded_sync_client: TensorZeroGateway):
    # Get unfiltered count for comparison
    unfiltered = embedded_sync_client.list_episodes(request=ListEpisodesRequest(limit=100))

    # Filter by boolean metric
    filtered = embedded_sync_client.list_episodes(
        request=ListEpisodesRequest(
            limit=100,
            filters=InferenceFilterBooleanMetric(
                metric_name="task_success",
                value=True,
            ),
        )
    )
    assert isinstance(filtered, ListEpisodesResponse)
    assert len(filtered.episodes) > 0, "Expected at least one episode matching the boolean metric filter"
    assert len(filtered.episodes) <= len(unfiltered.episodes), (
        "Filtered episodes should be a subset of unfiltered episodes"
    )


def test_list_episodes_with_float_filter(embedded_sync_client: TensorZeroGateway):
    # Get unfiltered count for comparison
    unfiltered = embedded_sync_client.list_episodes(request=ListEpisodesRequest(limit=100))

    # Filter by float metric (>= 4.0 should match)
    filtered = embedded_sync_client.list_episodes(
        request=ListEpisodesRequest(
            limit=100,
            filters=InferenceFilterFloatMetric(
                metric_name="user_rating",
                value=4.0,
                comparison_operator=">=",
            ),
        )
    )
    assert isinstance(filtered, ListEpisodesResponse)
    assert len(filtered.episodes) > 0, "Expected at least one episode matching the float metric filter"
    assert len(filtered.episodes) <= len(unfiltered.episodes), (
        "Filtered episodes should be a subset of unfiltered episodes"
    )


def test_list_episodes_combined_filters(embedded_sync_client: TensorZeroGateway):
    # Get unfiltered count for comparison
    unfiltered = embedded_sync_client.list_episodes(request=ListEpisodesRequest(limit=100))

    # Combine boolean and float filters with AND
    filtered = embedded_sync_client.list_episodes(
        request=ListEpisodesRequest(
            limit=100,
            function_name="basic_test",
            filters=InferenceFilterAnd(
                children=[
                    InferenceFilterBooleanMetric(
                        metric_name="task_success",
                        value=True,
                    ),
                    InferenceFilterFloatMetric(
                        metric_name="user_rating",
                        value=4.0,
                        comparison_operator=">=",
                    ),
                ],
            ),
        )
    )
    assert isinstance(filtered, ListEpisodesResponse)
    assert len(filtered.episodes) <= len(unfiltered.episodes), (
        "Filtered episodes should be a subset of unfiltered episodes"
    )


# Async versions


@pytest.mark.asyncio
async def test_list_episodes_async(
    embedded_async_client: AsyncTensorZeroGateway,
):
    response = await embedded_async_client.list_episodes(request=ListEpisodesRequest(limit=5))
    assert isinstance(response, ListEpisodesResponse)
    assert len(response.episodes) > 0
    assert len(response.episodes) <= 5

    for episode in response.episodes:
        assert isinstance(episode, EpisodeByIdRow)
        assert isinstance(episode.episode_id, str)
        assert isinstance(episode.count, int)
        assert episode.count > 0
        assert isinstance(episode.start_time, str)
        assert isinstance(episode.end_time, str)
        assert isinstance(episode.last_inference_id, str)


@pytest.mark.asyncio
async def test_list_episodes_pagination_async(
    embedded_async_client: AsyncTensorZeroGateway,
):
    # Get the first page
    first_page = await embedded_async_client.list_episodes(request=ListEpisodesRequest(limit=2))
    assert isinstance(first_page, ListEpisodesResponse)
    assert len(first_page.episodes) == 2

    # Get the next page using `before` cursor
    last_episode = first_page.episodes[-1]
    second_page = await embedded_async_client.list_episodes(
        request=ListEpisodesRequest(limit=2, before=last_episode.episode_id)
    )
    assert isinstance(second_page, ListEpisodesResponse)

    # Pages should not overlap
    first_page_ids = {e.episode_id for e in first_page.episodes}
    second_page_ids = {e.episode_id for e in second_page.episodes}
    assert first_page_ids.isdisjoint(second_page_ids), "Paginated pages should not overlap"


@pytest.mark.asyncio
async def test_list_episodes_with_function_name_async(
    embedded_async_client: AsyncTensorZeroGateway,
):
    response = await embedded_async_client.list_episodes(
        request=ListEpisodesRequest(limit=10, function_name="basic_test")
    )
    assert isinstance(response, ListEpisodesResponse)
    assert len(response.episodes) > 0, "Expected at least some episodes for function `basic_test`"


@pytest.mark.asyncio
async def test_list_episodes_with_boolean_filter_async(
    embedded_async_client: AsyncTensorZeroGateway,
):
    # Get unfiltered count for comparison
    unfiltered = await embedded_async_client.list_episodes(request=ListEpisodesRequest(limit=100))

    # Filter by boolean metric
    filtered = await embedded_async_client.list_episodes(
        request=ListEpisodesRequest(
            limit=100,
            filters=InferenceFilterBooleanMetric(
                metric_name="task_success",
                value=True,
            ),
        )
    )
    assert isinstance(filtered, ListEpisodesResponse)
    assert len(filtered.episodes) > 0, "Expected at least one episode matching the boolean metric filter"
    assert len(filtered.episodes) <= len(unfiltered.episodes), (
        "Filtered episodes should be a subset of unfiltered episodes"
    )


@pytest.mark.asyncio
async def test_list_episodes_with_float_filter_async(
    embedded_async_client: AsyncTensorZeroGateway,
):
    # Get unfiltered count for comparison
    unfiltered = await embedded_async_client.list_episodes(request=ListEpisodesRequest(limit=100))

    # Filter by float metric
    filtered = await embedded_async_client.list_episodes(
        request=ListEpisodesRequest(
            limit=100,
            filters=InferenceFilterFloatMetric(
                metric_name="user_rating",
                value=4.0,
                comparison_operator=">=",
            ),
        )
    )
    assert isinstance(filtered, ListEpisodesResponse)
    assert len(filtered.episodes) > 0, "Expected at least one episode matching the float metric filter"
    assert len(filtered.episodes) <= len(unfiltered.episodes), (
        "Filtered episodes should be a subset of unfiltered episodes"
    )


@pytest.mark.asyncio
async def test_list_episodes_combined_filters_async(
    embedded_async_client: AsyncTensorZeroGateway,
):
    # Get unfiltered count for comparison
    unfiltered = await embedded_async_client.list_episodes(request=ListEpisodesRequest(limit=100))

    # Combine boolean and float filters with AND
    filtered = await embedded_async_client.list_episodes(
        request=ListEpisodesRequest(
            limit=100,
            function_name="basic_test",
            filters=InferenceFilterAnd(
                children=[
                    InferenceFilterBooleanMetric(
                        metric_name="task_success",
                        value=True,
                    ),
                    InferenceFilterFloatMetric(
                        metric_name="user_rating",
                        value=4.0,
                        comparison_operator=">=",
                    ),
                ],
            ),
        )
    )
    assert isinstance(filtered, ListEpisodesResponse)
    assert len(filtered.episodes) <= len(unfiltered.episodes), (
        "Filtered episodes should be a subset of unfiltered episodes"
    )
