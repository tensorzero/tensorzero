import pytest
from tensorzero import (
    AsyncTensorZeroGateway,
    EpisodeByIdRow,
    ListEpisodesResponse,
    TensorZeroGateway,
)


def test_list_episodes(embedded_sync_client: TensorZeroGateway):
    response = embedded_sync_client.list_episodes(limit=5)
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
    first_page = embedded_sync_client.list_episodes(limit=2)
    assert isinstance(first_page, ListEpisodesResponse)
    assert len(first_page.episodes) == 2

    # Get the next page using `before` cursor
    last_episode = first_page.episodes[-1]
    second_page = embedded_sync_client.list_episodes(limit=2, before=last_episode.episode_id)
    assert isinstance(second_page, ListEpisodesResponse)

    # Pages should not overlap
    first_page_ids = {e.episode_id for e in first_page.episodes}
    second_page_ids = {e.episode_id for e in second_page.episodes}
    assert first_page_ids.isdisjoint(second_page_ids), "Paginated pages should not overlap"


def test_list_episodes_invalid_uuid(embedded_sync_client: TensorZeroGateway):
    with pytest.raises(ValueError, match="Invalid `before` UUID"):
        embedded_sync_client.list_episodes(limit=5, before="not-a-uuid")

    with pytest.raises(ValueError, match="Invalid `after` UUID"):
        embedded_sync_client.list_episodes(limit=5, after="not-a-uuid")


# Async versions


@pytest.mark.asyncio
async def test_list_episodes_async(
    embedded_async_client: AsyncTensorZeroGateway,
):
    response = await embedded_async_client.list_episodes(limit=5)
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
    first_page = await embedded_async_client.list_episodes(limit=2)
    assert isinstance(first_page, ListEpisodesResponse)
    assert len(first_page.episodes) == 2

    # Get the next page using `before` cursor
    last_episode = first_page.episodes[-1]
    second_page = await embedded_async_client.list_episodes(limit=2, before=last_episode.episode_id)
    assert isinstance(second_page, ListEpisodesResponse)

    # Pages should not overlap
    first_page_ids = {e.episode_id for e in first_page.episodes}
    second_page_ids = {e.episode_id for e in second_page.episodes}
    assert first_page_ids.isdisjoint(second_page_ids), "Paginated pages should not overlap"


@pytest.mark.asyncio
async def test_list_episodes_invalid_uuid_async(
    embedded_async_client: AsyncTensorZeroGateway,
):
    with pytest.raises(ValueError, match="Invalid `before` UUID"):
        await embedded_async_client.list_episodes(limit=5, before="not-a-uuid")

    with pytest.raises(ValueError, match="Invalid `after` UUID"):
        await embedded_async_client.list_episodes(limit=5, after="not-a-uuid")
