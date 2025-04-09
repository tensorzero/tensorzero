import inspect
import os

import pytest
from tensorzero import AsyncTensorZeroGateway


@pytest.mark.asyncio
async def test_curated_inferences():
    client_fut = AsyncTensorZeroGateway.build_embedded(
        config_file="../ui/fixtures/config/tensorzero.toml",
        clickhouse_url=os.environ["TENSORZERO_CLICKHOUSE_URL"],
    )
    assert inspect.isawaitable(client_fut)
    client = await client_fut

    boolean_results = await client._internal_get_curated_inferences(
        function_name="extract_entities",
        metric_name="exact_match",
        threshold=0,
        max_samples=None,
    )
    assert len(boolean_results) == 41

    float_results = await client._internal_get_curated_inferences(
        function_name="write_haiku",
        metric_name="haiku_rating",
        threshold=0.8,
        max_samples=None,
    )
    assert len(float_results) == 67

    all_results = await client._internal_get_curated_inferences(
        function_name="extract_entities",
        metric_name=None,
        threshold=0.8,
        max_samples=None,
    )
    assert len(all_results) == 566
