"""
Tests for the experimental_run_evaluation function

These tests validate that the Python client can run evaluations using both
sync and async clients in embedded gateway mode.
"""

import os

import pytest
from tensorzero import AsyncTensorZeroGateway, TensorZeroGateway

TEST_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../tensorzero-core/tests/e2e/tensorzero.toml",
)

CLICKHOUSE_URL = "http://chuser:chpassword@localhost:8123/tensorzero-python-e2e"


def test_sync_run_evaluation(evaluation_datasets):
    """Test sync client experimental_run_evaluation."""
    with TensorZeroGateway.build_embedded(
        config_file=TEST_CONFIG_FILE,
        clickhouse_url=CLICKHOUSE_URL,
    ) as client:
        job = client.experimental_run_evaluation(
            evaluation_name="entity_extraction",
            dataset_name=evaluation_datasets["extract_entities_0.8"],
            variant_name="gpt_4o_mini",
            concurrency=2,
            inference_cache="on",
        )

        # Test run_info property
        run_info = job.run_info
        assert "evaluation_run_id" in run_info
        assert "num_datapoints" in run_info
        assert run_info["num_datapoints"] > 0

        # Consume all results
        results = []
        for result in job.results():
            results.append(result)
            assert "type" in result
            assert result["type"] in ["success", "error"]

        assert len(results) > 0

        # Test summary stats
        stats = job.summary_stats()
        assert isinstance(stats, dict)


def test_sync_run_evaluation_invalid_cache_mode():
    """Test sync client experimental_run_evaluation with invalid cache mode."""
    with TensorZeroGateway.build_embedded(
        config_file=TEST_CONFIG_FILE,
        clickhouse_url=CLICKHOUSE_URL,
    ) as client:
        with pytest.raises(ValueError, match="Invalid inference_cache"):
            client.experimental_run_evaluation(
                evaluation_name="entity_extraction",
                dataset_name="extract_entities_0.8",
                variant_name="gpt_4o_mini",
                concurrency=1,
                inference_cache="invalid_mode",
            )


@pytest.mark.asyncio
async def test_async_run_evaluation(evaluation_datasets):
    """Test async client experimental_run_evaluation."""
    async with await AsyncTensorZeroGateway.build_embedded(
        config_file=TEST_CONFIG_FILE,
        clickhouse_url=CLICKHOUSE_URL,
    ) as client:
        job = await client.experimental_run_evaluation(
            evaluation_name="haiku_without_outputs",
            dataset_name=evaluation_datasets["good-haikus-no-output"],
            variant_name="gpt_4o_mini",
            concurrency=2,
            inference_cache="off",
        )

        # Test run_info property
        run_info = job.run_info
        assert "evaluation_run_id" in run_info
        assert "num_datapoints" in run_info

        # Consume all results
        results = []
        async for result in job.results():
            results.append(result)
            assert "type" in result
            assert result["type"] in ["success", "error"]

        assert len(results) > 0

        # Test summary stats
        stats = await job.summary_stats()
        assert isinstance(stats, dict)


@pytest.mark.asyncio
async def test_async_run_evaluation_invalid_cache_mode():
    """Test async client experimental_run_evaluation with invalid cache mode."""
    async with await AsyncTensorZeroGateway.build_embedded(
        config_file=TEST_CONFIG_FILE,
        clickhouse_url=CLICKHOUSE_URL,
    ) as client:
        with pytest.raises(ValueError, match="Invalid inference_cache"):
            await client.experimental_run_evaluation(
                evaluation_name="entity_extraction",
                dataset_name="extract_entities_0.8",
                variant_name="gpt_4o_mini",
                concurrency=1,
                inference_cache="invalid_mode",
            )
