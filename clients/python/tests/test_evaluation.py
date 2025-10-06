"""
Tests for the experimental_run_evaluation function

These tests validate that the Python client can run evaluations using both
sync and async clients in embedded gateway mode.
"""

from typing import Any, Dict, List

import pytest
from tensorzero import (
    AsyncTensorZeroGateway,
    TensorZeroGateway,
    TensorZeroInternalError,
)


def test_sync_run_evaluation(
    evaluation_datasets: Dict[str, str],
    embedded_sync_client: TensorZeroGateway,
):
    """Test sync client experimental_run_evaluation."""
    job = embedded_sync_client.experimental_run_evaluation(
        evaluation_name="entity_extraction",
        dataset_name=evaluation_datasets["extract_entities_0.8"],
        variant_name="gpt_4o_mini",
        concurrency=2,
        inference_cache="on",
    )

    # Test run_info property
    run_info: Dict[str, Any] = job.run_info
    assert "evaluation_run_id" in run_info
    assert "num_datapoints" in run_info
    assert run_info["num_datapoints"] > 0

    # Consume all results
    results: List[Dict[str, Any]] = []
    for result in job.results():
        results.append(result)
        assert "type" in result
        assert result["type"] in ["success", "error"]

    assert len(results) > 0

    # Test summary stats
    stats: Dict[str, Dict[str, float]] = job.summary_stats()
    assert isinstance(stats, dict)


def test_sync_run_evaluation_invalid_cache_mode(
    embedded_sync_client: TensorZeroGateway,
):
    """Test sync client experimental_run_evaluation with invalid cache mode."""
    with pytest.raises(TensorZeroInternalError, match="unknown variant"):
        embedded_sync_client.experimental_run_evaluation(
            evaluation_name="entity_extraction",
            dataset_name="extract_entities_0.8",
            variant_name="gpt_4o_mini",
            concurrency=1,
            inference_cache="invalid_mode",
        )


@pytest.mark.asyncio
async def test_async_run_evaluation(
    evaluation_datasets: Dict[str, str],
    embedded_async_client: AsyncTensorZeroGateway,
):
    """Test async client experimental_run_evaluation."""
    job = await embedded_async_client.experimental_run_evaluation(
        evaluation_name="haiku_without_outputs",
        dataset_name=evaluation_datasets["good-haikus-no-output"],
        variant_name="gpt_4o_mini",
        concurrency=2,
        inference_cache="off",
    )

    # Test run_info property
    run_info: Dict[str, Any] = job.run_info
    assert "evaluation_run_id" in run_info
    assert "num_datapoints" in run_info

    # Consume all results
    results: List[Dict[str, Any]] = []
    async for result in job.results():
        results.append(result)
        assert "type" in result
        assert result["type"] in ["success", "error"]

    assert len(results) > 0

    # Test summary stats
    stats: Dict[str, Dict[str, float]] = await job.summary_stats()
    assert isinstance(stats, dict)


@pytest.mark.asyncio
async def test_async_run_evaluation_invalid_cache_mode(
    embedded_async_client: AsyncTensorZeroGateway,
):
    """Test async client experimental_run_evaluation with invalid cache mode."""
    with pytest.raises(TensorZeroInternalError, match="unknown variant"):
        await embedded_async_client.experimental_run_evaluation(
            evaluation_name="entity_extraction",
            dataset_name="extract_entities_0.8",
            variant_name="gpt_4o_mini",
            concurrency=1,
            inference_cache="invalid_mode",
        )
