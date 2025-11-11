"""
Tests for the experimental_run_evaluation function

These tests validate that the Python client can run evaluations using both
sync and async clients in embedded gateway mode.
"""

from typing import Any, Dict, List

import pytest
from tensorzero import (
    AsyncTensorZeroGateway,
    EvaluatorStatsDict,
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
    assert isinstance(run_info["evaluation_run_id"], str)
    assert len(run_info["evaluation_run_id"]) == 36  # UUID string length
    assert isinstance(run_info["num_datapoints"], int)
    assert run_info["num_datapoints"] > 0
    assert set(run_info.keys()) == {"evaluation_run_id", "num_datapoints"}

    # Consume all results
    results: List[Dict[str, Any]] = []
    for result in job.results():
        results.append(result)
        assert "type" in result
        assert result["type"] in ["success", "error"]

        if result["type"] == "success":
            # Validate success result structure
            assert "datapoint" in result
            assert "response" in result
            assert "evaluations" in result
            assert "evaluator_errors" in result
            assert isinstance(result["datapoint"], dict)
            assert isinstance(result["response"], dict)
            assert isinstance(result["evaluations"], dict)
            assert isinstance(result["evaluator_errors"], dict)

            # Validate evaluators specific to entity_extraction
            assert "exact_match" in result["evaluations"]
            assert "count_sports" in result["evaluations"]
            # exact_match can be bool or None
            if result["evaluations"]["exact_match"] is not None:
                assert isinstance(result["evaluations"]["exact_match"], bool)
            # count_sports should be numeric
            if result["evaluations"]["count_sports"] is not None:
                assert isinstance(result["evaluations"]["count_sports"], (int, float))
        else:  # error
            # Validate error result structure
            assert "datapoint_id" in result
            assert "message" in result
            assert isinstance(result["datapoint_id"], str)
            assert isinstance(result["message"], str)
            assert len(result["message"]) > 0

    assert len(results) == run_info["num_datapoints"]

    # Test summary stats
    stats: Dict[str, EvaluatorStatsDict] = job.summary_stats()
    assert isinstance(stats, dict)
    assert len(stats) > 0  # Should have at least one evaluator

    # Validate specific evaluators for entity_extraction
    assert "exact_match" in stats
    assert "count_sports" in stats

    # Validate each evaluator's stats structure
    for evaluator_name, evaluator_stats in stats.items():
        assert isinstance(evaluator_name, str)
        assert isinstance(evaluator_stats, dict)
        assert "mean" in evaluator_stats
        assert "stderr" in evaluator_stats
        assert "count" in evaluator_stats
        assert isinstance(evaluator_stats["mean"], (int, float))
        assert isinstance(evaluator_stats["stderr"], (int, float))
        assert isinstance(evaluator_stats["count"], int)
        assert evaluator_stats["stderr"] >= 0  # Stderr is always non-negative
        assert evaluator_stats["count"] >= 0
        assert set(evaluator_stats.keys()) == {"mean", "stderr", "count"}

    # Validate precise expected values based on dataset
    # Expected mean 0.50, stderr 0.20 for this dataset
    assert abs(stats["count_sports"]["mean"] - 0.50) < 0.005
    assert abs(stats["count_sports"]["stderr"] - 0.20) < 0.005
    assert stats["count_sports"]["count"] == run_info["num_datapoints"]
    # Note: exact_match mean is not checked precisely as it's non-deterministic


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
    assert isinstance(run_info["evaluation_run_id"], str)
    assert len(run_info["evaluation_run_id"]) == 36  # UUID string length
    assert isinstance(run_info["num_datapoints"], int)
    assert run_info["num_datapoints"] > 0
    assert set(run_info.keys()) == {"evaluation_run_id", "num_datapoints"}

    # Consume all results
    results: List[Dict[str, Any]] = []
    async for result in job.results():
        results.append(result)
        assert "type" in result
        assert result["type"] in ["success", "error"]

        if result["type"] == "success":
            # Validate success result structure
            assert "datapoint" in result
            assert "response" in result
            assert "evaluations" in result
            assert "evaluator_errors" in result
            assert isinstance(result["datapoint"], dict)
            assert isinstance(result["response"], dict)
            assert isinstance(result["evaluations"], dict)
            assert isinstance(result["evaluator_errors"], dict)

            # Validate evaluators specific to haiku_without_outputs
            assert "exact_match" in result["evaluations"]
            assert "topic_starts_with_f" in result["evaluations"]
            # Both can be bool or None
            if result["evaluations"]["exact_match"] is not None:
                assert isinstance(result["evaluations"]["exact_match"], bool)
            if result["evaluations"]["topic_starts_with_f"] is not None:
                assert isinstance(result["evaluations"]["topic_starts_with_f"], bool)
        else:  # error
            # Validate error result structure
            assert "datapoint_id" in result
            assert "message" in result
            assert isinstance(result["datapoint_id"], str)
            assert isinstance(result["message"], str)
            assert len(result["message"]) > 0

    assert len(results) == run_info["num_datapoints"]

    # Test summary stats
    stats: Dict[str, EvaluatorStatsDict] = await job.summary_stats()
    assert isinstance(stats, dict)
    assert len(stats) > 0  # Should have at least one evaluator

    # Validate specific evaluators for haiku_without_outputs
    assert "exact_match" in stats
    assert "topic_starts_with_f" in stats

    # Validate each evaluator's stats structure
    for evaluator_name, evaluator_stats in stats.items():
        assert isinstance(evaluator_name, str)
        assert isinstance(evaluator_stats, dict)
        assert "mean" in evaluator_stats
        assert "stderr" in evaluator_stats
        assert "count" in evaluator_stats
        assert isinstance(evaluator_stats["mean"], (int, float))
        assert isinstance(evaluator_stats["stderr"], (int, float))
        assert isinstance(evaluator_stats["count"], int)
        assert evaluator_stats["stderr"] >= 0  # Stderr is always non-negative
        assert evaluator_stats["count"] >= 0
        assert set(evaluator_stats.keys()) == {"mean", "stderr", "count"}

        # Validate reasonable ranges for evaluators
        if evaluator_name in ["exact_match", "topic_starts_with_f"]:
            assert 0 <= evaluator_stats["mean"] <= 1

    # Validate precise expected values based on dataset
    # exact_match: Should be 0.00 ± 0.00
    assert abs(stats["exact_match"]["mean"] - 0.00) < 0.005
    assert abs(stats["exact_match"]["stderr"] - 0.00) < 0.005
    assert stats["exact_match"]["count"] == 0
    # topic_starts_with_f: 3 out of 10 topics start with 'f' (fusarium, force, formamide)
    # Expected: 0.30 ± 0.14
    assert abs(stats["topic_starts_with_f"]["mean"] - 0.30) < 0.005
    assert abs(stats["topic_starts_with_f"]["stderr"] - 0.14) < 0.005
    assert stats["topic_starts_with_f"]["count"] == 10


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
