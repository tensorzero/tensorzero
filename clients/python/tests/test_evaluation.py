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


def test_sync_run_evaluation_with_dynamic_variant(
    evaluation_datasets: Dict[str, str],
    embedded_sync_client: TensorZeroGateway,
):
    """Test sync client experimental_run_evaluation with dynamic variant.

    This test mirrors test_async_run_evaluation but uses a dynamic variant config
    instead of a config-file defined variant. It validates that dynamic variants
    produce the same results as config-defined variants.
    """
    # Define dynamic variant that replicates the gpt_4o_mini variant for write_haiku
    dynamic_variant = {
        "type": "chat_completion",
        "model": "gpt-4o-mini-2024-07-18",
        "system_template": {
            "__tensorzero_remapped_path": "system",
            "__data": "You are a haiku writer. You will be given a topic and you will write a haiku about the topic.\nA haiku is a three line poem with a 5-7-5 syllable pattern.\nYou may think out loud, but please be sure that the last 3 lines you write are the haiku.",
        },
        "user_template": {
            "__tensorzero_remapped_path": "user",
            "__data": "The topic is {{ topic }}.",
        },
    }

    # Run evaluation with dynamic variant
    job = embedded_sync_client.experimental_run_evaluation(
        evaluation_name="haiku_without_outputs",
        dataset_name=evaluation_datasets["good-haikus-no-output"],
        dynamic_variant_config=dynamic_variant,
        concurrency=10,
        inference_cache="off",
    )

    # Verify job runs successfully
    run_info: Dict[str, Any] = job.run_info
    assert "evaluation_run_id" in run_info
    assert "num_datapoints" in run_info
    assert run_info["num_datapoints"] == 10

    # Consume results and verify structure
    results: List[Dict[str, Any]] = []
    for result in job.results():
        results.append(result)
        assert "type" in result
        if result["type"] == "success":
            assert "datapoint" in result
            assert "response" in result
            assert "evaluations" in result
            assert "evaluator_errors" in result
        elif result["type"] == "error":
            assert "datapoint" in result
            assert "error" in result

    assert len(results) == 10

    # Verify summary stats - should match test_async_run_evaluation results
    stats: Dict[str, EvaluatorStatsDict] = job.summary_stats()
    assert isinstance(stats, dict)

    # exact_match should be 0 (no reference outputs in dataset)
    assert "exact_match" in stats
    assert abs(stats["exact_match"]["mean"] - 0.00) < 0.005
    assert abs(stats["exact_match"]["stderr"] - 0.00) < 0.005
    assert stats["exact_match"]["count"] == 0

    # topic_starts_with_f should be ~0.30 (3 out of 10 topics start with 'f')
    assert "topic_starts_with_f" in stats
    assert abs(stats["topic_starts_with_f"]["mean"] - 0.30) < 0.005
    assert abs(stats["topic_starts_with_f"]["stderr"] - 0.14) < 0.005
    assert stats["topic_starts_with_f"]["count"] == 10


@pytest.mark.asyncio
async def test_async_run_evaluation_with_dynamic_variant(
    evaluation_datasets: Dict[str, str],
    embedded_async_client: AsyncTensorZeroGateway,
):
    """Test async client experimental_run_evaluation with dynamic variant.

    This test mirrors test_sync_run_evaluation but uses a dynamic variant config
    instead of a config-file defined variant. It validates that dynamic variants
    produce the same results as config-defined variants for JSON inference.
    """
    # Define dynamic variant that replicates the gpt_4o_mini variant for extract_entities
    dynamic_variant = {
        "type": "chat_completion",
        "model": "gpt-4o-mini-2024-07-18",
        "system_template": {
            "__tensorzero_remapped_path": "system",
            "__data": 'You are an assistant that is performing a named entity recognition task.\nYour job is to extract entities from a given text.\n\nThe entities you are extracting are:\n- people\n- organizations\n- locations\n- miscellaneous other entities\n\nPlease return the entities in the following JSON format:\n\n{\n    "person": ["person1", "person2", ...],\n    "organization": ["organization1", "organization2", ...],\n    "location": ["location1", "location2", ...],\n    "miscellaneous": ["miscellaneous1", "miscellaneous2", ...]\n}',
        },
        "json_mode": "strict",
    }

    # Run evaluation with dynamic variant
    job = await embedded_async_client.experimental_run_evaluation(
        evaluation_name="entity_extraction",
        dataset_name=evaluation_datasets["extract_entities_0.8"],
        dynamic_variant_config=dynamic_variant,
        concurrency=10,
        inference_cache="off",
    )

    # Verify job runs successfully
    run_info: Dict[str, Any] = job.run_info
    assert "evaluation_run_id" in run_info
    assert "num_datapoints" in run_info
    assert run_info["num_datapoints"] > 0

    # Consume results and verify structure
    results: List[Dict[str, Any]] = []
    async for result in job.results():
        results.append(result)
        assert "type" in result
        if result["type"] == "success":
            assert "datapoint" in result
            assert "response" in result
            assert "evaluations" in result
            assert "evaluator_errors" in result
            # Validate evaluator results
            evals = result["evaluations"]
            assert "exact_match" in evals
            assert evals["exact_match"] is True or evals["exact_match"] is False or evals["exact_match"] is None
            assert "count_sports" in evals
            assert evals["count_sports"] is None or isinstance(evals["count_sports"], (int, float))
        elif result["type"] == "error":
            assert "datapoint" in result
            assert "error" in result

    assert len(results) == run_info["num_datapoints"]

    # Verify summary stats - should match test_sync_run_evaluation results
    stats: Dict[str, EvaluatorStatsDict] = await job.summary_stats()
    assert isinstance(stats, dict)

    # exact_match evaluator should have some results
    assert "exact_match" in stats
    assert isinstance(stats["exact_match"]["mean"], float)
    assert isinstance(stats["exact_match"]["stderr"], float)
    assert isinstance(stats["exact_match"]["count"], int)

    # count_sports should be ~0.50 (half of inputs are sports-related)
    assert "count_sports" in stats
    assert abs(stats["count_sports"]["mean"] - 0.50) < 0.005
    assert abs(stats["count_sports"]["stderr"] - 0.20) < 0.005
    assert stats["count_sports"]["count"] > 0


def test_sync_run_evaluation_both_variant_params_error(
    evaluation_datasets: Dict[str, str],
    embedded_sync_client: TensorZeroGateway,
):
    """Test sync client experimental_run_evaluation rejects both variant_name and dynamic_variant_config."""
    dynamic_variant = {
        "type": "chat_completion",
        "model": "gpt-4o-mini-2024-07-18",
        "system_template": {
            "__tensorzero_remapped_path": "system",
            "__data": "You are a test assistant.",
        },
    }

    # Providing both variant_name and dynamic_variant_config should raise ValueError
    with pytest.raises(ValueError, match="Cannot specify both.*variant_name.*dynamic_variant_config"):
        embedded_sync_client.experimental_run_evaluation(
            evaluation_name="haiku_without_outputs",
            dataset_name=evaluation_datasets["good-haikus-no-output"],
            variant_name="gpt_4o_mini",
            dynamic_variant_config=dynamic_variant,
            concurrency=1,
            inference_cache="off",
        )


@pytest.mark.asyncio
async def test_async_run_evaluation_both_variant_params_error(
    evaluation_datasets: Dict[str, str],
    embedded_async_client: AsyncTensorZeroGateway,
):
    """Test async client experimental_run_evaluation rejects both variant_name and dynamic_variant_config."""
    dynamic_variant = {
        "type": "chat_completion",
        "model": "gpt-4o-mini-2024-07-18",
        "system_template": {
            "__tensorzero_remapped_path": "system",
            "__data": "You are a test assistant.",
        },
    }

    # Providing both variant_name and dynamic_variant_config should raise ValueError
    with pytest.raises(ValueError, match="Cannot specify both.*variant_name.*dynamic_variant_config"):
        await embedded_async_client.experimental_run_evaluation(
            evaluation_name="haiku_without_outputs",
            dataset_name=evaluation_datasets["good-haikus-no-output"],
            variant_name="gpt_4o_mini",
            dynamic_variant_config=dynamic_variant,
            concurrency=1,
            inference_cache="off",
        )


def test_sync_run_evaluation_with_limit(
    evaluation_datasets: Dict[str, str],
    embedded_sync_client: TensorZeroGateway,
):
    """Test sync client experimental_run_evaluation with limit parameter."""
    job = embedded_sync_client.experimental_run_evaluation(
        evaluation_name="haiku_without_outputs",
        dataset_name=evaluation_datasets["good-haikus-no-output"],
        variant_name="gpt_4o_mini",
        concurrency=2,
        inference_cache="off",
        limit=5,
    )

    # Test run_info property
    run_info: Dict[str, Any] = job.run_info
    assert "evaluation_run_id" in run_info
    assert "num_datapoints" in run_info
    assert run_info["num_datapoints"] == 5  # Should be limited to 5

    # Consume all results
    results: List[Dict[str, Any]] = []
    for result in job.results():
        results.append(result)

    # Should have exactly 5 results due to limit
    assert len(results) == 5

    # Verify all results are success or error types
    for result in results:
        assert result["type"] in ["success", "error"]


@pytest.mark.asyncio
async def test_async_run_evaluation_with_limit(
    evaluation_datasets: Dict[str, str],
    embedded_async_client: AsyncTensorZeroGateway,
):
    """Test async client experimental_run_evaluation with limit parameter."""
    job = await embedded_async_client.experimental_run_evaluation(
        evaluation_name="haiku_without_outputs",
        dataset_name=evaluation_datasets["good-haikus-no-output"],
        variant_name="gpt_4o_mini",
        concurrency=2,
        inference_cache="off",
        limit=3,
    )

    # Test run_info property
    run_info: Dict[str, Any] = job.run_info
    assert "evaluation_run_id" in run_info
    assert "num_datapoints" in run_info
    assert run_info["num_datapoints"] == 3  # Should be limited to 3

    # Consume all results
    results: List[Dict[str, Any]] = []
    async for result in job.results():
        results.append(result)

    # Should have exactly 3 results due to limit
    assert len(results) == 3

    # Verify all results are success or error types
    for result in results:
        assert result["type"] in ["success", "error"]


def test_sync_run_evaluation_with_offset(
    evaluation_datasets: Dict[str, str],
    embedded_sync_client: TensorZeroGateway,
):
    """Test sync client experimental_run_evaluation with offset parameter."""
    # Run without offset to get baseline
    job1 = embedded_sync_client.experimental_run_evaluation(
        evaluation_name="haiku_without_outputs",
        dataset_name=evaluation_datasets["good-haikus-no-output"],
        variant_name="gpt_4o_mini",
        concurrency=2,
        inference_cache="off",
        limit=5,
        offset=0,
    )

    results1: List[Dict[str, Any]] = []
    for result in job1.results():
        results1.append(result)

    # Run with offset to skip first 2 datapoints
    job2 = embedded_sync_client.experimental_run_evaluation(
        evaluation_name="haiku_without_outputs",
        dataset_name=evaluation_datasets["good-haikus-no-output"],
        variant_name="gpt_4o_mini",
        concurrency=2,
        inference_cache="off",
        limit=3,
        offset=2,
    )

    results2: List[Dict[str, Any]] = []
    for result in job2.results():
        results2.append(result)

    # With offset=2 and limit=3, should get 3 results
    assert len(results2) == 3

    # Verify that results are different (offset should skip different datapoints)
    # Note: Results come back in async order, so we need to sort by datapoint ID in descending order
    # to match the database ordering (newest first). We then take the 3rd, 4th, 5th datapoints
    # (indices 2:5) from results1 and compare with results2
    if all(r["type"] == "success" for r in results1) and all(r["type"] == "success" for r in results2):
        # Sort results1 by datapoint ID in descending order (newest first), then extract IDs for positions 2-4 (0-indexed)
        sorted_results1 = sorted(results1, key=lambda r: r["datapoint"]["id"], reverse=True)
        ids1 = [r["datapoint"]["id"] for r in sorted_results1[2:5]]

        # Sort results2 by datapoint ID in descending order (newest first) and extract IDs
        sorted_results2 = sorted(results2, key=lambda r: r["datapoint"]["id"], reverse=True)
        ids2 = [r["datapoint"]["id"] for r in sorted_results2]

        assert ids1 == ids2, "Offset should skip the first 2 datapoints"


@pytest.mark.asyncio
async def test_async_run_evaluation_with_limit_and_offset(
    evaluation_datasets: Dict[str, str],
    embedded_async_client: AsyncTensorZeroGateway,
):
    """Test async client experimental_run_evaluation with both limit and offset parameters."""
    job = await embedded_async_client.experimental_run_evaluation(
        evaluation_name="haiku_without_outputs",
        dataset_name=evaluation_datasets["good-haikus-no-output"],
        variant_name="gpt_4o_mini",
        concurrency=2,
        inference_cache="off",
        limit=4,
        offset=3,
    )

    # Test run_info property
    run_info: Dict[str, Any] = job.run_info
    assert run_info["num_datapoints"] == 4  # Should be limited to 4

    # Consume all results
    results: List[Dict[str, Any]] = []
    async for result in job.results():
        results.append(result)

    # Should have exactly 4 results (limit=4, offset=3 means datapoints 3-6)
    assert len(results) == 4
