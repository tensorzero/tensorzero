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


# NEW TESTS FOR DYNAMIC VARIANT CONFIG


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
        internal_dynamic_variant_config=dynamic_variant,
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
        internal_dynamic_variant_config=dynamic_variant,
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
    """Test sync client experimental_run_evaluation rejects both variant_name and internal_dynamic_variant_config."""
    dynamic_variant = {
        "type": "chat_completion",
        "model": "gpt-4o-mini-2024-07-18",
        "system_template": {
            "__tensorzero_remapped_path": "system",
            "__data": "You are a test assistant.",
        },
    }

    # Providing both variant_name and internal_dynamic_variant_config should raise ValueError
    with pytest.raises(ValueError, match="Cannot specify both.*variant_name.*internal_dynamic_variant_config"):
        embedded_sync_client.experimental_run_evaluation(
            evaluation_name="haiku_without_outputs",
            dataset_name=evaluation_datasets["good-haikus-no-output"],
            variant_name="gpt_4o_mini",
            internal_dynamic_variant_config=dynamic_variant,
            concurrency=1,
            inference_cache="off",
        )


@pytest.mark.asyncio
async def test_async_run_evaluation_both_variant_params_error(
    evaluation_datasets: Dict[str, str],
    embedded_async_client: AsyncTensorZeroGateway,
):
    """Test async client experimental_run_evaluation rejects both variant_name and internal_dynamic_variant_config."""
    dynamic_variant = {
        "type": "chat_completion",
        "model": "gpt-4o-mini-2024-07-18",
        "system_template": {
            "__tensorzero_remapped_path": "system",
            "__data": "You are a test assistant.",
        },
    }

    # Providing both variant_name and internal_dynamic_variant_config should raise ValueError
    with pytest.raises(ValueError, match="Cannot specify both.*variant_name.*internal_dynamic_variant_config"):
        await embedded_async_client.experimental_run_evaluation(
            evaluation_name="haiku_without_outputs",
            dataset_name=evaluation_datasets["good-haikus-no-output"],
            variant_name="gpt_4o_mini",
            internal_dynamic_variant_config=dynamic_variant,
            concurrency=1,
            inference_cache="off",
        )


def test_sync_run_evaluation_with_adaptive_stopping(
    evaluation_datasets: Dict[str, str],
    embedded_sync_client: TensorZeroGateway,
):
    """Test sync client experimental_run_evaluation with adaptive stopping parameters."""
    job = embedded_sync_client.experimental_run_evaluation(
        evaluation_name="entity_extraction",
        dataset_name=evaluation_datasets["extract_entities_0.8"],
        variant_name="gpt_4o_mini",
        concurrency=2,
        inference_cache="on",
        max_datapoints=4,
        adaptive_stopping={"precision": {"exact_match": 0.2}},
    )

    # Test run_info property
    run_info: Dict[str, Any] = job.run_info
    assert "evaluation_run_id" in run_info
    assert "num_datapoints" in run_info
    assert isinstance(run_info["num_datapoints"], int)

    # With adaptive stopping, we should stop early if precision threshold is met
    # Dataset only has 6 datapoints, so it can't reach MIN_DATAPOINTS (20)
    # Should respect max_datapoints (4) or dataset size, whichever is smaller
    num_datapoints = run_info["num_datapoints"]
    assert num_datapoints <= 4, f"Should process all 6 datapoints in dataset, got {num_datapoints}"

    # Consume all results
    results: List[Dict[str, Any]] = []
    for result in job.results():
        results.append(result)
        assert "type" in result
        assert result["type"] in ["success", "error"]

    assert len(results) == num_datapoints

    # Test summary stats
    stats: Dict[str, EvaluatorStatsDict] = job.summary_stats()
    assert isinstance(stats, dict)
    assert "exact_match" in stats
    assert "count_sports" in stats


@pytest.mark.asyncio
async def test_async_run_evaluation_with_adaptive_stopping(
    evaluation_datasets: Dict[str, str],
    embedded_async_client: AsyncTensorZeroGateway,
):
    """Test async client experimental_run_evaluation with adaptive stopping parameters."""
    job = await embedded_async_client.experimental_run_evaluation(
        evaluation_name="haiku_without_outputs",
        dataset_name=evaluation_datasets["good-haikus-no-output"],
        variant_name="gpt_4o_mini",
        concurrency=2,
        inference_cache="off",
        max_datapoints=7,
        adaptive_stopping={"precision": {"topic_starts_with_f": 0.3}},
    )

    # Test run_info property
    run_info: Dict[str, Any] = job.run_info
    assert "evaluation_run_id" in run_info
    assert "num_datapoints" in run_info
    assert isinstance(run_info["num_datapoints"], int)

    # With adaptive stopping, we should stop early if precision threshold is met
    # Dataset only has 10 datapoints, so it can't reach MIN_DATAPOINTS (20)
    # Should respect max_datapoints (7) or dataset size, whichever is smaller
    num_datapoints = run_info["num_datapoints"]
    assert num_datapoints <= 7, f"Should process max of 7 datapoints, got {num_datapoints}"

    # Consume all results
    results: List[Dict[str, Any]] = []
    async for result in job.results():
        results.append(result)
        assert "type" in result
        assert result["type"] in ["success", "error"]

    assert len(results) == num_datapoints

    # Test summary stats
    stats: Dict[str, EvaluatorStatsDict] = await job.summary_stats()
    assert isinstance(stats, dict)
    assert "exact_match" in stats
    assert "topic_starts_with_f" in stats


# TESTS FOR DATAPOINT_IDS PARAMETER


def test_sync_run_evaluation_with_datapoint_ids(
    evaluation_datasets: Dict[str, str],
    embedded_sync_client: TensorZeroGateway,
):
    """Test sync client experimental_run_evaluation with specific datapoint_ids.

    This is a happy path test that:
    1. Runs an evaluation to collect available datapoint IDs
    2. Selects a subset of those IDs
    3. Runs evaluation with only the selected IDs
    4. Verifies only the selected datapoints were evaluated
    """
    # First, run evaluation to collect available datapoint IDs from the dataset
    first_job = embedded_sync_client.experimental_run_evaluation(
        evaluation_name="entity_extraction",
        dataset_name=evaluation_datasets["extract_entities_0.8"],
        variant_name="gpt_4o_mini",
        concurrency=2,
        inference_cache="on",
    )

    # Collect all datapoint IDs from the first run
    all_datapoint_ids: list[str] = []
    for result in first_job.results():
        if result["type"] == "success":
            datapoint_id = result["datapoint"]["id"]
            all_datapoint_ids.append(datapoint_id)

    # We should have at least some datapoints
    assert len(all_datapoint_ids) > 0, "Dataset should contain datapoints"

    # Select first 3 datapoint IDs (or fewer if dataset is smaller)
    num_to_select = min(3, len(all_datapoint_ids))
    selected_ids = all_datapoint_ids[:num_to_select]

    # Run evaluation with only the selected datapoint IDs
    second_job = embedded_sync_client.experimental_run_evaluation(
        evaluation_name="entity_extraction",
        datapoint_ids=selected_ids,
        variant_name="gpt_4o_mini",
        concurrency=2,
        inference_cache="on",
    )

    # Verify run_info reports correct number of datapoints
    run_info: Dict[str, Any] = second_job.run_info
    assert run_info["num_datapoints"] == num_to_select, (
        f"Expected {num_to_select} datapoints, got {run_info['num_datapoints']}"
    )

    # Collect results and verify they match selected IDs
    evaluated_ids: list[str] = []
    for result in second_job.results():
        assert result["type"] == "success", "All evaluations should succeed"
        datapoint_id = result["datapoint"]["id"]
        evaluated_ids.append(datapoint_id)

    # Verify correct number of results
    assert len(evaluated_ids) == num_to_select, f"Expected {num_to_select} results, got {len(evaluated_ids)}"

    # Verify all evaluated IDs are in the selected set
    assert set(evaluated_ids) == set(selected_ids), (
        f"Evaluated IDs {evaluated_ids} don't match selected IDs {selected_ids}"
    )


@pytest.mark.asyncio
async def test_async_run_evaluation_with_datapoint_ids(
    evaluation_datasets: Dict[str, str],
    embedded_async_client: AsyncTensorZeroGateway,
):
    """Test async client experimental_run_evaluation with specific datapoint_ids.

    This is a happy path test that:
    1. Runs an evaluation to collect available datapoint IDs
    2. Selects a subset of those IDs
    3. Runs evaluation with only the selected IDs
    4. Verifies only the selected datapoints were evaluated
    """
    # First, run evaluation to collect available datapoint IDs from the dataset
    first_job = await embedded_async_client.experimental_run_evaluation(
        evaluation_name="haiku_without_outputs",
        dataset_name=evaluation_datasets["good-haikus-no-output"],
        variant_name="gpt_4o_mini",
        concurrency=2,
        inference_cache="off",
    )

    # Collect all datapoint IDs from the first run
    all_datapoint_ids: list[str] = []
    async for result in first_job.results():
        if result["type"] == "success":
            datapoint_id = result["datapoint"]["id"]
            all_datapoint_ids.append(datapoint_id)

    # We should have at least some datapoints
    assert len(all_datapoint_ids) > 0, "Dataset should contain datapoints"

    # Select first 3 datapoint IDs (or fewer if dataset is smaller)
    num_to_select = min(3, len(all_datapoint_ids))
    selected_ids = all_datapoint_ids[:num_to_select]

    # Run evaluation with only the selected datapoint IDs
    second_job = await embedded_async_client.experimental_run_evaluation(
        evaluation_name="haiku_without_outputs",
        datapoint_ids=selected_ids,
        variant_name="gpt_4o_mini",
        concurrency=2,
        inference_cache="off",
    )

    # Verify run_info reports correct number of datapoints
    run_info: Dict[str, Any] = second_job.run_info
    assert run_info["num_datapoints"] == num_to_select, (
        f"Expected {num_to_select} datapoints, got {run_info['num_datapoints']}"
    )

    # Collect results and verify they match selected IDs
    evaluated_ids: list[str] = []
    async for result in second_job.results():
        assert result["type"] == "success", "All evaluations should succeed"
        datapoint_id = result["datapoint"]["id"]
        evaluated_ids.append(datapoint_id)

    # Verify correct number of results
    assert len(evaluated_ids) == num_to_select, f"Expected {num_to_select} results, got {len(evaluated_ids)}"

    # Verify all evaluated IDs are in the selected set
    assert set(evaluated_ids) == set(selected_ids), (
        f"Evaluated IDs {evaluated_ids} don't match selected IDs {selected_ids}"
    )


def test_sync_run_evaluation_both_dataset_and_datapoint_ids_error(
    embedded_sync_client: TensorZeroGateway,
):
    """Test sync client rejects both dataset_name and datapoint_ids."""
    with pytest.raises(RuntimeError, match="Cannot provide both"):
        embedded_sync_client.experimental_run_evaluation(
            evaluation_name="entity_extraction",
            dataset_name="some_dataset",
            datapoint_ids=["01957bbb-44a8-7490-bfe7-32f8ed2fc797"],
            variant_name="gpt_4o_mini",
            concurrency=1,
            inference_cache="on",
        )


@pytest.mark.asyncio
async def test_async_run_evaluation_both_dataset_and_datapoint_ids_error(
    embedded_async_client: AsyncTensorZeroGateway,
):
    """Test async client rejects both dataset_name and datapoint_ids."""
    with pytest.raises(RuntimeError, match="Cannot provide both"):
        await embedded_async_client.experimental_run_evaluation(
            evaluation_name="entity_extraction",
            dataset_name="some_dataset",
            datapoint_ids=["01957bbb-44a8-7490-bfe7-32f8ed2fc797"],
            variant_name="gpt_4o_mini",
            concurrency=1,
            inference_cache="on",
        )


def test_sync_run_evaluation_neither_dataset_nor_datapoint_ids_error(
    embedded_sync_client: TensorZeroGateway,
):
    """Test sync client rejects neither dataset_name nor datapoint_ids."""
    with pytest.raises(RuntimeError, match="Must provide either"):
        embedded_sync_client.experimental_run_evaluation(
            evaluation_name="entity_extraction",
            variant_name="gpt_4o_mini",
            concurrency=1,
            inference_cache="on",
        )


@pytest.mark.asyncio
async def test_async_run_evaluation_neither_dataset_nor_datapoint_ids_error(
    embedded_async_client: AsyncTensorZeroGateway,
):
    """Test async client rejects neither dataset_name nor datapoint_ids."""
    with pytest.raises(RuntimeError, match="Must provide either"):
        await embedded_async_client.experimental_run_evaluation(
            evaluation_name="entity_extraction",
            variant_name="gpt_4o_mini",
            concurrency=1,
            inference_cache="on",
        )


def test_sync_run_evaluation_datapoint_ids_and_max_datapoints_error(
    embedded_sync_client: TensorZeroGateway,
):
    """Test sync client rejects both datapoint_ids and max_datapoints."""
    with pytest.raises(RuntimeError, match="Cannot provide both datapoint_ids and max_datapoints"):
        embedded_sync_client.experimental_run_evaluation(
            evaluation_name="entity_extraction",
            datapoint_ids=["01957bbb-44a8-7490-bfe7-32f8ed2fc797"],
            variant_name="gpt_4o_mini",
            concurrency=1,
            inference_cache="on",
            max_datapoints=10,
        )


@pytest.mark.asyncio
async def test_async_run_evaluation_datapoint_ids_and_max_datapoints_error(
    embedded_async_client: AsyncTensorZeroGateway,
):
    """Test async client rejects both datapoint_ids and max_datapoints."""
    with pytest.raises(RuntimeError, match="Cannot provide both datapoint_ids and max_datapoints"):
        await embedded_async_client.experimental_run_evaluation(
            evaluation_name="entity_extraction",
            datapoint_ids=["01957bbb-44a8-7490-bfe7-32f8ed2fc797"],
            variant_name="gpt_4o_mini",
            concurrency=1,
            inference_cache="on",
            max_datapoints=10,
        )
