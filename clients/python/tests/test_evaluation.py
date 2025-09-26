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


def test_sync_run_evaluation_jsonl():
    """Test sync client experimental_run_evaluation with jsonl output format."""
    with TensorZeroGateway.build_embedded(
        config_file=TEST_CONFIG_FILE,
        clickhouse_url=CLICKHOUSE_URL,
    ) as client:
        client.experimental_run_evaluation(
            evaluation_name="entity_extraction",
            dataset_name="extract_entities_0.8",
            variant_name="gpt_4o_mini",
            concurrency=2,
            output_format="jsonl",
            inference_cache="on",
        )


def test_sync_run_evaluation_pretty():
    """Test sync client experimental_run_evaluation with pretty output format."""
    with TensorZeroGateway.build_embedded(
        config_file=TEST_CONFIG_FILE,
        clickhouse_url=CLICKHOUSE_URL,
    ) as client:
        client.experimental_run_evaluation(
            evaluation_name="haiku_without_outputs",
            dataset_name="good-haikus-no-output",
            variant_name="gpt_4o_mini",
            concurrency=2,
            output_format="pretty",
            inference_cache="off",
        )


def test_sync_run_evaluation_invalid_output_format():
    """Test sync client experimental_run_evaluation with invalid output format."""
    with TensorZeroGateway.build_embedded(
        config_file=TEST_CONFIG_FILE,
        clickhouse_url=CLICKHOUSE_URL,
    ) as client:
        with pytest.raises(ValueError, match="Invalid output_format"):
            client.experimental_run_evaluation(
                evaluation_name="entity_extraction",
                dataset_name="extract_entities_0.8",
                variant_name="gpt_4o_mini",
                concurrency=1,
                output_format="invalid_format",
                inference_cache="on",
            )


@pytest.mark.asyncio
async def test_async_run_evaluation_jsonl():
    """Test async client experimental_run_evaluation with jsonl output format."""
    async with await AsyncTensorZeroGateway.build_embedded(
        config_file=TEST_CONFIG_FILE,
        clickhouse_url=CLICKHOUSE_URL,
    ) as client:
        await client.experimental_run_evaluation(
            evaluation_name="entity_extraction",
            dataset_name="extract_entities_0.8",
            variant_name="gpt_4o_mini",
            concurrency=2,
            output_format="jsonl",
            inference_cache="on",
        )


@pytest.mark.asyncio
async def test_async_run_evaluation_pretty():
    """Test async client experimental_run_evaluation with pretty output format."""
    async with await AsyncTensorZeroGateway.build_embedded(
        config_file=TEST_CONFIG_FILE,
        clickhouse_url=CLICKHOUSE_URL,
    ) as client:
        await client.experimental_run_evaluation(
            evaluation_name="haiku_without_outputs",
            dataset_name="good-haikus-no-output",
            variant_name="gpt_4o_mini",
            concurrency=2,
            output_format="pretty",
            inference_cache="off",
        )


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
                output_format="jsonl",
                inference_cache="invalid_mode",
            )
