from time import sleep
from typing import List

import pytest
from tensorzero import (
    AsyncTensorZeroGateway,
    DICLOptimizationConfig,
    FireworksSFTConfig,
    GEPAConfig,
    OpenAIRFTConfig,
    OpenAISFTConfig,
    OptimizationJobStatus,
    RenderedSample,
    TensorZeroGateway,
    TogetherSFTConfig,
)
from uuid_utils import uuid7


@pytest.mark.mock
def test_sync_openai_rft(
    embedded_sync_client: TensorZeroGateway,
    mixed_rendered_samples: List[RenderedSample],
):
    grader = {
        "type": "multi",
        "name": "test_grader",
        "graders": {
            "string_check_grader": {
                "type": "string_check",
                "name": "string_check_grader",
                "operation": "eq",
                "input": "{{sample.output_text}}",
                "reference": "{{item.reference_text}}",
            },
            "score_model_grader": {
                "type": "score_model",
                "name": "score_model_grader",
                "model": "gpt-4.1-nano-2025-04-14",
                "input": [
                    {
                        "role": "developer",
                        "content": "You are an expert grader. Score the following response on a scale of 0 to 1.",
                    },
                    {
                        "role": "user",
                        "content": "Reference Text:\n{{item.reference_text}}\n\nResponse Text:\n{{sample.output_text}}\n\nReference Tool Calls:\n{{item.reference_tools}}\n\nResponse Tool Calls:\n{{sample.output_tools}}",
                    },
                ],
                "range": [0.0, 1.0],
            },
        },
        "calculate_output": "0.5 * string_check_grader + 0.5 * score_model_grader",
    }
    optimization_config = OpenAIRFTConfig(
        model="o4-mini-2025-04-16",
        grader=grader,
        n_epochs=1,
        reasoning_effort="low",
    )
    optimization_job_handle = embedded_sync_client.experimental_launch_optimization(
        train_samples=mixed_rendered_samples,
        val_samples=mixed_rendered_samples,
        optimization_config=optimization_config,
    )
    while True:
        job_info = embedded_sync_client.experimental_poll_optimization(job_handle=optimization_job_handle)
        if job_info.status == OptimizationJobStatus.Completed:
            break
        sleep(1)


@pytest.mark.mock
def test_sync_dicl_chat(
    embedded_sync_client: TensorZeroGateway,
    chat_function_rendered_samples: List[RenderedSample],
):
    optimization_config = {
        "type": "dicl",
        "embedding_model": "text-embedding-3-small",
        "variant_name": "test_dicl_chat",
        "function_name": "basic_test",
        "append_to_existing_variants": True,
    }
    optimization_job_handle = embedded_sync_client.experimental_launch_optimization(
        train_samples=chat_function_rendered_samples,
        val_samples=None,
        optimization_config=optimization_config,
    )
    while True:
        job_info = embedded_sync_client.experimental_poll_optimization(job_handle=optimization_job_handle)
        if job_info.status == OptimizationJobStatus.Completed:
            break
        sleep(1)


@pytest.mark.mock
def test_sync_dicl_json(
    embedded_sync_client: TensorZeroGateway,
    json_function_rendered_samples: List[RenderedSample],
):
    optimization_config = DICLOptimizationConfig(
        embedding_model="text-embedding-3-small",
        variant_name=f"test_dicl_json_{uuid7()}",
        function_name="json_success",
        dimensions=None,
        batch_size=None,
        max_concurrency=None,
        k=None,
        model=None,
    )
    optimization_job_handle = embedded_sync_client.experimental_launch_optimization(
        train_samples=json_function_rendered_samples,
        val_samples=None,
        optimization_config=optimization_config,
    )
    while True:
        job_info = embedded_sync_client.experimental_poll_optimization(job_handle=optimization_job_handle)
        if job_info.status == OptimizationJobStatus.Completed:
            break
        sleep(1)


@pytest.mark.mock
def test_sync_openai_sft(
    embedded_sync_client: TensorZeroGateway,
    mixed_rendered_samples: List[RenderedSample],
):
    optimization_config = {
        "type": "openai_sft",
        "model": "gpt-4o-mini",
    }
    optimization_job_handle = embedded_sync_client.experimental_launch_optimization(
        train_samples=mixed_rendered_samples,
        val_samples=None,
        optimization_config=optimization_config,
    )
    while True:
        job_info = embedded_sync_client.experimental_poll_optimization(job_handle=optimization_job_handle)
        if job_info.status == OptimizationJobStatus.Completed:
            break
        sleep(1)


@pytest.mark.mock
def test_sync_fireworks_sft(
    embedded_sync_client: TensorZeroGateway,
    mixed_rendered_samples: List[RenderedSample],
):
    optimization_config = FireworksSFTConfig(
        model="gpt-4o-mini",
        epochs=1,
    )
    optimization_job_handle = embedded_sync_client.experimental_launch_optimization(
        train_samples=mixed_rendered_samples,
        val_samples=None,
        optimization_config=optimization_config,
    )
    while True:
        job_info = embedded_sync_client.experimental_poll_optimization(job_handle=optimization_job_handle)
        if job_info.status == OptimizationJobStatus.Completed:
            break
        sleep(1)


@pytest.mark.mock
def test_sync_together_sft(
    embedded_sync_client: TensorZeroGateway,
    mixed_rendered_samples: List[RenderedSample],
):
    optimization_config = {
        "type": "together_sft",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
        "n_epochs": 1,
        "training_type": {"type": "Lora", "lora_r": 8, "lora_alpha": 16},
        "batch_size": "max",
    }
    optimization_job_handle = embedded_sync_client.experimental_launch_optimization(
        train_samples=mixed_rendered_samples,
        val_samples=None,
        optimization_config=optimization_config,
    )
    while True:
        job_info = embedded_sync_client.experimental_poll_optimization(job_handle=optimization_job_handle)
        if job_info.status == OptimizationJobStatus.Completed:
            break
        sleep(1)


@pytest.mark.mock
def test_sync_gepa_chat(
    embedded_sync_client: TensorZeroGateway,
    chat_function_rendered_samples: List[RenderedSample],
):
    optimization_config = GEPAConfig(
        function_name="basic_test",
        evaluation_name="test_evaluation",
        analysis_model="openai::gpt-4o-mini",
        mutation_model="openai::gpt-4o-mini",
        initial_variants=["anthropic"],
    )

    optimization_job_handle = embedded_sync_client.experimental_launch_optimization(
        train_samples=chat_function_rendered_samples,
        val_samples=chat_function_rendered_samples,
        optimization_config=optimization_config,
    )
    while True:
        job_info = embedded_sync_client.experimental_poll_optimization(job_handle=optimization_job_handle)
        if job_info.status == OptimizationJobStatus.Completed:
            break
        sleep(1)


@pytest.mark.mock
@pytest.mark.asyncio
async def test_async_openai_rft(
    embedded_async_client: AsyncTensorZeroGateway,
    mixed_rendered_samples: List[RenderedSample],
):
    grader = {
        "type": "multi",
        "name": "test_grader",
        "graders": {
            "string_check_grader": {
                "type": "string_check",
                "name": "string_check_grader",
                "operation": "eq",
                "input": "{{sample.output_text}}",
                "reference": "{{item.reference_text}}",
            },
            "score_model_grader": {
                "type": "score_model",
                "name": "score_model_grader",
                "model": "gpt-4.1-nano-2025-04-14",
                "input": [
                    {
                        "role": "developer",
                        "content": "You are an expert grader. Score the following response on a scale of 0 to 1.",
                    },
                    {
                        "role": "user",
                        "content": "Reference Text:\n{{item.reference_text}}\n\nResponse Text:\n{{sample.output_text}}\n\nReference Tool Calls:\n{{item.reference_tools}}\n\nResponse Tool Calls:\n{{sample.output_tools}}",
                    },
                ],
                "range": [0.0, 1.0],
            },
        },
        "calculate_output": "0.5 * string_check_grader + 0.5 * score_model_grader",
    }
    optimization_config = {
        "type": "openai_rft",
        "model": "o4-mini-2025-04-16",
        "grader": grader,
        "n_epochs": 1,
        "reasoning_effort": "low",
    }
    optimization_job_handle = await embedded_async_client.experimental_launch_optimization(
        train_samples=mixed_rendered_samples,
        val_samples=mixed_rendered_samples,
        optimization_config=optimization_config,
    )
    while True:
        job_info = await embedded_async_client.experimental_poll_optimization(job_handle=optimization_job_handle)
        if job_info.status == OptimizationJobStatus.Completed:
            break
        sleep(1)


@pytest.mark.mock
@pytest.mark.asyncio
async def test_async_dicl_chat(
    embedded_async_client: AsyncTensorZeroGateway,
    chat_function_rendered_samples: List[RenderedSample],
):
    optimization_config = DICLOptimizationConfig(
        embedding_model="text-embedding-3-small",
        variant_name="test_dicl_chat",
        function_name="basic_test",
        dimensions=None,
        batch_size=None,
        max_concurrency=None,
        k=None,
        model=None,
        append_to_existing_variants=True,
    )
    optimization_job_handle = await embedded_async_client.experimental_launch_optimization(
        train_samples=chat_function_rendered_samples,
        val_samples=None,
        optimization_config=optimization_config,
    )
    while True:
        job_info = await embedded_async_client.experimental_poll_optimization(job_handle=optimization_job_handle)
        if job_info.status == OptimizationJobStatus.Completed:
            break
        sleep(1)


@pytest.mark.mock
@pytest.mark.asyncio
async def test_async_dicl_json(
    embedded_async_client: AsyncTensorZeroGateway,
    json_function_rendered_samples: List[RenderedSample],
):
    optimization_config = {
        "type": "dicl",
        "embedding_model": "text-embedding-3-small",
        "variant_name": f"test_dicl_json_{uuid7()}",
        "function_name": "json_success",
    }
    optimization_job_handle = await embedded_async_client.experimental_launch_optimization(
        train_samples=json_function_rendered_samples,
        val_samples=None,
        optimization_config=optimization_config,
    )
    while True:
        job_info = await embedded_async_client.experimental_poll_optimization(job_handle=optimization_job_handle)
        if job_info.status == OptimizationJobStatus.Completed:
            break
        sleep(1)


@pytest.mark.mock
@pytest.mark.asyncio
async def test_async_openai_sft(
    embedded_async_client: AsyncTensorZeroGateway,
    mixed_rendered_samples: List[RenderedSample],
):
    optimization_config = OpenAISFTConfig(model="gpt-4o-mini")
    optimization_job_handle = await embedded_async_client.experimental_launch_optimization(
        train_samples=mixed_rendered_samples,
        val_samples=None,
        optimization_config=optimization_config,
    )
    while True:
        job_info = await embedded_async_client.experimental_poll_optimization(job_handle=optimization_job_handle)
        if job_info.status == OptimizationJobStatus.Completed:
            break


@pytest.mark.mock
@pytest.mark.asyncio
async def test_async_fireworks_sft(
    embedded_async_client: AsyncTensorZeroGateway,
    mixed_rendered_samples: List[RenderedSample],
):
    optimization_config = {
        "type": "fireworks_sft",
        "model": "gpt-4o-mini",
        "epochs": 1,
    }
    optimization_job_handle = await embedded_async_client.experimental_launch_optimization(
        train_samples=mixed_rendered_samples,
        val_samples=None,
        optimization_config=optimization_config,
    )
    while True:
        job_info = await embedded_async_client.experimental_poll_optimization(job_handle=optimization_job_handle)
        if job_info.status == OptimizationJobStatus.Completed:
            break
        sleep(1)


@pytest.mark.mock
@pytest.mark.asyncio
async def test_async_together_sft(
    embedded_async_client: AsyncTensorZeroGateway,
    mixed_rendered_samples: List[RenderedSample],
):
    optimization_config = TogetherSFTConfig(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
        n_epochs=1,
        training_type={"type": "Lora", "lora_r": 8, "lora_alpha": 16},
        batch_size="max",
    )
    optimization_job_handle = await embedded_async_client.experimental_launch_optimization(
        train_samples=mixed_rendered_samples,
        val_samples=None,
        optimization_config=optimization_config,
    )
    while True:
        job_info = await embedded_async_client.experimental_poll_optimization(job_handle=optimization_job_handle)
        if job_info.status == OptimizationJobStatus.Completed:
            break
        sleep(1)


@pytest.mark.mock
@pytest.mark.asyncio
async def test_async_gepa_json(
    embedded_async_client: AsyncTensorZeroGateway,
    json_function_rendered_samples: List[RenderedSample],
):
    optimization_config = GEPAConfig(
        function_name="json_success",
        evaluation_name="json_evaluation",
        analysis_model="openai::gpt-4o-mini",
        mutation_model="openai::gpt-4o-mini",
        initial_variants=["anthropic", "openai"],
    )

    optimization_job_handle = await embedded_async_client.experimental_launch_optimization(
        train_samples=json_function_rendered_samples,
        val_samples=json_function_rendered_samples,
        optimization_config=optimization_config,
    )
    while True:
        job_info = await embedded_async_client.experimental_poll_optimization(job_handle=optimization_job_handle)
        if job_info.status == OptimizationJobStatus.Completed:
            break
        sleep(1)


# Error handling tests
def test_invalid_config_missing_type(
    embedded_sync_client: TensorZeroGateway,
    mixed_rendered_samples: List[RenderedSample],
):
    """Test that a dictionary without a 'type' field produces a helpful error message."""
    optimization_config = {
        "model": "gpt-4o-mini",
        "api_base": "http://localhost:3030/openai/",
    }
    with pytest.raises(Exception) as exc_info:
        embedded_sync_client.experimental_launch_optimization(
            train_samples=mixed_rendered_samples,
            val_samples=None,
            optimization_config=optimization_config,  # type: ignore
        )
    error_message = str(exc_info.value)
    assert "Invalid optimization config" in error_message
    assert "OpenAISFTConfig" in error_message or "type" in error_message.lower()


def test_invalid_config_wrong_type(
    embedded_sync_client: TensorZeroGateway,
    mixed_rendered_samples: List[RenderedSample],
):
    """Test that a dictionary with an invalid 'type' field produces a helpful error message."""
    optimization_config = {
        "type": "invalid_optimizer_type",
        "model": "gpt-4o-mini",
    }
    with pytest.raises(Exception) as exc_info:
        embedded_sync_client.experimental_launch_optimization(
            train_samples=mixed_rendered_samples,
            val_samples=None,
            optimization_config=optimization_config,  # type: ignore
        )
    error_message = str(exc_info.value)
    assert "Invalid optimization config" in error_message or "unknown variant" in error_message.lower()


def test_invalid_config_missing_required_field(
    embedded_sync_client: TensorZeroGateway,
    mixed_rendered_samples: List[RenderedSample],
):
    """Test that a dictionary missing required fields produces a helpful error message."""
    optimization_config = {
        "type": "openai_sft",
        # Missing required 'model' field
    }
    with pytest.raises(Exception) as exc_info:
        embedded_sync_client.experimental_launch_optimization(
            train_samples=mixed_rendered_samples,
            val_samples=None,
            optimization_config=optimization_config,  # type: ignore
        )
    error_message = str(exc_info.value)
    assert "model" in error_message.lower() or "missing field" in error_message.lower()


def test_invalid_config_wrong_object_type(
    embedded_sync_client: TensorZeroGateway,
    mixed_rendered_samples: List[RenderedSample],
):
    """Test that passing a completely wrong type produces a helpful error message."""
    with pytest.raises(Exception) as exc_info:
        embedded_sync_client.experimental_launch_optimization(
            train_samples=mixed_rendered_samples,
            val_samples=None,
            optimization_config="not_a_valid_config",  # type: ignore
        )
    error_message = str(exc_info.value)
    assert "Invalid optimization config" in error_message or "expected" in error_message.lower()
