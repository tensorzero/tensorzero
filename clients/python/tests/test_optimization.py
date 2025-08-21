from time import sleep
from typing import List

import pytest
from tensorzero import (
    AsyncTensorZeroGateway,
    FireworksSFTConfig,
    OpenAISFTConfig,
    OptimizationJobStatus,
    RenderedSample,
    TensorZeroGateway,
    TogetherSFTConfig,
)


def test_sync_openai_sft(
    embedded_sync_client: TensorZeroGateway,
    mixed_rendered_samples: List[RenderedSample],
):
    optimization_config = OpenAISFTConfig(
        model="gpt-4o-mini", api_base="http://localhost:3030/openai/"
    )
    optimization_job_handle = embedded_sync_client.experimental_launch_optimization(
        train_samples=mixed_rendered_samples,
        val_samples=None,
        optimization_config=optimization_config,
    )
    while True:
        job_info = embedded_sync_client.experimental_poll_optimization(
            job_handle=optimization_job_handle
        )
        if job_info.status == OptimizationJobStatus.Completed:
            break
        sleep(1)


def test_sync_fireworks_sft(
    embedded_sync_client: TensorZeroGateway,
    mixed_rendered_samples: List[RenderedSample],
):
    optimization_config = FireworksSFTConfig(
        model="gpt-4o-mini",
        api_base="http://localhost:3030/fireworks/",
        account_id="test",
        epochs=1,
    )
    optimization_job_handle = embedded_sync_client.experimental_launch_optimization(
        train_samples=mixed_rendered_samples,
        val_samples=None,
        optimization_config=optimization_config,
    )
    while True:
        job_info = embedded_sync_client.experimental_poll_optimization(
            job_handle=optimization_job_handle
        )
        if job_info.status == OptimizationJobStatus.Completed:
            break
        sleep(1)


def test_sync_together_sft(
    embedded_sync_client: TensorZeroGateway,
    mixed_rendered_samples: List[RenderedSample],
):
    optimization_config = TogetherSFTConfig(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
        api_base="http://localhost:3030/together/",
        n_epochs=1,
    )
    optimization_job_handle = embedded_sync_client.experimental_launch_optimization(
        train_samples=mixed_rendered_samples,
        val_samples=None,
        optimization_config=optimization_config,
    )
    while True:
        job_info = embedded_sync_client.experimental_poll_optimization(
            job_handle=optimization_job_handle
        )
        if job_info.status == OptimizationJobStatus.Completed:
            break
        sleep(1)


@pytest.mark.asyncio
async def test_async_openai_sft(
    embedded_async_client: AsyncTensorZeroGateway,
    mixed_rendered_samples: List[RenderedSample],
):
    optimization_config = OpenAISFTConfig(
        model="gpt-4o-mini", api_base="http://localhost:3030/openai/"
    )
    optimization_job_handle = (
        await embedded_async_client.experimental_launch_optimization(
            train_samples=mixed_rendered_samples,
            val_samples=None,
            optimization_config=optimization_config,
        )
    )
    while True:
        job_info = await embedded_async_client.experimental_poll_optimization(
            job_handle=optimization_job_handle
        )
        if job_info.status == OptimizationJobStatus.Completed:
            break


@pytest.mark.asyncio
async def test_async_fireworks_sft(
    embedded_async_client: AsyncTensorZeroGateway,
    mixed_rendered_samples: List[RenderedSample],
):
    optimization_config = FireworksSFTConfig(
        model="gpt-4o-mini",
        api_base="http://localhost:3030/fireworks/",
        account_id="test",
        epochs=1,
    )
    optimization_job_handle = (
        await embedded_async_client.experimental_launch_optimization(
            train_samples=mixed_rendered_samples,
            val_samples=None,
            optimization_config=optimization_config,
        )
    )
    while True:
        job_info = await embedded_async_client.experimental_poll_optimization(
            job_handle=optimization_job_handle
        )
        if job_info.status == OptimizationJobStatus.Completed:
            break
        sleep(1)


@pytest.mark.asyncio
async def test_async_together_sft(
    embedded_async_client: AsyncTensorZeroGateway,
    mixed_rendered_samples: List[RenderedSample],
):
    optimization_config = TogetherSFTConfig(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
        api_base="http://localhost:3030/together/",
        n_epochs=1,
    )
    optimization_job_handle = (
        await embedded_async_client.experimental_launch_optimization(
            train_samples=mixed_rendered_samples,
            val_samples=None,
            optimization_config=optimization_config,
        )
    )
    while True:
        job_info = await embedded_async_client.experimental_poll_optimization(
            job_handle=optimization_job_handle
        )
        if job_info.status == OptimizationJobStatus.Completed:
            break
        sleep(1)
