from time import sleep
from typing import List

from tensorzero import (
    FireworksSFTConfig,
    OpenAISFTConfig,
    RenderedSample,
    TensorZeroGateway,
)


def test_sync_openai_sft(
    embedded_sync_client: TensorZeroGateway,
    mixed_rendered_samples: List[RenderedSample],
):
    optimizer_config = OpenAISFTConfig(
        model="gpt-4o-mini", api_base="http://localhost:3030/openai/"
    )
    optimizer_job_handle = embedded_sync_client.experimental_launch_optimization(
        train_examples=mixed_rendered_samples,
        val_examples=None,
        optimizer_config=optimizer_config,
    )
    while True:
        status = embedded_sync_client.experimental_poll_optimization(
            job_handle=optimizer_job_handle
        )
        if status.status == "completed":
            break
        sleep(1)


def test_sync_fireworks_sft(
    embedded_sync_client: TensorZeroGateway,
    mixed_rendered_samples: List[RenderedSample],
):
    optimizer_config = FireworksSFTConfig(
        model="gpt-4o-mini",
        api_base="http://localhost:3030/fireworks/",
        account_id="test",
    )
    optimizer_job_handle = embedded_sync_client.experimental_launch_optimization(
        train_examples=mixed_rendered_samples,
        val_examples=None,
        optimizer_config=optimizer_config,
    )
    while True:
        status = embedded_sync_client.experimental_poll_optimization(
            job_handle=optimizer_job_handle
        )
        if status.status == "completed":
            break
        sleep(1)
