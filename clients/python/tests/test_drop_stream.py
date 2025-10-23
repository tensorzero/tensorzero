import gc
import typing as t

import pytest
from pytest import CaptureFixture
from tensorzero import (
    AsyncTensorZeroGateway,
    TensorZeroGateway,
)
from tensorzero.types import ChatChunk, TextChunk


def test_drop_sync_stream_with_completion(embedded_sync_client: TensorZeroGateway, capfd: CaptureFixture[str]):
    stream = embedded_sync_client.inference(
        model_name="dummy::good",
        input={
            "messages": [
                {"role": "user", "content": "Hello, world!"},
            ],
        },
        stream=True,
    )
    # Consume the stream to completion,
    # and check that we don't log any warnings
    assert isinstance(stream, t.Iterator)
    chunks = list(stream)
    assert len(chunks) == 17
    del stream
    gc.collect()

    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_drop_sync_stream_without_completion(embedded_sync_client: TensorZeroGateway, capfd: CaptureFixture[str]):
    stream = embedded_sync_client.inference(
        model_name="dummy::good",
        input={
            "messages": [
                {"role": "user", "content": "Hello, world!"},
            ],
        },
        stream=True,
    )
    assert isinstance(stream, t.Iterator)
    first_chunk = next(stream)
    assert isinstance(first_chunk, ChatChunk)
    assert first_chunk.content == [TextChunk(id="0", text="Wally,")]
    del stream
    gc.collect()

    captured = capfd.readouterr()
    assert "Stream was garbage-collected without being iterated to completion" in captured.out
    assert captured.err == ""


@pytest.mark.asyncio
async def test_drop_async_stream_without_completion(
    embedded_async_client: AsyncTensorZeroGateway, capfd: CaptureFixture[str]
):
    stream = await embedded_async_client.inference(
        model_name="dummy::good",
        input={
            "messages": [
                {"role": "user", "content": "Hello, world!"},
            ],
        },
        stream=True,
    )
    assert isinstance(stream, t.AsyncIterator)
    first_chunk = await anext(stream)
    assert isinstance(first_chunk, ChatChunk)
    assert first_chunk.content == [TextChunk(id="0", text="Wally,")]
    del stream
    gc.collect()

    captured = capfd.readouterr()
    assert "Stream was garbage-collected without being iterated to completion" in captured.out
    assert captured.err == ""


@pytest.mark.asyncio
async def test_drop_async_stream_with_completion(
    embedded_async_client: AsyncTensorZeroGateway, capfd: CaptureFixture[str]
):
    stream = await embedded_async_client.inference(
        model_name="dummy::good",
        input={
            "messages": [
                {"role": "user", "content": "Hello, world!"},
            ],
        },
        stream=True,
    )
    assert isinstance(stream, t.AsyncIterator)
    chunks = [chunk async for chunk in stream]
    assert len(chunks) == 17
    del stream
    gc.collect()

    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""
