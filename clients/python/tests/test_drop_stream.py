import gc
import inspect
import typing as t

import pytest
from pytest import CaptureFixture
from tensorzero import (
    AsyncTensorZeroGateway,
    TensorZeroGateway,
)
from tensorzero.types import ChatChunk, TextChunk


def test_drop_sync_stream_with_completion(capfd: CaptureFixture[str]):
    embedded_sync_client = TensorZeroGateway.build_embedded()
    stream = embedded_sync_client.inference(
        model_name="dummy::good",
        input={
            "messages": [
                {"role": "user", "content": "Hello, world!"},
            ],
        },
        dryrun=True,
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
    assert "iterated" not in captured.out
    assert captured.err == ""


def test_drop_sync_stream_without_completion(capfd: CaptureFixture[str]):
    embedded_sync_client = TensorZeroGateway.build_embedded()
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
async def test_drop_async_stream_without_completion(capfd: CaptureFixture[str]):
    client_fut = AsyncTensorZeroGateway.build_embedded()
    assert inspect.isawaitable(client_fut)
    embedded_async_client = await client_fut
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
async def test_drop_async_stream_with_completion(capfd: CaptureFixture[str]):
    client_fut = AsyncTensorZeroGateway.build_embedded()
    assert inspect.isawaitable(client_fut)
    embedded_async_client = await client_fut
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
    assert "iterated" not in captured.out
    assert captured.err == ""
