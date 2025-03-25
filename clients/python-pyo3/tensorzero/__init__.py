import typing as t
from importlib.metadata import version

import httpx

from .client import AsyncTensorZeroGateway, BaseTensorZeroGateway, TensorZeroGateway
from .tensorzero import _start_http_gateway as _start_http_gateway
from .types import (
    BaseTensorZeroError,
    ChatInferenceResponse,
    ContentBlock,
    FeedbackResponse,
    FinishReason,
    ImageBase64,
    ImageUrl,
    InferenceChunk,
    InferenceInput,
    InferenceResponse,
    JsonInferenceOutput,
    JsonInferenceResponse,
    RawText,
    TensorZeroError,
    TensorZeroInternalError,
    Text,
    TextChunk,
    ThoughtChunk,
    ToolCall,
    ToolCallChunk,
    ToolResult,
    Usage,
)

__all__ = [
    "AsyncTensorZeroGateway",
    "BaseTensorZeroGateway",
    "BaseTensorZeroError",
    "ChatInferenceResponse",
    "ContentBlock",
    "FeedbackResponse",
    "FinishReason",
    "InferenceChunk",
    "InferenceInput",
    "InferenceResponse",
    "JsonInferenceOutput",
    "JsonInferenceResponse",
    "ImageBase64",
    "ImageUrl",
    "RawText",
    "TensorZeroError",
    "TensorZeroInternalError",
    "TensorZeroGateway",
    "Text",
    "TextChunk",
    "ThoughtChunk",
    "ToolCall",
    "ToolCallChunk",
    "ToolResult",
    "Usage",
    "patch_openai_client",
]

T = t.TypeVar("T", bound=t.Any)

__version__ = version("tensorzero")


def _attach_fields(client: T, gateway: t.Any) -> T:
    if hasattr(client, "__tensorzero_gateway"):
        raise RuntimeError(
            "TensorZero: Already called 'tensorzero.patch_openai_client' on this OpenAI client."
        )
    client.base_url = gateway.base_url
    # Store the gateway so that it doesn't get garbage collected
    client.__tensorzero_gateway = gateway
    return client


async def _async_attach_fields(client: T, awaitable: t.Awaitable[t.Any]) -> T:
    gateway = await awaitable
    return _attach_fields(client, gateway)


class ATTENTION_TENSORZERO_PLEASE_AWAIT_RESULT_OF_PATCH_OPENAI_CLIENT(httpx.URL):
    # This is called by httpx when making a request (to join the base url with the path)
    # We throw an error to try to produce a nicer message for the user
    def copy_with(self, *args: t.Any, **kwargs: t.Any):
        raise RuntimeError(
            "TensorZero: Please await the result of `tensorzero.patch_openai_client` before using the client."
        )


def close_patched_openai_client_gateway(client: t.Any) -> None:
    """
    Closes the TensorZero gateway associated with a patched OpenAI client from `tensorzero.patch_openai_client`
    After calling this function, the patched client becomes unusable

    :param client: The OpenAI client previously patched with `tensorzero.patch_openai_client`
    """
    if hasattr(client, "__tensorzero_gateway"):
        client.__tensorzero_gateway.close()
    else:
        raise ValueError(
            "TensorZero: Called 'close_patched_client_gateway' on an OpenAI client that was not patched with 'tensorzero.patch_openai_client'."
        )


def patch_openai_client(
    client: T,
    *,
    config_file: t.Optional[str] = None,
    clickhouse_url: t.Optional[str] = None,
    async_setup: bool = True,
) -> t.Union[T, t.Awaitable[T]]:
    """
    Starts a new TensorZero gateway, and patching the provided OpenAI client to use it

    :param client: The OpenAI client to patch. This can be an 'OpenAI' or 'AsyncOpenAI' client
    :param config_file: (Optional) The path to the TensorZero configuration file.
    :param clickhouse_url: (Optional) The URL of the ClickHouse database.
    :param async_setup: (Optional) If True, returns an Awaitable that resolves to the patched client once the gateway has started. If False, blocks until the gateway has started.

    :return: The patched OpenAI client, or an Awaitable that resolves to it, depending on the value of `async_setup`
    """
    # If the user passes `async_setup=True`, then they need to 'await' the result of this function for the base_url to set to the running gateway
    # (since we need to await the future for our tensorzero gateway to start up)
    # To prevent requests from getting sent to the real OpenAI server if the user forgets to `await`,
    # we set a fake 'base_url' immediately, which will prevent the client from working until the real 'base_url' is set.
    # This type is set up to (hopefully) produce a nicer error message for the user
    client.base_url = ATTENTION_TENSORZERO_PLEASE_AWAIT_RESULT_OF_PATCH_OPENAI_CLIENT(
        "http://ATTENTION_TENSORZERO_PLEASE_AWAIT_RESULT_OF_PATCH_OPENAI_CLIENT.invalid/"
    )
    gateway = _start_http_gateway(
        config_file=config_file, clickhouse_url=clickhouse_url, async_setup=async_setup
    )
    if async_setup:
        # In 'async_setup' mode, return a `Future` that sets the needed fields after the gateway has started
        return _async_attach_fields(client, gateway)
    return _attach_fields(client, gateway)
