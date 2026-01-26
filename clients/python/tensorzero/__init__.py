import typing as t
import warnings
from importlib.metadata import version

import httpx
from typing_extensions import Any, deprecated

from .client import (
    AsyncTensorZeroGateway,
    BaseTensorZeroGateway,
    TensorZeroGateway,
)

# Generated dataclasses
from .generated_types import (
    AlwaysExtraBody,
    AlwaysExtraBodyDelete,
    AlwaysExtraHeader,
    AlwaysExtraHeaderDelete,
    ChatCompletionInferenceParams,
    ContentBlockChatOutput,
    ContentBlockChatOutputText,
    ContentBlockChatOutputToolCall,
    CreateDatapointRequest,
    CreateDatapointRequestChat,
    CreateDatapointRequestJson,
    CreateDatapointsFromInferenceRequestParamsInferenceIds,
    CreateDatapointsResponse,
    DatapointMetadataUpdate,
    DeleteDatapointsResponse,
    ExtraBody,
    ExtraHeader,
    FunctionTool,
    GetDatapointsResponse,
    GetInferencesRequest,
    GetInferencesResponse,
    InferenceFilter,
    InferenceFilterAnd,
    InferenceFilterBooleanMetric,
    InferenceFilterDemonstrationFeedback,
    InferenceFilterFloatMetric,
    InferenceFilterNot,
    InferenceFilterOr,
    InferenceFilterTag,
    InferenceFilterTime,
    InferenceParams,
    Input,
    InputMessage,
    InputMessageContentTemplate,
    InputMessageContentText,
    JsonDatapointOutputUpdate,
    JsonInferenceOutput,
    ListDatapointsRequest,
    ListInferencesRequest,
    ModelProviderExtraBody,
    ModelProviderExtraBodyDelete,
    ModelProviderExtraHeader,
    ModelProviderExtraHeaderDelete,
    ProviderExtraBody,  # DEPRECATED
    ProviderExtraBodyDelete,  # DEPRECATED
    ProviderExtraHeader,  # DEPRECATED
    ProviderExtraHeaderDelete,  # DEPRECATED
    StorageKind,
    StorageKindDisabled,
    StorageKindFilesystem,
    StorageKindS3Compatible,
    StoragePath,
    StoredInference,
    StoredInferenceChat,
    StoredInferenceJson,
    StoredInput,
    StoredInputMessage,
    StoredInputMessageContentFile,
    StoredInputMessageContentTemplate,
    StoredInputMessageContentText,
    StoredInputMessageContentThought,
    StoredInputMessageContentToolCall,
    StoredInputMessageContentToolResult,
    StoredInputMessageContentUnknown,
    UpdateDatapointMetadataRequest,
    UpdateDatapointsResponse,
    VariantExtraBody,
    VariantExtraBodyDelete,
    VariantExtraHeader,
    VariantExtraHeaderDelete,
)
from .tensorzero import (
    BestOfNSamplingConfig,
    ChainOfThoughtConfig,
    ChatCompletionConfig,
    Config,
    DICLConfig,
    DICLOptimizationConfig,
    FireworksSFTConfig,
    FunctionConfigChat,
    FunctionConfigJson,
    FunctionsConfig,
    GCPVertexGeminiSFTConfig,
    GEPAConfig,
    LegacyDatapoint,
    MixtureOfNConfig,
    OpenAIRFTConfig,
    OpenAISFTConfig,
    OptimizationJobHandle,
    OptimizationJobInfo,
    OptimizationJobStatus,
    RenderedSample,
    ResolvedInput,
    ResolvedInputMessage,
    TogetherSFTConfig,
    VariantsConfig,
)
from .tensorzero import (
    _start_http_gateway as _start_http_gateway,
)
from .types import (
    AndFilter,  # pyright: ignore[reportDeprecated]
    ApiType,
    BaseTensorZeroError,
    BooleanMetricFilter,  # pyright: ignore[reportDeprecated]
    ChatDatapointInsert,
    ChatInferenceResponse,
    ContentBlock,
    DynamicEvaluationRunEpisodeResponse,  # DEPRECATED
    DynamicEvaluationRunResponse,  # DEPRECATED
    EvaluatorStatsDict,
    FeedbackResponse,
    FileBase64,
    FileUrl,
    FinishReason,
    FloatMetricFilter,  # pyright: ignore[reportDeprecated]
    ImageBase64,
    ImageUrl,
    InferenceChunk,
    InferenceInput,
    InferenceResponse,
    JsonDatapointInsert,
    JsonInferenceResponse,
    Message,
    NotFilter,  # pyright: ignore[reportDeprecated]
    OrderBy,
    OrFilter,  # pyright: ignore[reportDeprecated]
    RawResponseEntry,
    RawText,
    RawUsageEntry,
    System,
    TagFilter,  # pyright: ignore[reportDeprecated]
    Template,
    TensorZeroError,
    TensorZeroInternalError,
    Text,
    TextChunk,
    Thought,
    ThoughtChunk,
    TimeFilter,  # pyright: ignore[reportDeprecated]
    Tool,
    ToolCall,
    ToolCallChunk,
    ToolChoice,
    ToolParams,
    ToolResult,
    UnknownContentBlock,
    Usage,
    WorkflowEvaluationRunEpisodeResponse,
    WorkflowEvaluationRunResponse,
)

# DEPRECATED: use RenderedSample instead
RenderedStoredInference = RenderedSample
# Type aliases to preserve backward compatibility with main
ChatDatapoint = LegacyDatapoint.Chat
JsonDatapoint = LegacyDatapoint.Json


# CAREFUL: deprecated
class DiclOptimizationConfig:
    def __new__(cls, *args: Any, **kwargs: Any):
        warnings.warn(
            "Please use `DICLOptimizationConfig` instead of `DiclOptimizationConfig`. In a future release, `DiclOptimizationConfig` will be removed.",
            DeprecationWarning,
            stacklevel=2,
        )
        return DICLOptimizationConfig(*args, **kwargs)


# CAREFUL: deprecated alias
DiclConfig = deprecated("Use DICLConfig instead")(DICLConfig)


OptimizationConfig = t.Union[
    OpenAISFTConfig,
    FireworksSFTConfig,
    GCPVertexGeminiSFTConfig,
    TogetherSFTConfig,
    DICLOptimizationConfig,
    OpenAIRFTConfig,
    GEPAConfig,
    t.Dict[str, Any],
]
ChatInferenceOutput = t.List[ContentBlock]


__all__ = [
    "AlwaysExtraBody",
    "AlwaysExtraBodyDelete",
    "AlwaysExtraHeader",
    "AlwaysExtraHeaderDelete",
    "AndFilter",
    "ApiType",
    "AsyncTensorZeroGateway",
    "BaseTensorZeroError",
    "BaseTensorZeroGateway",
    "BestOfNSamplingConfig",
    "BooleanMetricFilter",
    "ChainOfThoughtConfig",
    "ChatCompletionConfig",
    "ChatCompletionInferenceParams",
    "ChatDatapoint",
    "ChatDatapointInsert",
    "ChatInferenceResponse",
    "Config",
    "ContentBlock",
    "ContentBlockChatOutput",
    "ContentBlockChatOutputText",
    "ContentBlockChatOutputToolCall",
    "CreateDatapointRequest",
    "CreateDatapointRequestChat",
    "CreateDatapointRequestJson",
    "CreateDatapointsFromInferenceRequestParamsInferenceIds",
    "CreateDatapointsResponse",
    "DatapointMetadataUpdate",
    "DeleteDatapointsResponse",
    "DICLConfig",
    "DiclConfig",  # DEPRECATED
    "DICLOptimizationConfig",
    "DiclOptimizationConfig",  # DEPRECATED
    "DynamicEvaluationRunEpisodeResponse",  # DEPRECATED
    "DynamicEvaluationRunResponse",  # DEPRECATED
    "EvaluatorStatsDict",
    "ExtraBody",
    "ExtraHeader",
    "FeedbackResponse",
    "FileBase64",
    "FileUrl",
    "FinishReason",
    "FireworksSFTConfig",
    "FloatMetricFilter",
    "FunctionConfigChat",
    "FunctionConfigJson",
    "FunctionsConfig",
    "FunctionTool",
    "GCPVertexGeminiSFTConfig",
    "GEPAConfig",
    "GetDatapointsResponse",
    "GetInferencesRequest",
    "GetInferencesResponse",
    "ImageBase64",
    "ImageUrl",
    "InferenceChunk",
    "InferenceFilter",
    "InferenceFilterAnd",
    "InferenceFilterBooleanMetric",
    "InferenceFilterDemonstrationFeedback",
    "InferenceFilterFloatMetric",
    "InferenceFilterNot",
    "InferenceFilterOr",
    "InferenceFilterTag",
    "InferenceFilterTime",
    "InferenceInput",
    "InferenceParams",
    "InferenceResponse",
    "Input",
    "InputMessage",
    "InputMessageContentTemplate",
    "InputMessageContentText",
    "JsonDatapoint",
    "JsonDatapointInsert",
    "JsonDatapointOutputUpdate",
    "JsonInferenceOutput",
    "JsonInferenceResponse",
    "LegacyDatapoint",
    "ListDatapointsRequest",
    "ListInferencesRequest",
    "Message",
    "MixtureOfNConfig",
    "ModelProviderExtraBody",
    "ModelProviderExtraBodyDelete",
    "ModelProviderExtraHeader",
    "ModelProviderExtraHeaderDelete",
    "NotFilter",
    "OpenAIRFTConfig",
    "OpenAISFTConfig",
    "OptimizationConfig",
    "OptimizationJobHandle",
    "OptimizationJobInfo",
    "OptimizationJobStatus",
    "OrderBy",
    "OrFilter",
    "patch_openai_client",
    "ProviderExtraBody",  # DEPRECATED
    "ProviderExtraBodyDelete",  # DEPRECATED
    "ProviderExtraHeader",  # DEPRECATED
    "ProviderExtraHeaderDelete",  # DEPRECATED
    "RawResponseEntry",
    "RawText",
    "RawUsageEntry",
    "RenderedSample",
    "RenderedStoredInference",  # DEPRECATED
    "ResolvedInput",
    "ResolvedInputMessage",
    "StorageKind",
    "StorageKindDisabled",
    "StorageKindFilesystem",
    "StorageKindS3Compatible",
    "StoragePath",
    "StoredInference",
    "StoredInferenceChat",
    "StoredInferenceJson",
    "StoredInput",
    "StoredInputMessage",
    "StoredInputMessageContentFile",
    "StoredInputMessageContentTemplate",
    "StoredInputMessageContentText",
    "StoredInputMessageContentThought",
    "StoredInputMessageContentToolCall",
    "StoredInputMessageContentToolResult",
    "StoredInputMessageContentUnknown",
    "System",
    "TagFilter",
    "Template",
    "TensorZeroError",
    "TensorZeroGateway",
    "TensorZeroInternalError",
    "Text",
    "TextChunk",
    "Thought",
    "ThoughtChunk",
    "TimeFilter",
    "TogetherSFTConfig",
    "Tool",
    "ToolCall",
    "ToolCallChunk",
    "ToolChoice",
    "ToolParams",
    "ToolResult",
    "UnknownContentBlock",
    "UpdateDatapointMetadataRequest",
    "UpdateDatapointsResponse",
    "Usage",
    "VariantExtraBody",
    "VariantExtraBodyDelete",
    "VariantExtraHeader",
    "VariantExtraHeaderDelete",
    "VariantsConfig",
    "WorkflowEvaluationRunEpisodeResponse",
    "WorkflowEvaluationRunResponse",
]

T = t.TypeVar("T", bound=t.Any)

__version__ = version("tensorzero")


def _attach_fields(client: T, gateway: t.Any) -> T:
    if hasattr(client, "__tensorzero_gateway"):
        raise RuntimeError("TensorZero: Already called 'tensorzero.patch_openai_client' on this OpenAI client.")
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
        config_file=config_file,
        clickhouse_url=clickhouse_url,
        postgres_url=None,
        valkey_url=None,
        async_setup=async_setup,
    )
    if async_setup:
        # In 'async_setup' mode, return a `Future` that sets the needed fields after the gateway has started
        return _async_attach_fields(client, gateway)
    return _attach_fields(client, gateway)
