# Type stubs for tensorzero.tensorzero
# Auto-generated - DO NOT EDIT MANUALLY
# Run: uv run python generate_stubs.py > tensorzero/tensorzero.pyi

from typing import Any, Coroutine, Dict, List, Literal, Optional, Type, Union
from uuid import UUID
from typing_extensions import final

__all__ = [
    "BaseTensorZeroGateway",
    "AsyncTensorZeroGateway",
    "TensorZeroGateway",
    "LocalHttpGateway",
    "RenderedSample",
    "StoredInference",
    "EvaluationJobHandler",
    "AsyncEvaluationJobHandler",
    "OpenAIRFTConfig",
    "OpenAISFTConfig",
    "FireworksSFTConfig",
    "DICLOptimizationConfig",
    "GCPVertexGeminiSFTConfig",
    "TogetherSFTConfig",
    "Datapoint",
    "ResolvedInput",
    "ResolvedInputMessage",
    "CreateChatDatapointRequest",
    "CreateJsonDatapointRequest",
    "CreateDatapointsRequest",
    "CreateDatapointsResponse",
    "UpdateChatDatapointRequest",
    "UpdateJsonDatapointRequest",
    "UpdateDatapointsRequest",
    "UpdateDatapointsResponse",
    "UpdateDatapointMetadataRequest",
    "UpdateDatapointsMetadataRequest",
    "DatapointMetadataUpdate",
    "JsonDatapointOutputUpdate",
    "ListDatapointsRequest",
    "GetDatapointsRequest",
    "GetDatapointsResponse",
    "DeleteDatapointsRequest",
    "DeleteDatapointsResponse",
    "CreateDatapointsFromInferenceOutputSource",
    "CreateDatapointsFromInferenceRequest",
    "Config",
    "FunctionsConfig",
    "FunctionConfigChat",
    "FunctionConfigJson",
    "VariantsConfig",
    "ChatCompletionConfig",
    "BestOfNSamplingConfig",
    "DICLConfig",
    "MixtureOfNConfig",
    "ChainOfThoughtConfig",
    "OptimizationJobHandle",
    "OptimizationJobInfo",
    "OptimizationJobStatus",
    "_start_http_gateway",
]

# ============================================================================
# Dataset V1 API Types
# ============================================================================

@final
class JsonDatapointOutputUpdate:
    """Update for JSON datapoint output."""
    def __init__(self, raw: str, *args: Any) -> None: ...
    @property
    def raw(self) -> str: ...

@final
class DatapointMetadataUpdate:
    """Update for datapoint metadata."""
    def __init__(self, *args: Any, name: Optional[str], **kwargs: Any) -> None: ...
    @property
    def name(self) -> Optional[str]: ...

@final
class CreateChatDatapointRequest:
    """Request to create a chat datapoint."""
    def __init__(
        self,
        function_name: str,
        input: Dict[str, Any],
        episode_id: Optional[UUID] = None,
        output: Optional[Dict[str, Any]] = None,
        dynamic_tool_params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    @property
    def function_name(self) -> str: ...
    @property
    def episode_id(self) -> UUID: ...
    @property
    def input(self) -> Dict[str, Any]: ...
    @property
    def output(self) -> Dict[str, Any]: ...
    @property
    def dynamic_tool_params(self) -> Dict[str, Any]: ...
    @property
    def tags(self) -> Dict[str, str]: ...
    @property
    def name(self) -> str: ...

@final
class CreateJsonDatapointRequest:
    """Request to create a JSON datapoint."""
    def __init__(
        self,
        function_name: str,
        input: Dict[str, Any],
        episode_id: Optional[UUID] = None,
        output: Optional[JsonDatapointOutputUpdate] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    @property
    def function_name(self) -> str: ...
    @property
    def episode_id(self) -> UUID: ...
    @property
    def input(self) -> Dict[str, Any]: ...
    @property
    def output(self) -> Dict[str, Any]: ...
    @property
    def output_schema(self) -> Dict[str, Any]: ...
    @property
    def tags(self) -> Dict[str, str]: ...
    @property
    def name(self) -> str: ...

@final
class UpdateChatDatapointRequest:
    """Request to update a chat datapoint."""
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    @property
    def id(self) -> UUID: ...
    @property
    def input(self) -> Dict[str, Any]: ...
    @property
    def output(self) -> Dict[str, Any]: ...
    @property
    def tool_params(self) -> Dict[str, Any]: ...
    @property
    def tags(self) -> Dict[str, str]: ...
    @property
    def metadata(self) -> Dict[str, Any]: ...

@final
class UpdateJsonDatapointRequest:
    """Request to update a JSON datapoint."""
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    @property
    def id(self) -> UUID: ...
    @property
    def input(self) -> Dict[str, Any]: ...
    @property
    def output(self) -> Dict[str, Any]: ...
    @property
    def output_schema(self) -> Dict[str, Any]: ...
    @property
    def tags(self) -> Dict[str, str]: ...
    @property
    def metadata(self) -> Dict[str, Any]: ...

@final
class UpdateDatapointMetadataRequest:
    """Request to update datapoint metadata."""
    def __init__(self, id: UUID, metadata: DatapointMetadataUpdate, *args: Any, **kwargs: Any) -> None: ...
    @property
    def id(self) -> UUID: ...
    @property
    def metadata(self) -> DatapointMetadataUpdate: ...

@final
class CreateDatapointsRequest:
    """CreateDatapointsRequest"""
    def __init__(
        self, datapoints: List[Union[CreateChatDatapointRequest, CreateJsonDatapointRequest]], *args: Any, **kwargs: Any
    ) -> None: ...
    @property
    def datapoints(self) -> List[Union[CreateChatDatapointRequest, CreateJsonDatapointRequest]]: ...

@final
class UpdateDatapointsRequest:
    """UpdateDatapointsRequest"""
    def __init__(
        self, datapoints: List[Union[UpdateChatDatapointRequest, UpdateJsonDatapointRequest]], *args: Any, **kwargs: Any
    ) -> None: ...
    @property
    def datapoints(self) -> List[Union[UpdateChatDatapointRequest, UpdateJsonDatapointRequest]]: ...

@final
class UpdateDatapointsMetadataRequest:
    """UpdateDatapointsMetadataRequest"""
    def __init__(self, datapoints: List[UpdateDatapointMetadataRequest], *args: Any, **kwargs: Any) -> None: ...
    @property
    def datapoints(self) -> List[UpdateDatapointMetadataRequest]: ...

@final
class GetDatapointsRequest:
    """GetDatapointsRequest"""
    def __init__(self, ids: List[UUID], *args: Any, **kwargs: Any) -> None: ...
    @property
    def ids(self) -> List[UUID]: ...

@final
class DeleteDatapointsRequest:
    """DeleteDatapointsRequest"""
    def __init__(self, ids: List[UUID], *args: Any, **kwargs: Any) -> None: ...
    @property
    def ids(self) -> List[UUID]: ...

@final
class CreateDatapointsResponse:
    """Response from creating datapoints."""
    @property
    def ids(self) -> List[UUID]: ...

@final
class UpdateDatapointsResponse:
    """Response from updating datapoints."""
    @property
    def ids(self) -> List[UUID]: ...

@final
class GetDatapointsResponse:
    """Response containing retrieved datapoints."""
    @property
    def datapoints(self) -> List["Datapoint"]: ...

@final
class DeleteDatapointsResponse:
    """Response from deleting datapoints."""
    @property
    def num_deleted_datapoints(self) -> int: ...

@final
class ListDatapointsRequest:
    """Request to list datapoints."""
    def __init__(
        self,
        function_name: Optional[str] = None,
        limit: Optional[int] = None,
        page_size: Optional[int] = None,
        offset: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    @property
    def function_name(self) -> str: ...
    @property
    def limit(self) -> int: ...
    @property
    def page_size(self) -> int: ...
    @property
    def offset(self) -> int: ...
    @property
    def filter(self) -> Dict[str, Any]: ...

@final
class CreateDatapointsFromInferenceRequest:
    """Request to create datapoints from inferences."""
    def __init__(
        self,
        params: Dict[str, Any],
        output_source: Optional["CreateDatapointsFromInferenceOutputSource"] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    @property
    def params(self) -> Dict[str, Any]: ...
    @property
    def output_source(self) -> Optional["CreateDatapointsFromInferenceOutputSource"]: ...

@final
class CreateDatapointsFromInferenceOutputSource:
    """Output source for creating datapoints from inferences."""

    Demonstration: Type["CreateDatapointsFromInferenceOutputSource"]
    Inference: Type["CreateDatapointsFromInferenceOutputSource"]
    NoOutput: Type["CreateDatapointsFromInferenceOutputSource"]
    def __init__(self, *args: Any) -> None: ...

@final
class Datapoint:
    """A datapoint - tagged enum with Chat and Json variants."""

    Chat: Type["Datapoint"]
    Json: Type["Datapoint"]
    @property
    def additional_tools(self) -> List[Dict[str, Any]]: ...
    @property
    def allowed_tools(self) -> List[str]: ...
    @property
    def dataset_name(self) -> str: ...
    @property
    def function_name(self) -> str: ...
    @property
    def id(self) -> UUID: ...
    @property
    def input(self) -> Dict[str, Any]: ...
    @property
    def is_custom(self) -> bool: ...
    @property
    def name(self) -> str: ...
    @property
    def output(self) -> Dict[str, Any]: ...
    @property
    def output_schema(self) -> Dict[str, Any]: ...
    @property
    def parallel_tool_calls(self) -> bool: ...
    @property
    def provider_tools(self) -> List[Dict[str, Any]]: ...

# ============================================================================
# Configuration Types
# ============================================================================

@final
class Config:
    """Config"""
    @property
    def functions(self) -> FunctionsConfig: ...

@final
class FunctionsConfig:
    """FunctionsConfig"""

    ...

@final
class FunctionConfigChat:
    """FunctionConfigChat"""
    @property
    def assistant_schema(self) -> Dict[str, Any]: ...
    @property
    def system_schema(self) -> Dict[str, Any]: ...
    @property
    def type(self) -> str: ...
    @property
    def user_schema(self) -> Dict[str, Any]: ...
    @property
    def variants(self) -> VariantsConfig: ...

@final
class FunctionConfigJson:
    """FunctionConfigJson"""
    @property
    def assistant_schema(self) -> Dict[str, Any]: ...
    @property
    def output_schema(self) -> Dict[str, Any]: ...
    @property
    def system_schema(self) -> Dict[str, Any]: ...
    @property
    def type(self) -> str: ...
    @property
    def user_schema(self) -> Dict[str, Any]: ...
    @property
    def variants(self) -> VariantsConfig: ...

@final
class VariantsConfig:
    """VariantsConfig"""

    ...

@final
class ChatCompletionConfig:
    """ChatCompletionConfig"""
    @property
    def assistant_template(self) -> str: ...
    @property
    def model(self) -> str: ...
    @property
    def system_template(self) -> str: ...
    @property
    def user_template(self) -> str: ...

@final
class BestOfNSamplingConfig:
    """BestOfNSamplingConfig"""

    ...

@final
class MixtureOfNConfig:
    """MixtureOfNConfig"""

    ...

@final
class ChainOfThoughtConfig:
    """ChainOfThoughtConfig"""

    ...

# ============================================================================
# Gateway Types
# ============================================================================

class BaseTensorZeroGateway:
    """Base class for TensorZero gateways."""
    def experimental_get_config(self) -> Config: ...

@final
class TensorZeroGateway(BaseTensorZeroGateway):
    """A synchronous client for a TensorZero gateway."""
    def __enter__(self) -> "TensorZeroGateway": ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...
    @classmethod
    def build_embedded(
        cls,
        *,
        config_file: Optional[str] = None,
        clickhouse_url: Optional[str] = None,
        postgres_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> "TensorZeroGateway": ...
    @classmethod
    def build_http(
        cls,
        *,
        gateway_url: str,
        timeout: Optional[float] = None,
        verbose_errors: bool = False,
        api_key: Optional[str] = None,
    ) -> "TensorZeroGateway": ...
    def bulk_insert_datapoints(self, *, dataset_name: str, datapoints: List[Dict[str, Any]]) -> Any: ...
    def close(
        self,
    ) -> Any: ...
    def create_datapoints(self, *, dataset_name: str, datapoints: List[Dict[str, Any]]) -> CreateDatapointsResponse: ...
    def create_from_inferences(
        self, *, dataset_name: str, params: Dict[str, Any], output_source: Optional[str] = None
    ) -> Any: ...
    def delete_datapoint(self, *, dataset_name: str, datapoint_id: UUID) -> DeleteDatapointsResponse: ...
    def delete_datapoints(self, *, dataset_name: str, ids: List[UUID]) -> DeleteDatapointsResponse: ...
    def delete_dataset(self, *, dataset_name: str) -> Any: ...
    def dynamic_evaluation_run(
        self,
        *,
        variants: List[str],
        tags: Optional[Dict[str, str]] = None,
        project_name: Any = None,
        display_name: Any = None,
    ) -> EvaluationJobHandler: ...
    def dynamic_evaluation_run_episode(
        self, *, run_id: UUID, task_name: Any = None, tags: Optional[Dict[str, str]] = None
    ) -> Any: ...
    def experimental_get_config(
        self,
    ) -> Any: ...
    def experimental_launch_optimization(
        self,
        *,
        train_samples: Any,
        val_samples: Any = None,
        optimization_config: Union[
            OpenAISFTConfig,
            OpenAIRFTConfig,
            FireworksSFTConfig,
            TogetherSFTConfig,
            GCPVertexGeminiSFTConfig,
            DICLConfig,
            DICLOptimizationConfig,
        ],
    ) -> OptimizationJobHandle: ...
    def experimental_list_inferences(
        self,
        *,
        function_name: str,
        variant_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        output_source: Optional[str] = "inference",
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Any: ...
    def experimental_poll_optimization(self, *, job_handle: OptimizationJobHandle) -> OptimizationJobInfo: ...
    def experimental_render_inferences(self, *, stored_inferences: Any, variants: List[str]) -> Any: ...
    def experimental_render_samples(self, *, stored_samples: Any, variants: List[str]) -> Any: ...
    def experimental_run_evaluation(
        self,
        *,
        evaluation_name: str,
        dataset_name: str,
        variant_name: str,
        concurrency: Optional[int] = 1,
        inference_cache: Optional[str] = "on",
    ) -> EvaluationJobHandler: ...
    def feedback(
        self,
        *,
        metric_name: str,
        value: float,
        inference_id: Optional[UUID] = None,
        episode_id: Optional[UUID] = None,
        dryrun: Optional[bool] = None,
        internal: Optional[bool] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Any: ...
    def get_datapoint(self, *, dataset_name: str, datapoint_id: UUID) -> Union[Datapoint, GetDatapointsResponse]: ...
    def get_datapoints(self, *, ids: List[UUID]) -> Union[Datapoint, GetDatapointsResponse]: ...
    def inference(
        self,
        *,
        input: Dict[str, Any],
        function_name: Optional[str] = None,
        model_name: Optional[str] = None,
        episode_id: Optional[UUID] = None,
        stream: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
        variant_name: Optional[str] = None,
        dryrun: Optional[bool] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        allowed_tools: Optional[List[str]] = None,
        additional_tools: Optional[List[Dict[str, Any]]] = None,
        provider_tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
        internal: Optional[bool] = None,
        tags: Optional[Dict[str, str]] = None,
        credentials: Optional[Dict[str, Any]] = None,
        cache_options: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        include_original_response: Optional[bool] = None,
        otlp_traces_extra_headers: Optional[Dict[str, str]] = None,
        internal_dynamic_variant_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: ...
    def list_datapoints(
        self,
        *,
        dataset_name: str,
        function_name: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Datapoint]: ...
    def update_datapoints(
        self, *, dataset_name: str, requests: List[Union[UpdateChatDatapointRequest, UpdateJsonDatapointRequest]]
    ) -> UpdateDatapointsResponse: ...
    def update_datapoints_metadata(
        self, *, dataset_name: str, datapoints: List[Dict[str, Any]]
    ) -> UpdateDatapointsResponse: ...
    def workflow_evaluation_run(
        self,
        *,
        variants: List[str],
        tags: Optional[Dict[str, str]] = None,
        project_name: Any = None,
        display_name: Any = None,
    ) -> EvaluationJobHandler: ...
    def workflow_evaluation_run_episode(
        self, *, run_id: UUID, task_name: Any = None, tags: Optional[Dict[str, str]] = None
    ) -> Any: ...

@final
class AsyncTensorZeroGateway(BaseTensorZeroGateway):
    """An async client for a TensorZero gateway."""
    async def __aenter__(self) -> "AsyncTensorZeroGateway": ...
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...
    @classmethod
    def build_embedded(
        cls,
        *,
        config_file: Optional[str] = None,
        clickhouse_url: Optional[str] = None,
        postgres_url: Optional[str] = None,
        timeout: Optional[float] = None,
        async_setup: bool = True,
    ) -> Coroutine[Any, Any, "AsyncTensorZeroGateway"]: ...
    @classmethod
    def build_http(
        cls,
        *,
        gateway_url: str,
        timeout: Optional[float] = None,
        verbose_errors: bool = False,
        async_setup: bool = True,
        api_key: Optional[str] = None,
    ) -> Coroutine[Any, Any, "AsyncTensorZeroGateway"]: ...
    def bulk_insert_datapoints(
        self, *, dataset_name: str, datapoints: List[Dict[str, Any]]
    ) -> Coroutine[Any, Any, Any]: ...
    def close(
        self,
    ) -> Any: ...
    def create_datapoints(
        self, *, dataset_name: str, datapoints: List[Dict[str, Any]]
    ) -> Coroutine[Any, Any, CreateDatapointsResponse]: ...
    def create_from_inferences(
        self, *, dataset_name: str, params: Dict[str, Any], output_source: Optional[str] = None
    ) -> Coroutine[Any, Any, Any]: ...
    def delete_datapoint(
        self, *, dataset_name: str, datapoint_id: UUID
    ) -> Coroutine[Any, Any, DeleteDatapointsResponse]: ...
    def delete_datapoints(
        self, *, dataset_name: str, ids: List[UUID]
    ) -> Coroutine[Any, Any, DeleteDatapointsResponse]: ...
    def delete_dataset(self, *, dataset_name: str) -> Coroutine[Any, Any, Any]: ...
    def dynamic_evaluation_run(
        self,
        *,
        variants: List[str],
        tags: Optional[Dict[str, str]] = None,
        project_name: Any = None,
        display_name: Any = None,
    ) -> Coroutine[Any, Any, AsyncEvaluationJobHandler]: ...
    def dynamic_evaluation_run_episode(
        self, *, run_id: UUID, task_name: Any = None, tags: Optional[Dict[str, str]] = None
    ) -> Coroutine[Any, Any, Any]: ...
    def experimental_get_config(
        self,
    ) -> Config: ...
    def experimental_launch_optimization(
        self,
        *,
        train_samples: Any,
        val_samples: Any = None,
        optimization_config: Union[
            OpenAISFTConfig,
            OpenAIRFTConfig,
            FireworksSFTConfig,
            TogetherSFTConfig,
            GCPVertexGeminiSFTConfig,
            DICLConfig,
            DICLOptimizationConfig,
        ],
    ) -> Coroutine[Any, Any, OptimizationJobHandle]: ...
    def experimental_list_inferences(
        self,
        *,
        function_name: str,
        variant_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        output_source: Optional[str] = "inference",
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Coroutine[Any, Any, Any]: ...
    def experimental_poll_optimization(
        self, *, job_handle: OptimizationJobHandle
    ) -> Coroutine[Any, Any, OptimizationJobInfo]: ...
    def experimental_render_inferences(
        self, *, stored_inferences: Any, variants: List[str]
    ) -> Coroutine[Any, Any, Any]: ...
    def experimental_render_samples(self, *, stored_samples: Any, variants: List[str]) -> Coroutine[Any, Any, Any]: ...
    def experimental_run_evaluation(
        self,
        *,
        evaluation_name: str,
        dataset_name: str,
        variant_name: str,
        concurrency: Optional[int] = 1,
        inference_cache: Optional[str] = "on",
    ) -> Coroutine[Any, Any, AsyncEvaluationJobHandler]: ...
    def feedback(
        self,
        *,
        metric_name: str,
        value: float,
        inference_id: Optional[UUID] = None,
        episode_id: Optional[UUID] = None,
        dryrun: Optional[bool] = None,
        internal: Optional[bool] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Coroutine[Any, Any, Any]: ...
    def get_datapoint(
        self, *, dataset_name: str, datapoint_id: UUID
    ) -> Coroutine[Any, Any, Union[Datapoint, GetDatapointsResponse]]: ...
    def get_datapoints(self, *, ids: List[UUID]) -> Coroutine[Any, Any, Union[Datapoint, GetDatapointsResponse]]: ...
    def inference(
        self,
        *,
        input: Dict[str, Any],
        function_name: Optional[str] = None,
        model_name: Optional[str] = None,
        episode_id: Optional[UUID] = None,
        stream: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
        variant_name: Optional[str] = None,
        dryrun: Optional[bool] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        allowed_tools: Optional[List[str]] = None,
        additional_tools: Optional[List[Dict[str, Any]]] = None,
        provider_tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
        internal: Optional[bool] = None,
        tags: Optional[Dict[str, str]] = None,
        credentials: Optional[Dict[str, Any]] = None,
        cache_options: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        include_original_response: Optional[bool] = None,
        otlp_traces_extra_headers: Optional[Dict[str, str]] = None,
        internal_dynamic_variant_config: Optional[Dict[str, Any]] = None,
    ) -> Coroutine[Any, Any, Dict[str, Any]]: ...
    def list_datapoints(
        self,
        *,
        dataset_name: str,
        function_name: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Coroutine[Any, Any, List[Datapoint]]: ...
    def update_datapoints(
        self, *, dataset_name: str, requests: List[Union[UpdateChatDatapointRequest, UpdateJsonDatapointRequest]]
    ) -> Coroutine[Any, Any, UpdateDatapointsResponse]: ...
    def update_datapoints_metadata(
        self, *, dataset_name: str, datapoints: List[Dict[str, Any]]
    ) -> Coroutine[Any, Any, UpdateDatapointsResponse]: ...
    def workflow_evaluation_run(
        self,
        *,
        variants: List[str],
        tags: Optional[Dict[str, str]] = None,
        project_name: Any = None,
        display_name: Any = None,
    ) -> Coroutine[Any, Any, AsyncEvaluationJobHandler]: ...
    def workflow_evaluation_run_episode(
        self, *, run_id: UUID, task_name: Any = None, tags: Optional[Dict[str, str]] = None
    ) -> Coroutine[Any, Any, Any]: ...

@final
class LocalHttpGateway(BaseTensorZeroGateway):
    """LocalHttpGateway"""
    @property
    def base_url(self) -> str: ...
    def close(self) -> None: ...

# ============================================================================
# Inference Types
# ============================================================================

@final
class StoredInference:
    """StoredInference - tagged enum."""

    Chat: Type["StoredInference"]
    Json: Type["StoredInference"]
    @property
    def additional_tools(self) -> List[Dict[str, Any]]: ...
    @property
    def allowed_tools(self) -> List[str]: ...
    @property
    def dispreferred_outputs(self) -> List[Dict[str, Any]]: ...
    @property
    def episode_id(self) -> UUID: ...
    @property
    def function_name(self) -> str: ...
    @property
    def inference_id(self) -> UUID: ...
    @property
    def input(self) -> Dict[str, Any]: ...
    @property
    def output(self) -> Dict[str, Any]: ...
    @property
    def output_schema(self) -> Dict[str, Any]: ...
    @property
    def parallel_tool_calls(self) -> bool: ...
    @property
    def provider_tools(self) -> List[Dict[str, Any]]: ...
    @property
    def tags(self) -> Dict[str, str]: ...
    @property
    def timestamp(self) -> str: ...
    @property
    def type(self) -> str: ...
    @property
    def variant_name(self) -> str: ...

@final
class RenderedSample:
    """RenderedSample"""
    @property
    def additional_tools(self) -> List[Dict[str, Any]]: ...
    @property
    def allowed_tools(self) -> List[str]: ...
    @property
    def dispreferred_outputs(self) -> List[Dict[str, Any]]: ...
    @property
    def episode_id(self) -> UUID: ...
    @property
    def function_name(self) -> str: ...
    @property
    def inference_id(self) -> UUID: ...
    @property
    def input(self) -> Dict[str, Any]: ...
    @property
    def output(self) -> Dict[str, Any]: ...
    @property
    def output_schema(self) -> Dict[str, Any]: ...
    @property
    def parallel_tool_calls(self) -> bool: ...
    @property
    def provider_tools(self) -> List[Dict[str, Any]]: ...
    @property
    def stored_input(self) -> Dict[str, Any]: ...
    @property
    def stored_output(self) -> Dict[str, Any]: ...
    @property
    def tags(self) -> Dict[str, str]: ...

@final
class ResolvedInput:
    """ResolvedInput"""
    @property
    def messages(self) -> List[Dict[str, Any]]: ...
    @property
    def system(self) -> str: ...

@final
class ResolvedInputMessage:
    """ResolvedInputMessage"""
    @property
    def content(self) -> str: ...
    @property
    def role(self) -> str: ...

# ============================================================================
# Optimization Types
# ============================================================================

@final
class OptimizationJobHandle:
    """OptimizationJobHandle - tagged enum."""

    Dicl: Type["OptimizationJobHandle"]
    FireworksSFT: Type["OptimizationJobHandle"]
    GCPVertexGeminiSFT: Type["OptimizationJobHandle"]
    OpenAIRFT: Type["OptimizationJobHandle"]
    OpenAISFT: Type["OptimizationJobHandle"]
    TogetherSFT: Type["OptimizationJobHandle"]

@final
class OptimizationJobInfo:
    """OptimizationJobInfo"""
    @property
    def estimated_finish(self) -> str: ...
    @property
    def message(self) -> str: ...
    @property
    def output(self) -> Dict[str, Any]: ...
    @property
    def status(self) -> OptimizationJobStatus: ...

@final
class OptimizationJobStatus:
    """OptimizationJobStatus - enum."""

    Completed: Type["OptimizationJobStatus"]
    Failed: Type["OptimizationJobStatus"]
    Pending: Type["OptimizationJobStatus"]

# ============================================================================
# Evaluation Types
# ============================================================================

@final
class EvaluationJobHandler:
    """EvaluationJobHandler"""
    @property
    def run_info(self) -> Dict[str, Any]: ...
    def results(self) -> Dict[str, Any]: ...
    def summary_stats(self) -> Dict[str, Any]: ...

@final
class AsyncEvaluationJobHandler:
    """AsyncEvaluationJobHandler"""
    @property
    def run_info(self) -> Dict[str, Any]: ...
    def results(self) -> Dict[str, Any]: ...
    def summary_stats(self) -> Dict[str, Any]: ...

# ============================================================================
# Optimization Config Types
# ============================================================================

@final
class OpenAISFTConfig:
    """OpenAISFTConfig"""
    def __init__(
        self,
        *,
        model: str,
        batch_size: Optional[int] = None,
        learning_rate_multiplier: Any = None,
        n_epochs: Any = None,
        credentials: Optional[Dict[str, Any]] = None,
        api_base: Any = None,
        seed: Any = None,
        suffix: Any = None,
    ) -> None: ...

@final
class OpenAIRFTConfig:
    """OpenAIRFTConfig"""
    def __init__(
        self,
        *,
        model: str,
        grader: Any,
        response_format: Any = None,
        batch_size: Optional[int] = None,
        compute_multiplier: Any = None,
        eval_interval: Any = None,
        eval_samples: Any = None,
        learning_rate_multiplier: Any = None,
        n_epochs: Any = None,
        reasoning_effort: Any = None,
        credentials: Optional[Dict[str, Any]] = None,
        api_base: Any = None,
        seed: Any = None,
        suffix: Any = None,
    ) -> None: ...

@final
class FireworksSFTConfig:
    """FireworksSFTConfig"""
    def __init__(
        self,
        *,
        model: str,
        early_stop: Any = None,
        epochs: Any = None,
        learning_rate: Any = None,
        max_context_length: Any = None,
        lora_rank: Any = None,
        batch_size: Optional[int] = None,
        display_name: Any = None,
        output_model: Any = None,
        warm_start_from: Any = None,
        is_turbo: Any = None,
        eval_auto_carveout: Any = None,
        nodes: Any = None,
        mtp_enabled: Any = None,
        mtp_num_draft_tokens: Any = None,
        mtp_freeze_base_model: Any = None,
        credentials: Optional[Dict[str, Any]] = None,
        account_id: str,
        api_base: Any = None,
    ) -> None: ...

@final
class TogetherSFTConfig:
    """TogetherSFTConfig"""
    def __init__(
        self,
        *,
        model: str,
        credentials: Optional[Dict[str, Any]] = None,
        api_base: Any = None,
        n_epochs: Any = None,
        n_checkpoints: Any = None,
        n_evals: Any = None,
        batch_size: Optional[int] = None,
        learning_rate: Any = None,
        warmup_ratio: Any = None,
        max_grad_norm: Any = None,
        weight_decay: Any = None,
        suffix: Any = None,
        lr_scheduler: Any = None,
        wandb_api_key: Any = None,
        wandb_base_url: Any = None,
        wandb_project_name: Any = None,
        wandb_name: Any = None,
        training_method: Any = None,
        training_type: Any = None,
        from_checkpoint: Any = None,
        from_hf_model: Any = None,
        hf_model_revision: Any = None,
        hf_api_token: Any = None,
        hf_output_repo_name: Any = None,
    ) -> None: ...

@final
class GCPVertexGeminiSFTConfig:
    """GCPVertexGeminiSFTConfig"""
    def __init__(
        self,
        *,
        model: str,
        bucket_name: Any,
        project_id: Any,
        region: Any,
        learning_rate_multiplier: Any = None,
        adapter_size: Any = None,
        n_epochs: Any = None,
        export_last_checkpoint_only: Any = None,
        credentials: Optional[Dict[str, Any]] = None,
        api_base: Any = None,
        seed: Any = None,
        service_account: Any = None,
        kms_key_name: Any = None,
        tuned_model_display_name: Any = None,
        bucket_path_prefix: Any = None,
    ) -> None: ...

@final
class DICLConfig:
    """DICLConfig"""
    def __init__(self, args: Any, kwargs: Any) -> None: ...
    __deprecated__: str

@final
class DICLOptimizationConfig:
    """DICLOptimizationConfig"""
    def __init__(
        self,
        *,
        embedding_model: str,
        variant_name: str,
        function_name: str,
        dimensions: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_concurrency: Optional[int] = None,
        k: Optional[int] = None,
        model: Optional[str] = None,
        append_to_existing_variants: Optional[bool] = None,
        credentials: Optional[Dict[str, Any]] = None,
    ) -> None: ...

# ============================================================================
# Internal Functions
# ============================================================================

def _start_http_gateway(*, config_file: str, clickhouse_url: str, postgres_url: str, async_setup: bool) -> Any: ...
