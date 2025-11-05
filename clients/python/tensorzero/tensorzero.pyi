# Type stubs for tensorzero.tensorzero
# This file defines type signatures for the Rust extension module

from typing import Any, ClassVar, Dict, List, Optional, final

__all__ = [
    # Dataset V1 API Types
    "JsonDatapointOutputUpdate",
    "DatapointMetadataUpdate",
    "CreateChatDatapointRequest",
    "CreateJsonDatapointRequest",
    "UpdateChatDatapointRequest",
    "UpdateJsonDatapointRequest",
    "UpdateDatapointMetadataRequest",
    "CreateDatapointsRequest",
    "UpdateDatapointsRequest",
    "UpdateDatapointsMetadataRequest",
    "GetDatapointsRequest",
    "DeleteDatapointsRequest",
    "CreateDatapointsFromInferenceRequest",
    "CreateDatapointsFromInferenceOutputSource",
    "Datapoint",
    "CreateDatapointsResponse",
    "UpdateDatapointsResponse",
    "GetDatapointsResponse",
    "DeleteDatapointsResponse",
    "ListDatapointsRequest",
    # Configuration Types
    "Config",
    "FunctionsConfig",
    "FunctionConfigChat",
    "FunctionConfigJson",
    "VariantsConfig",
    "ChatCompletionConfig",
    "BestOfNSamplingConfig",
    "MixtureOfNConfig",
    "ChainOfThoughtConfig",
    # Gateway Types
    "BaseTensorZeroGateway",
    "TensorZeroGateway",
    "AsyncTensorZeroGateway",
    "LocalHttpGateway",
    # Inference Types
    "StoredInference",
    "RenderedSample",
    "ResolvedInput",
    "ResolvedInputMessage",
    # Optimization Types
    "OptimizationJobHandle",
    "OptimizationJobInfo",
    "OptimizationJobStatus",
    # Evaluation Types
    "EvaluationJobHandler",
    "AsyncEvaluationJobHandler",
    # Optimization Config Types
    "OpenAISFTConfig",
    "OpenAIRFTConfig",
    "FireworksSFTConfig",
    "TogetherSFTConfig",
    "GCPVertexGeminiSFTConfig",
    "DICLConfig",
    "DICLOptimizationConfig",
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
    def __init__(self, *args: Any, name: Optional[Optional[str]], **kwargs: Any) -> None: ...
    @property
    def name(self) -> Optional[Optional[str]]: ...

@final
class CreateChatDatapointRequest:
    """Request to create a chat datapoint."""
    def __init__(
        self,
        function_name: str,
        input: Any,
        episode_id: Optional[Any] = None,
        output: Optional[Any] = None,
        dynamic_tool_params: Optional[Any] = None,
        tags: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    @property
    def function_name(self) -> str: ...
    @property
    def episode_id(self) -> Optional[Any]: ...
    @property
    def input(self) -> Any: ...
    @property
    def output(self) -> Optional[Any]: ...
    @property
    def dynamic_tool_params(self) -> Any: ...
    @property
    def tags(self) -> Optional[Dict[str, str]]: ...
    @property
    def name(self) -> Optional[str]: ...

@final
class CreateJsonDatapointRequest:
    """Request to create a JSON datapoint."""
    def __init__(
        self,
        function_name: str,
        input: Any,
        episode_id: Optional[Any] = None,
        output: Optional[JsonDatapointOutputUpdate] = None,
        output_schema: Optional[Any] = None,
        tags: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    @property
    def function_name(self) -> str: ...
    @property
    def episode_id(self) -> Optional[Any]: ...
    @property
    def input(self) -> Any: ...
    @property
    def output(self) -> Optional[JsonDatapointOutputUpdate]: ...
    @property
    def output_schema(self) -> Optional[Any]: ...
    @property
    def tags(self) -> Optional[Dict[str, str]]: ...
    @property
    def name(self) -> Optional[str]: ...

@final
class UpdateChatDatapointRequest:
    """Request to update a chat datapoint."""
    def __init__(
        self,
        *,
        id: Any,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        tool_params: Optional[Optional[Any]],
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[DatapointMetadataUpdate] = None,
        **kwargs: Any,
    ) -> None: ...
    @property
    def id(self) -> Any: ...
    @property
    def input(self) -> Optional[Any]: ...
    @property
    def output(self) -> Optional[Any]: ...
    @property
    def tool_params(self) -> Optional[Optional[Any]]: ...
    @property
    def tags(self) -> Optional[Dict[str, str]]: ...
    @property
    def metadata(self) -> Optional[DatapointMetadataUpdate]: ...

@final
class UpdateJsonDatapointRequest:
    """Request to update a JSON datapoint."""
    def __init__(
        self,
        *,
        id: Any,
        input: Optional[Any] = None,
        output: Optional[Optional[JsonDatapointOutputUpdate]],
        output_schema: Optional[Any] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[DatapointMetadataUpdate] = None,
        **kwargs: Any,
    ) -> None: ...
    @property
    def id(self) -> Any: ...
    @property
    def input(self) -> Optional[Any]: ...
    @property
    def output(self) -> Optional[Optional[JsonDatapointOutputUpdate]]: ...
    @property
    def output_schema(self) -> Optional[Any]: ...
    @property
    def tags(self) -> Optional[Dict[str, str]]: ...
    @property
    def metadata(self) -> Optional[DatapointMetadataUpdate]: ...

@final
class UpdateDatapointMetadataRequest:
    """Request to update datapoint metadata."""
    def __init__(self, id: Any, metadata: DatapointMetadataUpdate, *args: Any, **kwargs: Any) -> None: ...
    @property
    def id(self) -> Any: ...
    @property
    def metadata(self) -> DatapointMetadataUpdate: ...

@final
class ListDatapointsRequest:
    """Request to list datapoints with pagination and filters."""
    def __init__(
        self,
        function_name: Optional[str] = None,
        limit: Optional[int] = None,
        page_size: Optional[int] = None,
        offset: Optional[int] = None,
        filter: Optional[Any] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    @property
    def function_name(self) -> Optional[str]: ...
    @property
    def limit(self) -> Optional[int]: ...
    @property
    def page_size(self) -> Optional[int]: ...
    @property
    def offset(self) -> Optional[int]: ...
    @property
    def filter(self) -> Optional[Any]: ...

@final
class CreateDatapointsResponse:
    """Response from creating datapoints."""
    @property
    def ids(self) -> List[Any]: ...

@final
class UpdateDatapointsResponse:
    """Response from updating datapoints."""
    @property
    def ids(self) -> List[Any]: ...

@final
class GetDatapointsResponse:
    """Response containing retrieved datapoints."""
    @property
    def datapoints(self) -> List[Any]: ...

@final
class DeleteDatapointsResponse:
    """Response from deleting datapoints."""
    @property
    def num_deleted_datapoints(self) -> int: ...

@final
class CreateDatapointsRequest:
    """Request to create multiple datapoints."""
    def __init__(self, datapoints: List[Any], *args: Any, **kwargs: Any) -> None: ...
    @property
    def datapoints(self) -> List[Any]: ...

@final
class UpdateDatapointsRequest:
    """Request to update multiple datapoints."""
    def __init__(self, datapoints: List[Any], *args: Any, **kwargs: Any) -> None: ...
    @property
    def datapoints(self) -> List[Any]: ...

@final
class UpdateDatapointsMetadataRequest:
    """Request to update metadata for multiple datapoints."""
    def __init__(self, datapoints: List[Any], *args: Any, **kwargs: Any) -> None: ...
    @property
    def datapoints(self) -> List[Any]: ...

@final
class GetDatapointsRequest:
    """Request to get specific datapoints by their IDs."""
    def __init__(self, ids: List[Any], *args: Any, **kwargs: Any) -> None: ...
    @property
    def ids(self) -> List[Any]: ...

@final
class DeleteDatapointsRequest:
    """Request to delete specific datapoints by their IDs."""
    def __init__(self, ids: List[Any], *args: Any, **kwargs: Any) -> None: ...
    @property
    def ids(self) -> List[Any]: ...

@final
class CreateDatapointsFromInferenceRequest:
    """Request to create datapoints from inferences."""
    def __init__(
        self,
        params: Any,
        output_source: Optional[CreateDatapointsFromInferenceOutputSource] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    @property
    def params(self) -> Any: ...
    @property
    def output_source(self) -> Optional[CreateDatapointsFromInferenceOutputSource]: ...

@final
class CreateDatapointsFromInferenceOutputSource:
    """Specifies the source of output when creating datapoints from inferences.

    Variants (accessed as class attributes):
        - CreateDatapointsFromInferenceOutputSource.None
        - CreateDatapointsFromInferenceOutputSource.Inference
        - CreateDatapointsFromInferenceOutputSource.Demonstration
    """
    # Note: Cannot annotate None as it's a Python keyword, but it exists at runtime
    Inference: ClassVar["CreateDatapointsFromInferenceOutputSource"]
    Demonstration: ClassVar["CreateDatapointsFromInferenceOutputSource"]
    def __init__(self, *args: Any) -> None: ...
    def __getattribute__(self, name: str) -> Any: ...  # Allow runtime access to None

@final
class Datapoint:
    """A datapoint retrieved from the dataset."""
    def Chat(self, *args: Any, **kwargs: Any) -> Any: ...
    def Json(self, *args: Any, **kwargs: Any) -> Any: ...
    @property
    def additional_tools(self) -> Any: ...
    @property
    def allowed_tools(self) -> Any: ...
    @property
    def dataset_name(self) -> str: ...
    @property
    def function_name(self) -> str: ...
    @property
    def id(self) -> Any: ...
    @property
    def input(self) -> Any: ...
    @property
    def is_custom(self) -> bool: ...
    @property
    def name(self) -> Optional[str]: ...
    @property
    def output(self) -> Any: ...
    @property
    def output_schema(self) -> Optional[Any]: ...
    @property
    def parallel_tool_calls(self) -> Any: ...
    @property
    def provider_tools(self) -> Any: ...

# ============================================================================
# Configuration Types
# ============================================================================

@final
class Config:
    """TensorZero configuration."""
    @property
    def functions(self) -> FunctionsConfig: ...

@final
class FunctionsConfig:
    """Functions configuration."""
    ...

@final
class FunctionConfigChat:
    """Chat function configuration."""
    @property
    def assistant_schema(self) -> Optional[Any]: ...
    @property
    def system_schema(self) -> Optional[Any]: ...
    @property
    def type(self) -> str: ...
    @property
    def user_schema(self) -> Optional[Any]: ...
    @property
    def variants(self) -> VariantsConfig: ...

@final
class FunctionConfigJson:
    """JSON function configuration."""
    @property
    def assistant_schema(self) -> Optional[Any]: ...
    @property
    def output_schema(self) -> Any: ...
    @property
    def system_schema(self) -> Optional[Any]: ...
    @property
    def type(self) -> str: ...
    @property
    def user_schema(self) -> Optional[Any]: ...
    @property
    def variants(self) -> VariantsConfig: ...

@final
class VariantsConfig:
    """Variants configuration."""
    ...

@final
class ChatCompletionConfig:
    """Chat completion configuration."""
    @property
    def assistant_template(self) -> Optional[Any]: ...
    @property
    def model(self) -> str: ...
    @property
    def system_template(self) -> Optional[Any]: ...
    @property
    def user_template(self) -> Optional[Any]: ...

@final
class BestOfNSamplingConfig:
    """Best of N sampling configuration."""
    ...

@final
class MixtureOfNConfig:
    """Mixture of N configuration."""
    ...

@final
class ChainOfThoughtConfig:
    """Chain of thought configuration."""
    ...

# ============================================================================
# Gateway Types
# ============================================================================

class BaseTensorZeroGateway:
    """Base class for TensorZero gateways."""
    def experimental_get_config(self) -> Config: ...

@final
class TensorZeroGateway(BaseTensorZeroGateway):
    """Synchronous TensorZero gateway."""
    @staticmethod
    def build_embedded(
        *,
        config_file: Optional[str] = None,
        clickhouse_url: Optional[str] = None,
        postgres_url: Optional[str] = None,
        timeout: Optional[float] = None,
        async_setup: bool = True,
    ) -> "TensorZeroGateway": ...
    @staticmethod
    def build_http(
        *,
        gateway_url: str,
        timeout: Optional[float] = None,
        verbose_errors: bool = False,
        async_setup: bool = True,
        api_key: Optional[str] = None,
    ) -> "TensorZeroGateway": ...
    def bulk_insert_datapoints(
        self, *, dataset_name: str, datapoints: List[Any]
    ) -> List[Any]: ...
    def close(self) -> None: ...
    def create_datapoints(
        self, *, dataset_name: str, datapoints: List[Any]
    ) -> Any: ...
    def create_from_inferences(
        self, *, dataset_name: str, params: Any, output_source: Optional[str] = None
    ) -> List[Any]: ...
    def delete_datapoint(self, *, dataset_name: str, datapoint_id: Any) -> None: ...
    def delete_datapoints(self, *, dataset_name: str, ids: List[Any]) -> int: ...
    def delete_dataset(self, *, dataset_name: str) -> int: ...
    def dynamic_evaluation_run(
        self,
        *,
        variants: Dict[str, str],
        tags: Optional[Dict[str, str]] = None,
        project_name: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> EvaluationJobHandler: ...
    def dynamic_evaluation_run_episode(
        self, *, run_id: Any, task_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ) -> Any: ...
    def experimental_launch_optimization(
        self, *, train_samples: List[Any], val_samples: Optional[List[Any]] = None, optimization_config: Any
    ) -> OptimizationJobHandle: ...
    def experimental_list_inferences(
        self,
        *,
        function_name: str,
        variant_name: Optional[str] = None,
        filters: Optional[Any] = None,
        output_source: str = "inference",
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[StoredInference]: ...
    def experimental_poll_optimization(self, *, job_handle: OptimizationJobHandle) -> OptimizationJobInfo: ...
    def experimental_render_inferences(self, *, stored_inferences: List[StoredInference], variants: Dict[str, str]) -> List[RenderedSample]: ...
    def experimental_render_samples(self, *, stored_samples: List[Any], variants: Dict[str, str]) -> List[RenderedSample]: ...
    def experimental_run_evaluation(
        self,
        *,
        evaluation_name: str,
        dataset_name: str,
        variant_name: str,
        concurrency: int = 1,
        inference_cache: str = "on",
    ) -> EvaluationJobHandler: ...
    def feedback(self, *, inference_id: Any, metric_name: str, value: Any, tags: Optional[Dict[str, str]] = None) -> None: ...
    def get_datapoint(self, *, dataset_name: str, datapoint_id: Any) -> Datapoint: ...
    def get_datapoints(self, ids: List[Any]) -> GetDatapointsResponse: ...
    def inference(
        self,
        *,
        function_name: str,
        input: Any,
        stream: bool = False,
        episode_id: Optional[Any] = None,
        dryrun: bool = False,
        variant_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        credentials: Optional[Dict[str, str]] = None,
        params: Optional[Any] = None,
        dynamic_tool_params: Optional[Any] = None,
    ) -> Any: ...
    def list_datapoints(
        self,
        *,
        dataset_name: str,
        function_name: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filter: Optional[Any] = None,
    ) -> List[Datapoint]: ...
    def update_datapoints(self, *, dataset_name: str, requests: List[Any]) -> List[Any]: ...
    def update_datapoints_metadata(self, *, dataset_name: str, datapoints: List[Any]) -> List[Any]: ...
    def workflow_evaluation_run(
        self,
        *,
        variants: Dict[str, str],
        tags: Optional[Dict[str, str]] = None,
        project_name: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> EvaluationJobHandler: ...
    def workflow_evaluation_run_episode(
        self, *, run_id: Any, task_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ) -> Any: ...

@final
class AsyncTensorZeroGateway(BaseTensorZeroGateway):
    """Asynchronous TensorZero gateway."""
    async def __aenter__(self) -> "AsyncTensorZeroGateway": ...
    async def __aexit__(self, *args: Any) -> None: ...
    @staticmethod
    def build_embedded(
        *,
        config_file: Optional[str] = None,
        clickhouse_url: Optional[str] = None,
        postgres_url: Optional[str] = None,
        timeout: Optional[float] = None,
        async_setup: bool = True,
    ) -> "AsyncTensorZeroGateway": ...
    @staticmethod
    def build_http(
        *,
        gateway_url: str,
        timeout: Optional[float] = None,
        verbose_errors: bool = False,
        async_setup: bool = True,
        api_key: Optional[str] = None,
    ) -> "AsyncTensorZeroGateway": ...
    async def bulk_insert_datapoints(
        self, *, dataset_name: str, datapoints: List[Any]
    ) -> List[Any]: ...
    async def close(self) -> None: ...
    async def create_datapoints(
        self, *, dataset_name: str, datapoints: List[Any]
    ) -> Any: ...
    async def create_from_inferences(
        self, *, dataset_name: str, params: Any, output_source: Optional[str] = None
    ) -> List[Any]: ...
    async def delete_datapoint(self, *, dataset_name: str, datapoint_id: Any) -> None: ...
    async def delete_datapoints(self, *, dataset_name: str, ids: List[Any]) -> int: ...
    async def delete_dataset(self, *, dataset_name: str) -> int: ...
    async def dynamic_evaluation_run(
        self,
        *,
        variants: Dict[str, str],
        tags: Optional[Dict[str, str]] = None,
        project_name: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> AsyncEvaluationJobHandler: ...
    async def dynamic_evaluation_run_episode(
        self, *, run_id: Any, task_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ) -> Any: ...
    async def experimental_launch_optimization(
        self, *, train_samples: List[Any], val_samples: Optional[List[Any]] = None, optimization_config: Any
    ) -> OptimizationJobHandle: ...
    async def experimental_list_inferences(
        self,
        *,
        function_name: str,
        variant_name: Optional[str] = None,
        filters: Optional[Any] = None,
        output_source: str = "inference",
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[StoredInference]: ...
    async def experimental_poll_optimization(self, *, job_handle: OptimizationJobHandle) -> OptimizationJobInfo: ...
    async def experimental_render_inferences(self, *, stored_inferences: List[StoredInference], variants: Dict[str, str]) -> List[RenderedSample]: ...
    async def experimental_render_samples(self, *, stored_samples: List[Any], variants: Dict[str, str]) -> List[RenderedSample]: ...
    async def experimental_run_evaluation(
        self,
        *,
        evaluation_name: str,
        dataset_name: str,
        variant_name: str,
        concurrency: int = 1,
        inference_cache: str = "on",
    ) -> AsyncEvaluationJobHandler: ...
    async def feedback(self, *, inference_id: Any, metric_name: str, value: Any, tags: Optional[Dict[str, str]] = None) -> None: ...
    async def get_datapoint(self, *, dataset_name: str, datapoint_id: Any) -> Datapoint: ...
    async def get_datapoints(self, ids: List[Any]) -> GetDatapointsResponse: ...
    async def inference(
        self,
        *,
        function_name: str,
        input: Any,
        stream: bool = False,
        episode_id: Optional[Any] = None,
        dryrun: bool = False,
        variant_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        credentials: Optional[Dict[str, str]] = None,
        params: Optional[Any] = None,
        dynamic_tool_params: Optional[Any] = None,
    ) -> Any: ...
    async def list_datapoints(
        self,
        *,
        dataset_name: str,
        function_name: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filter: Optional[Any] = None,
    ) -> List[Datapoint]: ...
    async def update_datapoints(self, *, dataset_name: str, requests: List[Any]) -> List[Any]: ...
    async def update_datapoints_metadata(self, *, dataset_name: str, datapoints: List[Any]) -> List[Any]: ...
    async def workflow_evaluation_run(
        self,
        *,
        variants: Dict[str, str],
        tags: Optional[Dict[str, str]] = None,
        project_name: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> AsyncEvaluationJobHandler: ...
    async def workflow_evaluation_run_episode(
        self, *, run_id: Any, task_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ) -> Any: ...

@final
class LocalHttpGateway:
    """Local HTTP gateway."""
    @property
    def base_url(self) -> str: ...
    def close(self) -> None: ...

# ============================================================================
# Inference Types
# ============================================================================

@final
class StoredInference:
    """A stored inference result."""
    def Chat(self, *args: Any, **kwargs: Any) -> Any: ...
    def Json(self, *args: Any, **kwargs: Any) -> Any: ...
    @property
    def additional_tools(self) -> Any: ...
    @property
    def allowed_tools(self) -> Any: ...
    @property
    def dispreferred_outputs(self) -> Any: ...
    @property
    def episode_id(self) -> Optional[Any]: ...
    @property
    def function_name(self) -> str: ...
    @property
    def inference_id(self) -> Any: ...
    @property
    def input(self) -> Any: ...
    @property
    def output(self) -> Any: ...
    @property
    def output_schema(self) -> Optional[Any]: ...
    @property
    def parallel_tool_calls(self) -> Any: ...
    @property
    def provider_tools(self) -> Any: ...
    @property
    def tags(self) -> Dict[str, str]: ...
    @property
    def timestamp(self) -> Any: ...
    @property
    def type(self) -> str: ...
    @property
    def variant_name(self) -> str: ...

@final
class RenderedSample:
    """A rendered sample for evaluation."""
    @property
    def additional_tools(self) -> Any: ...
    @property
    def allowed_tools(self) -> Any: ...
    @property
    def dispreferred_outputs(self) -> Any: ...
    @property
    def episode_id(self) -> Optional[Any]: ...
    @property
    def function_name(self) -> str: ...
    @property
    def inference_id(self) -> Optional[Any]: ...
    @property
    def input(self) -> Any: ...
    @property
    def output(self) -> Any: ...
    @property
    def output_schema(self) -> Optional[Any]: ...
    @property
    def parallel_tool_calls(self) -> Any: ...
    @property
    def provider_tools(self) -> Any: ...
    @property
    def stored_input(self) -> Any: ...
    @property
    def stored_output(self) -> Any: ...
    @property
    def tags(self) -> Dict[str, str]: ...

@final
class ResolvedInput:
    """Resolved input for inference."""
    @property
    def messages(self) -> List[ResolvedInputMessage]: ...
    @property
    def system(self) -> Optional[str]: ...

@final
class ResolvedInputMessage:
    """Resolved input message."""
    @property
    def content(self) -> Any: ...
    @property
    def role(self) -> str: ...

# ============================================================================
# Optimization Types
# ============================================================================

@final
class OptimizationJobHandle:
    """Handle for an optimization job."""
    def Dicl(self, *args: Any, **kwargs: Any) -> Any: ...
    def FireworksSFT(self, *args: Any, **kwargs: Any) -> Any: ...
    def GCPVertexGeminiSFT(self, *args: Any, **kwargs: Any) -> Any: ...
    def OpenAIRFT(self, *args: Any, **kwargs: Any) -> Any: ...
    def OpenAISFT(self, *args: Any, **kwargs: Any) -> Any: ...
    def TogetherSFT(self, *args: Any, **kwargs: Any) -> Any: ...

@final
class OptimizationJobInfo:
    """Information about an optimization job."""
    @property
    def estimated_finish(self) -> Optional[Any]: ...
    @property
    def message(self) -> Optional[str]: ...
    @property
    def output(self) -> Optional[Any]: ...
    @property
    def status(self) -> OptimizationJobStatus: ...

@final
class OptimizationJobStatus:
    """Status of an optimization job."""
    Completed: ClassVar["OptimizationJobStatus"]
    Failed: ClassVar["OptimizationJobStatus"]
    Pending: ClassVar["OptimizationJobStatus"]

# ============================================================================
# Evaluation Types
# ============================================================================

@final
class EvaluationJobHandler:
    """Synchronous evaluation job handler."""
    @property
    def run_info(self) -> Any: ...
    def results(self) -> Any: ...
    def summary_stats(self) -> Any: ...

@final
class AsyncEvaluationJobHandler:
    """Asynchronous evaluation job handler."""
    @property
    def run_info(self) -> Any: ...
    async def results(self) -> Any: ...
    async def summary_stats(self) -> Any: ...

# ============================================================================
# Optimization Config Types
# ============================================================================

@final
class OpenAISFTConfig:
    """OpenAI SFT configuration."""
    ...

@final
class OpenAIRFTConfig:
    """OpenAI RFT configuration."""
    ...

@final
class FireworksSFTConfig:
    """Fireworks SFT configuration."""
    ...

@final
class TogetherSFTConfig:
    """Together SFT configuration."""
    ...

@final
class GCPVertexGeminiSFTConfig:
    """GCP Vertex Gemini SFT configuration."""
    ...

@final
class DICLConfig:
    """DICL configuration."""
    ...

@final
class DICLOptimizationConfig:
    """DICL optimization configuration."""
    ...
