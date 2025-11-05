# Type stubs for tensorzero.tensorzero
# This file defines type signatures for the Rust extension module

from typing import Any, ClassVar, Dict, List, Optional, final

# Dataset V1 API Types

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

# Datapoint Type
class Datapoint:
    """A datapoint retrieved from the dataset."""
    ...

# Configuration Types
class Config:
    """TensorZero configuration."""
    ...

class FunctionsConfig:
    """Functions configuration."""
    ...

class FunctionConfigChat:
    """Chat function configuration."""
    ...

class FunctionConfigJson:
    """JSON function configuration."""
    ...

class VariantsConfig:
    """Variants configuration."""
    ...

# Gateway Types
class BaseTensorZeroGateway:
    """Base class for TensorZero gateways."""
    ...

class TensorZeroGateway(BaseTensorZeroGateway):
    """Synchronous TensorZero gateway."""
    ...

class AsyncTensorZeroGateway(BaseTensorZeroGateway):
    """Asynchronous TensorZero gateway."""
    ...

class LocalHttpGateway:
    """Local HTTP gateway."""
    ...

# Inference Types
class StoredInference:
    """A stored inference result."""
    ...

class RenderedSample:
    """A rendered sample for evaluation."""
    ...

class ResolvedInput:
    """Resolved input for inference."""
    ...

class ResolvedInputMessage:
    """Resolved input message."""
    ...

# Optimization Types
class OptimizationJobHandle:
    """Handle for an optimization job."""
    ...

class OptimizationJobInfo:
    """Information about an optimization job."""
    ...

class OptimizationJobStatus:
    """Status of an optimization job."""
    ...

# Evaluation Types
class EvaluationJobHandler:
    """Synchronous evaluation job handler."""
    ...

class AsyncEvaluationJobHandler:
    """Asynchronous evaluation job handler."""
    ...

# Optimization Config Types
class OpenAISFTConfig:
    """OpenAI SFT configuration."""
    ...

class OpenAIRFTConfig:
    """OpenAI RFT configuration."""
    ...

class FireworksSFTConfig:
    """Fireworks SFT configuration."""
    ...

class TogetherSFTConfig:
    """Together SFT configuration."""
    ...

class GCPVertexGeminiSFTConfig:
    """GCP Vertex Gemini SFT configuration."""
    ...

class DICLConfig:
    """DICL configuration."""
    ...

class DICLOptimizationConfig:
    """DICL optimization configuration."""
    ...

# Sampling Config Types
class ChatCompletionConfig:
    """Chat completion configuration."""
    ...

class BestOfNSamplingConfig:
    """Best of N sampling configuration."""
    ...

class MixtureOfNConfig:
    """Mixture of N configuration."""
    ...

class ChainOfThoughtConfig:
    """Chain of thought configuration."""
    ...
