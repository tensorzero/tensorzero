from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
    final,
)
from uuid import UUID

import uuid_utils
from typing_extensions import deprecated

# PyO3
from tensorzero import (
    ChatDatapointInsert,
    ChatInferenceOutput,
    ContentBlock,
    DynamicEvaluationRunEpisodeResponse,  # DEPRECATED
    DynamicEvaluationRunResponse,  # DEPRECATED
    ExtraBody,
    ExtraHeader,
    FeedbackResponse,
    InferenceChunk,
    InferenceInput,
    InferenceResponse,
    JsonDatapointInsert,
    OptimizationConfig,
    WorkflowEvaluationRunEpisodeResponse,
    WorkflowEvaluationRunResponse,
)
from tensorzero.internal import ModelInput, ToolCallConfigDatabaseInsert

# TODO: clean these up.
from tensorzero.types import (
    EvaluatorStatsDict,
    JsonInferenceOutput,
    OrderBy,
)

# Generated types
# NOTE(shuyangli): generated types should not be re-exported from the stub; they should be exported in __init__.py.
from .generated_types import (
    CreateDatapointRequest,
    CreateDatapointsFromInferenceRequestParams,
    CreateDatapointsResponse,
    Datapoint,
    DeleteDatapointsResponse,
    GetDatapointsResponse,
    GetInferencesResponse,
    InferenceFilter,
    Input,
    ListDatapointsRequest,
    ListInferencesRequest,
    StoredInference,
    UpdateDatapointMetadataRequest,
    UpdateDatapointRequest,
    UpdateDatapointsResponse,
)

@final
class ResolvedInputMessage:
    role: Literal["user", "assistant"]
    content: List[ContentBlock]

@final
class ResolvedInput:
    system: Optional[str | Dict[str, Any]]
    messages: List[ResolvedInputMessage]

@final
class EvaluationJobHandler:
    """
    Handler for synchronous evaluation job results.

    Results are cached in memory as you iterate to support summary_stats().
    For large evaluations, this may use significant memory.
    """

    @property
    def run_info(self) -> dict[str, Any]:
        """Get evaluation run metadata (evaluation_run_id, num_datapoints)."""
        ...

    def results(self) -> "EvaluationJobHandler":
        """Returns an iterator over evaluation results."""
        ...

    def __iter__(self) -> "EvaluationJobHandler": ...
    def __next__(self) -> dict[str, Any]:
        """
        Get next evaluation result.

        Returns dict with:
          - type: "success" | "error"
          - For success: datapoint, response, evaluations, evaluator_errors (all as dicts)
          - For error: datapoint_id (str), message (str)

        Note: Results are cached in memory for summary_stats() computation.
        """
        ...

    def summary_stats(self) -> dict[str, EvaluatorStatsDict]:
        """
        Get summary statistics from all consumed results.

        Uses cached results collected during iteration.
        Returns dict mapping evaluator names to
        {"mean": float, "stderr": float, "count": int}.
        """
        ...

    def __repr__(self) -> str: ...

@final
class AsyncEvaluationJobHandler:
    """
    Handler for asynchronous evaluation job results.

    Results are cached in memory as you iterate to support summary_stats().
    For large evaluations, this may use significant memory.
    """

    @property
    def run_info(self) -> dict[str, Any]:
        """Get evaluation run metadata (evaluation_run_id, num_datapoints)."""
        ...

    def results(self) -> "AsyncEvaluationJobHandler":
        """Returns an async iterator over evaluation results."""
        ...

    def __aiter__(self) -> "AsyncEvaluationJobHandler": ...
    async def __anext__(self) -> dict[str, Any]:
        """
        Get next evaluation result asynchronously.

        Returns dict with:
          - type: "success" | "error"
          - For success: datapoint, response, evaluations, evaluator_errors (all as dicts)
          - For error: datapoint_id (str), message (str)

        Note: Results are cached in memory for summary_stats() computation.
        """
        ...

    async def summary_stats(self) -> dict[str, EvaluatorStatsDict]:
        """
        Get summary statistics from all consumed results.

        Uses cached results collected during iteration.
        Returns dict mapping evaluator names to
        {"mean": float, "stderr": float, "count": int}.
        """
        ...

    def __repr__(self) -> str: ...

@final
class RenderedSample:
    function_name: str
    input: ModelInput
    stored_input: ResolvedInput
    output: Optional[ChatInferenceOutput]
    stored_output: Optional[Union[ChatInferenceOutput, JsonInferenceOutput]]
    episode_id: Optional[UUID]
    inference_id: Optional[UUID]
    tool_params: Optional[ToolCallConfigDatabaseInsert]
    output_schema: Optional[Dict[str, Any]]
    dispreferred_outputs: List[ChatInferenceOutput] = []
    tags: Dict[str, str]
    @property
    def allowed_tools(self) -> Optional[List[str]]: ...
    @property
    def additional_tools(self) -> Optional[List[Any]]: ...
    @property
    def parallel_tool_calls(self) -> Optional[bool]: ...
    @property
    def provider_tools(self) -> Optional[List[Any]]: ...

@final
class OptimizationJobHandle:
    Dicl: Type["OptimizationJobHandle"]
    OpenAISFT: Type["OptimizationJobHandle"]
    OpenAIRFT: Type["OptimizationJobHandle"]
    FireworksSFT: Type["OptimizationJobHandle"]
    GCPVertexGeminiSFT: Type["OptimizationJobHandle"]
    TogetherSFT: Type["OptimizationJobHandle"]
    GEPA: Type["OptimizationJobHandle"]

@final
class OptimizationJobStatus:
    Pending: Type["OptimizationJobStatus"]
    Completed: Type["OptimizationJobStatus"]
    Failed: Type["OptimizationJobStatus"]

@final
class OptimizationJobInfo:
    Dicl: Type["OptimizationJobInfo"]
    OpenAISFT: Type["OptimizationJobInfo"]
    OpenAIRFT: Type["OptimizationJobInfo"]
    FireworksSFT: Type["OptimizationJobInfo"]
    GCPVertexGeminiSFT: Type["OptimizationJobInfo"]
    TogetherSFT: Type["OptimizationJobInfo"]
    GEPA: Type["OptimizationJobInfo"]
    @property
    def message(self) -> str: ...
    @property
    def status(self) -> Type[OptimizationJobStatus]: ...
    @property
    def output(self) -> Optional[Any]: ...
    @property
    def estimated_finish(self) -> Optional[int]: ...

@final
class DICLOptimizationConfig:
    """
    Configuration for DICL (Dynamic In-Context Learning) optimization.
    """

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
    ) -> None:
        """
        Initialize the DICLOptimizationConfig.

        :param embedding_model: The embedding model to use (required).
        :param variant_name: The name to be used for the DICL variant (required).
        :param function_name: The name of the function to optimize (required).
        :param dimensions: The dimensions of the embeddings. If None, uses the model's default.
        :param batch_size: The batch size to use for getting embeddings.
        :param max_concurrency: The maximum concurrency to use for getting embeddings.
        :param k: The number of nearest neighbors to use for the DICL variant.
        :param model: The model to use for the DICL variant. This field will be required in a future release.
        :param append_to_existing_variants: Whether to append to existing variants.
        """
        ...

@final
class OpenAISFTConfig:
    def __init__(
        self,
        *,
        model: str,
        batch_size: Optional[int] = None,
        learning_rate_multiplier: Optional[float] = None,
        n_epochs: Optional[int] = None,
        seed: Optional[int] = None,
        suffix: Optional[str] = None,
    ) -> None: ...

@final
class OpenAIRFTConfig:
    def __init__(
        self,
        *,
        model: str,
        grader: Dict[str, Any],
        response_format: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None,
        compute_multiplier: Optional[float] = None,
        eval_interval: Optional[int] = None,
        eval_samples: Optional[int] = None,
        learning_rate_multiplier: Optional[float] = None,
        n_epochs: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        seed: Optional[int] = None,
        suffix: Optional[str] = None,
    ) -> None: ...

@final
class FireworksSFTConfig:
    def __init__(
        self,
        *,
        model: str,
        early_stop: Optional[bool] = None,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        max_context_length: Optional[int] = None,
        lora_rank: Optional[int] = None,
        batch_size: Optional[int] = None,
        display_name: Optional[str] = None,
        output_model: Optional[str] = None,
        warm_start_from: Optional[str] = None,
        is_turbo: Optional[bool] = None,
        eval_auto_carveout: Optional[bool] = None,
        nodes: Optional[int] = None,
        mtp_enabled: Optional[bool] = None,
        mtp_num_draft_tokens: Optional[int] = None,
        mtp_freeze_base_model: Optional[bool] = None,
        deploy_after_training: Optional[bool] = None,
    ) -> None: ...

@final
class GCPVertexGeminiSFTConfig:
    def __init__(
        self,
        *,
        model: str,
        learning_rate_multiplier: Optional[float] = None,
        adapter_size: Optional[int] = None,
        n_epochs: Optional[int] = None,
        export_last_checkpoint_only: Optional[bool] = None,
        seed: Optional[int] = None,
        tuned_model_display_name: Optional[str] = None,
    ) -> None: ...

@final
class GEPAConfig:
    def __init__(
        self,
        *,
        function_name: str,
        evaluation_name: str,
        analysis_model: str,
        mutation_model: str,
        initial_variants: Optional[List[str]] = None,
        variant_prefix: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_iterations: Optional[int] = None,
        max_concurrency: Optional[int] = None,
        seed: Optional[int] = None,
        timeout: Optional[int] = None,
        include_inference_for_mutation: Optional[bool] = None,
        retries: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
    ) -> None: ...

@final
class TogetherSFTConfig:
    """
    Configuration for Together supervised fine-tuning.

    For detailed API documentation, see: https://docs.together.ai/reference/post-fine-tunes
    """
    def __init__(
        self,
        *,
        model: str,
        n_epochs: Optional[int] = None,
        n_checkpoints: Optional[int] = None,
        n_evals: Optional[int] = None,
        batch_size: Optional[Union[int, str]] = None,
        learning_rate: Optional[float] = None,
        warmup_ratio: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        weight_decay: Optional[float] = None,
        suffix: Optional[str] = None,
        lr_scheduler: Optional[Dict[str, Any]] = None,
        wandb_name: Optional[str] = None,
        training_method: Optional[Dict[str, Any]] = None,
        training_type: Optional[Dict[str, Any]] = None,
        from_checkpoint: Optional[str] = None,
        from_hf_model: Optional[str] = None,
        hf_model_revision: Optional[str] = None,
        hf_output_repo_name: Optional[str] = None,
    ) -> None: ...

@final
class LegacyDatapoint:
    """
    A legacy type representing a datapoint.
    Deprecated; use `Datapoint` instead from v1 Datapoint APIs.
    """

    Chat: Type["LegacyDatapoint"]
    Json: Type["LegacyDatapoint"]

    @property
    def id(self) -> UUID: ...
    @property
    def input(self) -> Input: ...
    @property
    def output(self) -> Any: ...
    @property
    def dataset_name(self) -> str: ...
    @property
    def function_name(self) -> str: ...
    @property
    def allowed_tools(self) -> Optional[List[str]]: ...
    @property
    def additional_tools(self) -> Optional[List[Any]]: ...
    @property
    def parallel_tool_calls(self) -> Optional[bool]: ...
    @property
    def provider_tools(self) -> Optional[List[Any]]: ...
    @property
    def output_schema(self) -> Optional[Any]: ...
    @property
    def name(self) -> Optional[str]: ...
    @property
    def is_custom(self) -> bool: ...

@final
class ChatCompletionConfig:
    @property
    def system_template(self) -> Optional[str]: ...
    @property
    def user_template(self) -> Optional[str]: ...
    @property
    def assistant_template(self) -> Optional[str]: ...
    @property
    def model(self) -> str: ...

@final
class BestOfNSamplingConfig:
    pass

@final
class DICLConfig:
    __deprecated__: str = ...

@final
class MixtureOfNConfig:
    pass

@final
class ChainOfThoughtConfig:
    pass

@final
class VariantsConfig:
    def __len__(self) -> int: ...
    def __getitem__(
        self, key: str
    ) -> Union[
        ChatCompletionConfig,
        BestOfNSamplingConfig,
        DICLConfig,
        MixtureOfNConfig,
        ChainOfThoughtConfig,
    ]: ...

@final
class FunctionConfigChat:
    @property
    def type(self) -> Literal["chat"]: ...
    @property
    def variants(self) -> VariantsConfig: ...
    @property
    def system_schema(self) -> Optional[Any]: ...
    @property
    def user_schema(self) -> Optional[Any]: ...
    @property
    def assistant_schema(self) -> Optional[Any]: ...

@final
class FunctionConfigJson:
    @property
    def type(self) -> Literal["json"]: ...
    @property
    def variants(self) -> VariantsConfig: ...
    @property
    def system_schema(self) -> Optional[Any]: ...
    @property
    def user_schema(self) -> Optional[Any]: ...
    @property
    def assistant_schema(self) -> Optional[Any]: ...
    @property
    def output_schema(self) -> Optional[Any]: ...

@final
class FunctionsConfig:
    def __len__(self) -> int: ...
    def __getitem__(self, key: str) -> Union[FunctionConfigChat, FunctionConfigJson]: ...

@final
class Config:
    @property
    def functions(self) -> FunctionsConfig: ...

class BaseTensorZeroGateway:
    def experimental_get_config(self) -> Config: ...

@final
class TensorZeroGateway(BaseTensorZeroGateway):
    @classmethod
    def build_http(
        cls,
        *,
        gateway_url: str,
        timeout: Optional[float] = None,
        verbose_errors: bool = False,
        api_key: Optional[str] = None,
    ) -> "TensorZeroGateway":
        """
        Initialize the TensorZero client, using the HTTP gateway.
        :param gateway_url: The base URL of the TensorZero gateway. Example: "http://localhost:3000"
        :param timeout: The timeout for the HTTP client in seconds. If not provided, no timeout will be set.
        :param verbose_errors: If true, the client will increase the detail in errors (increasing the risk of leaking sensitive information).
        :param api_key: The API key to use for authentication with the TensorZero Gateway. If not provided, the client will attempt to read from the TENSORZERO_API_KEY environment variable.
        :return: A `TensorZeroGateway` instance configured to use the HTTP gateway.
        """

    @classmethod
    def build_embedded(
        cls,
        *,
        config_file: Optional[str] = None,
        clickhouse_url: Optional[str] = None,
        postgres_url: Optional[str] = None,
        valkey_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> "TensorZeroGateway":
        """
        Build a TensorZeroGateway instance.

        :param config_file: (Optional) The path to the TensorZero configuration file.
        :param clickhouse_url: (Optional) The URL of the ClickHouse database.
        :param postgres_url: (Optional) The URL of the Postgres database.
        :param valkey_url: (Optional) The URL of the Valkey instance.
        :param timeout: (Optional) The timeout for embedded gateway request processing, in seconds. If this timeout is hit, any in-progress LLM requests may be aborted. If not provided, no timeout will be set.
        """

    def inference(
        self,
        *,
        input: InferenceInput | Dict[str, Any],
        function_name: Optional[str] = None,
        model_name: Optional[str] = None,
        episode_id: Optional[str | UUID | uuid_utils.UUID] = None,
        stream: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
        variant_name: Optional[str] = None,
        dryrun: Optional[bool] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        allowed_tools: Optional[List[str]] = None,
        additional_tools: Optional[List[Dict[str, Any]]] = None,
        provider_tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[Literal["auto", "required", "off"], Dict[Literal["specific"], str]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        internal: Optional[bool] = None,
        tags: Optional[Dict[str, str]] = None,
        credentials: Optional[Dict[str, str]] = None,
        cache_options: Optional[Dict[str, Any]] = None,
        extra_body: Optional[List[ExtraBody | Dict[str, Any]]] = None,
        extra_headers: Optional[List[ExtraHeader | Dict[str, Any]]] = None,
        otlp_traces_extra_headers: Optional[Dict[str, str]] = None,
        otlp_traces_extra_attributes: Optional[Dict[str, str]] = None,
        otlp_traces_extra_resources: Optional[Dict[str, str]] = None,
        include_original_response: Optional[bool] = None,
        include_raw_response: Optional[bool] = None,
        include_raw_usage: Optional[bool] = None,
        internal_dynamic_variant_config: Optional[Dict[str, Any]] = None,
    ) -> Union[InferenceResponse, Iterator[InferenceChunk]]:
        """
        Make a POST request to the /inference endpoint.

        :param function_name: The name of the function to call
        :param input: The input to the function
                      Structure: {"system": Optional[str], "messages": List[{"role": "user" | "assistant", "content": Any}]}
                      The input will be validated server side against the input schema of the function being called.
        :param episode_id: The episode ID to use for the inference.
                           If this is the first inference in an episode, leave this field blank. The TensorZero gateway will generate and return a new episode ID.
                           Note: Only use episode IDs generated by the TensorZero gateway. Don't generate them yourself.
        :param stream: If set, the TensorZero gateway will stream partial message deltas (e.g. generated tokens) as it receives them from model providers.
        :param params: Override inference-time parameters for a particular variant type. Currently, we support:
                        {"chat_completion": {"temperature": float, "max_tokens": int, "seed": int}}
        :param variant_name: If set, pins the inference request to a particular variant.
                             Note: You should generally not do this, and instead let the TensorZero gateway assign a
                             particular variant. This field is primarily used for testing or debugging purposes.
        :param dryrun: If true, the request will be executed but won't be stored to the database.
        :param output_schema: If set, the JSON schema of a JSON function call will be validated against the given JSON Schema.
                              Overrides the output schema configured for the function.
        :param allowed_tools: If set, restricts the tools available during this inference request.
                              The list of names should be a subset of the tools configured for the function.
                              Tools provided at inference time in `additional_tools` (if any) are always available.
        :param additional_tools: A list of additional tools to use for the request. Each element should look like {"name": str, "parameters": valid JSON Schema, "description": str}
        :param provider_tools: A list of provider-specific tools to use for the request. Structure matches provider requirements.
        :param tool_choice: If set, overrides the tool choice strategy for the request.
                            It should be one of: "auto", "required", "off", or {"specific": str}. The last option pins the request to a specific tool name.
        :param parallel_tool_calls: If true, the request will allow for multiple tool calls in a single inference request.
        :param tags: If set, adds tags to the inference request.
        :param extra_body: If set, injects extra fields into the provider request body.
        :param extra_headers: If set, injects extra headers into the provider request.
        :param otlp_traces_extra_headers: If set, adds custom headers to OTLP trace exports. Headers will be automatically prefixed with "tensorzero-otlp-traces-extra-header-".
        :param otlp_traces_extra_attributes: If set, attaches custom HTTP headers to OTLP trace exports for this request.
                                             Headers will be automatically prefixed with "tensorzero-otlp-traces-extra-attributes-".
        :param otlp_traces_extra_resources: If set, attaches custom HTTP headers to OTLP trace exports for this request.
                                            Headers will be automatically prefixed with "tensorzero-otlp-traces-extra-resources-".
        :param include_original_response: DEPRECATED. Use `include_raw_response` instead.
        :param include_raw_response: If set, include raw provider-specific response data from all model inferences.
        :param include_raw_usage: If set, include raw provider-specific usage data in the response.
        :return: If stream is false, returns an InferenceResponse.
                 If stream is true, returns an async iterator that yields InferenceChunks as they come in.
        """

    def feedback(
        self,
        *,
        metric_name: str,
        value: Any,
        inference_id: Optional[str | UUID | uuid_utils.UUID] = None,
        episode_id: Optional[str | UUID | uuid_utils.UUID] = None,
        dryrun: Optional[bool] = None,
        internal: Optional[bool] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> FeedbackResponse:
        """
        Make a POST request to the /feedback endpoint.

        :param metric_name: The name of the metric to provide feedback for
        :param value: The value of the feedback. It should correspond to the metric type.
        :param inference_id: The inference ID to assign the feedback to.
                             Only use inference IDs that were returned by the TensorZero gateway.
                             Note: You can assign feedback to either an episode or an inference, but not both.
        :param episode_id: The episode ID to use for the request
                           Only use episode IDs that were returned by the TensorZero gateway.
                           Note: You can assign feedback to either an episode or an inference, but not both.
        :param dryrun: If true, the feedback request will be executed but won't be stored to the database (i.e. no-op).
        :param tags: If set, adds tags to the feedback request.
        :return: {"feedback_id": str}
        """

    def dynamic_evaluation_run(
        self,
        *,
        variants: Dict[str, str],
        tags: Optional[Dict[str, str]] = None,
        project_name: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> DynamicEvaluationRunResponse:
        """
        DEPRECATED: Use `workflow_evaluation_run` instead.
        Make a POST request to the /dynamic_evaluation_run endpoint.

        :param variants: A dictionary of variant names to variant values.
        :param tags: A dictionary of tags to add to the dynamic evaluation run.
        :param project_name: The name of the project to use for the dynamic evaluation run.
        :param display_name: The display name of the dynamic evaluation run.
        :return: A `DynamicEvaluationRunResponse` instance ({"run_id": str}).
        """

    def dynamic_evaluation_run_episode(
        self,
        *,
        run_id: str | UUID | uuid_utils.UUID,
        task_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> DynamicEvaluationRunEpisodeResponse:
        """
        DEPRECATED: Use `workflow_evaluation_run_episode` instead.

        Make a POST request to the /dynamic_evaluation_run/{run_id}/episode endpoint.

        :param run_id: The run ID to use for the dynamic evaluation run.
        :param task_name: The name of the task to use for the dynamic evaluation run.
        :param tags: A dictionary of tags to add to the dynamic evaluation run.
        :return: A `DynamicEvaluationRunEpisodeResponse` instance ({"episode_id": str}).
        """

    def workflow_evaluation_run(
        self,
        *,
        variants: Dict[str, str],
        tags: Optional[Dict[str, str]] = None,
        project_name: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> WorkflowEvaluationRunResponse:
        """
        Make a POST request to the /workflow_evaluation_run endpoint.

        :param variants: A dictionary of variant names to variant values.
        :param tags: A dictionary of tags to add to the workflow evaluation run.
        :param project_name: The name of the project to use for the workflow evaluation run.
        :param display_name: The display name of the workflow evaluation run.
        :return: A `WorkflowEvaluationRunResponse` instance ({"run_id": str}).
        """

    def workflow_evaluation_run_episode(
        self,
        *,
        run_id: str | UUID | uuid_utils.UUID,
        task_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> WorkflowEvaluationRunEpisodeResponse:
        """
        Make a POST request to the /workflow_evaluation_run/{run_id}/episode endpoint.

        :param run_id: The run ID to use for the workflow evaluation run.
        :param task_name: The name of the task to use for the workflow evaluation run.
        :param tags: A dictionary of tags to add to the workflow evaluation run.
        :return: A `WorkflowEvaluationRunEpisodeResponse` instance ({"episode_id": str}).
        """

    @deprecated("Deprecated since version 2025.11.4; use `create_datapoints` instead.")
    def create_datapoints_legacy(
        self,
        *,
        dataset_name: str,
        datapoints: Sequence[Union[ChatDatapointInsert, JsonDatapointInsert]],
    ) -> List[UUID]:
        """
        Make a POST request to the /datasets/{dataset_name}/datapoints endpoint.

        :param dataset_name: The name of the dataset to insert the datapoints into.
        :param datapoints: A list of datapoints to insert.
        """

    @deprecated("Deprecated since version 2025.11.4; use `create_datapoints` instead.")
    def bulk_insert_datapoints(
        self,
        *,
        dataset_name: str,
        datapoints: Sequence[Union[ChatDatapointInsert, JsonDatapointInsert]],
    ) -> List[UUID]:
        """
        DEPRECATED: Use `create_datapoints` instead.

        Make a POST request to the /datasets/{dataset_name}/datapoints/bulk endpoint.

        :param dataset_name: The name of the dataset to insert the datapoints into.
        :param datapoints: A list of datapoints to insert.
        """

    @deprecated("Deprecated since 2025.11.4; use `delete_datapoints` instead.")
    def delete_datapoint(
        self,
        *,
        dataset_name: str,
        datapoint_id: UUID,
    ) -> None:
        """
        Make a DELETE request to the /datasets/{dataset_name}/datapoints/{datapoint_id} endpoint.

        :param dataset_name: The name of the dataset to delete the datapoint from.
        :param datapoint_id: The ID of the datapoint to delete.
        """

    @deprecated("Deprecated since 2025.11.4; use `list_datapoints` instead.")
    def list_datapoints_legacy(
        self,
        *,
        dataset_name: str,
        function_name: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Datapoint]:
        """
        Make a GET request to the /datasets/{dataset_name}/datapoints endpoint.

        :param dataset_name: The name of the dataset to list the datapoints from.
        :param function_name: The name of the function to list the datapoints from.
        :param limit: The maximum number of datapoints to return.
        :param offset: The offset to start the list from.
        :return: A list of `Datapoint` instances.
        """

    def list_datapoints(
        self,
        *,
        dataset_name: str,
        request: ListDatapointsRequest,
    ) -> GetDatapointsResponse:
        """
        Lists datapoints in the dataset.

        :param dataset_name: The name of the dataset to list the datapoints from.
        :param request: The request to list the datapoints.
        :return: A `GetDatapointsResponse` containing the datapoints.
        """

    def get_datapoint(
        self,
        *,
        dataset_name: str,
        datapoint_id: UUID,
    ) -> LegacyDatapoint:
        """
        Make a GET request to the /datasets/{dataset_name}/datapoints/{datapoint_id} endpoint.

        :param dataset_name: The name of the dataset to get the datapoint from.
        :param datapoint_id: The ID of the datapoint to get.
        :return: A `Datapoint` instance.
        """

    def create_datapoints(
        self,
        *,
        dataset_name: str,
        requests: Sequence[CreateDatapointRequest],
    ) -> CreateDatapointsResponse:
        """
        Creates new datapoints in the dataset.

        :param dataset_name: The name of the dataset to create the datapoints in.
        :param requests: A list of datapoints to create.
        :return: A CreateDatapointsResponse object containing the IDs of the newly-created datapoints.
        """

    def update_datapoints(
        self,
        *,
        dataset_name: str,
        requests: Sequence[UpdateDatapointRequest],
    ) -> UpdateDatapointsResponse:
        """
        Update one or more datapoints in a dataset.

        :param dataset_name: The name of the dataset containing the datapoints to update.
        :param requests: A sequence of UpdateDatapointRequest objects.
        :return: An `UpdateDatapointsResponse` object.
        """

    def get_datapoints(
        self,
        *,
        dataset_name: str | None = ...,
        ids: Sequence[str],
    ) -> GetDatapointsResponse:
        """
        Get specific datapoints by their IDs.

        :param dataset_name: Optional dataset name containing the datapoints. Including this improves
            query performance because the dataset is part of the sorting key.
        :param ids: A sequence of datapoint IDs to retrieve. They should be in UUID format.
        :return: A `GetDatapointsResponse` object.
        """

    def update_datapoints_metadata(
        self,
        *,
        dataset_name: str,
        requests: Sequence[UpdateDatapointMetadataRequest],
    ) -> UpdateDatapointsResponse:
        """
        Update metadata for one or more datapoints.

        :param dataset_name: The name of the dataset containing the datapoints.
        :param requests: A sequence of UpdateDatapointMetadataRequest objects.
        :return: A `UpdateDatapointsResponse` object.
        """

    def delete_datapoints(
        self,
        *,
        dataset_name: str,
        ids: Sequence[str],
    ) -> DeleteDatapointsResponse:
        """
        Delete multiple datapoints from a dataset.

        :param dataset_name: The name of the dataset to delete datapoints from.
        :param ids: A sequence of datapoint IDs to delete. They should be in UUID format.
        :return: A `DeleteDatapointsResponse` object.
        """

    def delete_dataset(
        self,
        *,
        dataset_name: str,
    ) -> DeleteDatapointsResponse:
        """
        Delete a dataset and all of its datapoints.

        :param dataset_name: The name of the dataset to delete.
        :return: A `DeleteDatapointsResponse` object.
        """

    def create_datapoints_from_inferences(
        self,
        *,
        dataset_name: str,
        params: CreateDatapointsFromInferenceRequestParams,
        output_source: Optional[Literal["none", "inference", "demonstration"]] = None,
    ) -> CreateDatapointsResponse:
        """
        Create datapoints from inferences.

        :param dataset_name: The name of the dataset to create datapoints in.
        :param params: The parameters specifying which inferences to convert to datapoints.
        :param output_source: The source of the output to create datapoints from. "none", "inference", or "demonstration".
                             Can also be specified inside `params.output_source`. If both are provided, an error is raised.
        :return: A `CreateDatapointsResponse` object.
        """

    def get_inferences(
        self,
        *,
        ids: Sequence[str],
        function_name: Optional[str] = None,
        output_source: str = "inference",
    ) -> GetInferencesResponse:
        """
        Get specific inferences by their IDs.

        :param ids: A sequence of inference IDs to retrieve. They should be in UUID format.
        :param function_name: Optional function name to filter by (improves query performance).
        :param output_source: The source of the output ("inference" or "demonstration"). Default: "inference".
        :return: A `GetInferencesResponse` object.
        """

    def list_inferences(
        self,
        *,
        request: ListInferencesRequest,
    ) -> GetInferencesResponse:
        """
        List inferences with optional filtering, pagination, and sorting.

        :param request: A `ListInferencesRequest` object with filter parameters.
        :return: A `GetInferencesResponse` object.
        """

    def experimental_list_inferences(
        self,
        *,
        function_name: str,
        variant_name: Optional[str] = None,
        filters: Optional[InferenceFilter] = None,
        output_source: str = "inference",
        order_by: Optional[List[OrderBy]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[StoredInference]:
        """
        Query the Clickhouse database for inferences.
        This function is only available in EmbeddedGateway mode.

        :param function_name: The name of the function to query.
        :param variant_name: The name of the variant to query. Optional
        :param filters: A filter tree to apply to the query. Optional
        :param output_source: The source of the output to query. "inference" or "demonstration"
        :param limit: The maximum number of inferences to return. Optional
        :param offset: The offset to start from. Optional
        :return: A list of `StoredInference` instances.
        """

    def experimental_render_samples(
        self,
        *,
        stored_samples: Sequence[Union[StoredInference, Datapoint]],
        variants: Dict[str, str],
        concurrency: Optional[int] = None,
    ) -> List[RenderedSample]:
        """
        Render a list of stored samples (datapoints or inferences) into a list of rendered stored samples.

        This function performs two main tasks:
        1. Resolves all network resources (e.g., images) in the stored samples.
        2. Prepares all messages into "simple" messages that have been templated for a particular variant.
            To do this, the function needs to know which variant to use for each function that might appear in the data.

        IMPORTANT: For now, this function drops datapoints that are invalid, such as those where templating fails,
        the function has no variant specified, or the process of downloading resources fails.
        In the future, this behavior may be made configurable by the caller.

        :param stored_samples: A list of stored samples (datapoints or inferences) to render.
        :param variants: A mapping from function name to variant name.
        :param concurrency: Maximum number of samples to process concurrently. Defaults to 100.
        :return: A list of rendered samples.
        """
        ...

    def experimental_launch_optimization(
        self,
        *,
        train_samples: List[RenderedSample],
        val_samples: Optional[List[RenderedSample]] = None,
        optimization_config: OptimizationConfig,
    ) -> OptimizationJobHandle:
        """
        Launch an optimization job.

        :param train_samples: A list of RenderedSample objects that will be used for training.
        :param val_samples: A list of RenderedSample objects that will be used for validation.
        :param optimization_config: The optimization config.
        :return: A `OptimizerJobHandle` object that can be used to poll the optimization job.
        """
        ...

    def experimental_poll_optimization(
        self,
        *,
        job_handle: OptimizationJobHandle,
    ) -> OptimizationJobInfo:
        """
        Poll an optimization job.

        :param job_handle: The job handle returned by `experimental_launch_optimization`.
        :return: An `OptimizerStatus` object.
        """
        ...

    def experimental_run_evaluation(
        self,
        *,
        evaluation_name: str,
        dataset_name: Optional[str] = None,
        datapoint_ids: Optional[List[str]] = None,
        variant_name: Optional[str] = None,
        concurrency: int = 1,
        inference_cache: str = "on",
        internal_dynamic_variant_config: Optional[Dict[str, Any]] = None,
        max_datapoints: Optional[int] = None,
        adaptive_stopping: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> EvaluationJobHandler:
        """
        Run an evaluation for a specific variant on a dataset or specific datapoints.
        This function is only available in EmbeddedGateway mode.

        :param evaluation_name: The name of the evaluation to run
        :param dataset_name: The name of the dataset to use for evaluation (mutually exclusive with datapoint_ids)
        :param datapoint_ids: Specific datapoint IDs to evaluate (mutually exclusive with dataset_name)
        :param variant_name: The name of the variant to evaluate
        :param concurrency: The number of concurrent evaluations to run
        :param inference_cache: Cache configuration for inference requests ("on", "off", "read_only", or "write_only")
        :param internal_dynamic_variant_config: Optional dynamic variant configuration [INTERNAL: This field is unstable and may change without notice.]
        :param max_datapoints: Maximum number of datapoints to evaluate from the dataset
        :param adaptive_stopping: Optional dict configuring adaptive stopping behavior. Example: {"precision": {"exact_match": 0.2, "llm_judge": 0.15}}. The "precision" field maps evaluator names to CI half-width thresholds.
        :return: An EvaluationJobHandler for iterating over evaluation results
        """
        ...

    def close(self) -> None:
        """
        Close the connection to the TensorZero gateway.
        """

    def __enter__(self) -> "TensorZeroGateway": ...
    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None: ...

@final
class AsyncTensorZeroGateway(BaseTensorZeroGateway):
    @classmethod
    def build_http(
        cls,
        *,
        gateway_url: str,
        timeout: Optional[float] = None,
        verbose_errors: bool = False,
        async_setup: bool = True,
        api_key: Optional[str] = None,
    ) -> Union[Awaitable["AsyncTensorZeroGateway"], "AsyncTensorZeroGateway"]:
        """
        Initialize the TensorZero client, using the HTTP gateway.
        :param gateway_url: The base URL of the TensorZero gateway. Example: "http://localhost:3000"
        :param timeout: The timeout for the HTTP client in seconds. If not provided, no timeout will be set.
        :param verbose_errors: If true, the client will increase the detail in errors (increasing the risk of leaking sensitive information).
        :param async_setup (Optional): If True, this method will return a `Future` that resolves to an `AsyncTensorZeroGateway` instance. Otherwise, it will block and return an `AsyncTensorZeroGateway` directly.
        :param api_key: The API key to use for authentication with the TensorZero Gateway. If not provided, the client will attempt to read from the TENSORZERO_API_KEY environment variable.
        :return: An `AsyncTensorZeroGateway` instance configured to use the HTTP gateway.
        """

    @classmethod
    def build_embedded(
        cls,
        *,
        config_file: Optional[str] = None,
        clickhouse_url: Optional[str] = None,
        postgres_url: Optional[str] = None,
        valkey_url: Optional[str] = None,
        timeout: Optional[float] = None,
        async_setup: bool = True,
    ) -> Union[Awaitable["AsyncTensorZeroGateway"], "AsyncTensorZeroGateway"]:
        """
        Build an AsyncTensorZeroGateway instance.

        :param config_file: (Optional) The path to the TensorZero configuration file.
        :param clickhouse_url: (Optional) The URL of the ClickHouse database.
        :param postgres_url: (Optional) The URL of the Postgres database.
        :param timeout: (Optional) The timeout for embedded gateway request processing, in seconds. If this timeout is hit, any in-progress LLM requests may be aborted. If not provided, no timeout will be set.
        :param async_setup (Optional): If True, this method will return a `Future` that resolves to an `AsyncTensorZeroGateway` instance. Otherwise, it will block and return an `AsyncTensorZeroGateway` directly.
        """

    async def inference(  # type: ignore[override]
        self,
        *,
        input: InferenceInput | Dict[str, Any],
        function_name: Optional[str] = None,
        model_name: Optional[str] = None,
        episode_id: Optional[str | UUID | uuid_utils.UUID] = None,
        stream: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
        variant_name: Optional[str] = None,
        dryrun: Optional[bool] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        allowed_tools: Optional[List[str]] = None,
        additional_tools: Optional[List[Dict[str, Any]]] = None,
        provider_tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[Literal["auto", "required", "off"], Dict[Literal["specific"], str]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        internal: Optional[bool] = None,
        tags: Optional[Dict[str, str]] = None,
        credentials: Optional[Dict[str, str]] = None,
        cache_options: Optional[Dict[str, Any]] = None,
        extra_body: Optional[List[ExtraBody | Dict[str, Any]]] = None,
        extra_headers: Optional[List[ExtraHeader | Dict[str, Any]]] = None,
        otlp_traces_extra_headers: Optional[Dict[str, str]] = None,
        otlp_traces_extra_attributes: Optional[Dict[str, str]] = None,
        otlp_traces_extra_resources: Optional[Dict[str, str]] = None,
        include_original_response: Optional[bool] = None,
        include_raw_response: Optional[bool] = None,
        include_raw_usage: Optional[bool] = None,
        internal_dynamic_variant_config: Optional[Dict[str, Any]] = None,
    ) -> Union[InferenceResponse, AsyncIterator[InferenceChunk]]:
        """
        Make a POST request to the /inference endpoint.

        :param function_name: The name of the function to call
        :param input: The input to the function
                      Structure: {"system": Optional[str], "messages": List[{"role": "user" | "assistant", "content": Any}]}
                      The input will be validated server side against the input schema of the function being called.
        :param episode_id: The episode ID to use for the inference.
                           If this is the first inference in an episode, leave this field blank. The TensorZero gateway will generate and return a new episode ID.
                           Note: Only use episode IDs generated by the TensorZero gateway. Don't generate them yourself.
        :param stream: If set, the TensorZero gateway will stream partial message deltas (e.g. generated tokens) as it receives them from model providers.
        :param params: Override inference-time parameters for a particular variant type. Currently, we support:
                        {"chat_completion": {"temperature": float, "max_tokens": int, "seed": int}}
        :param variant_name: If set, pins the inference request to a particular variant.
                             Note: You should generally not do this, and instead let the TensorZero gateway assign a
                             particular variant. This field is primarily used for testing or debugging purposes.
        :param dryrun: If true, the request will be executed but won't be stored to the database.
        :param output_schema: If set, the JSON schema of a JSON function call will be validated against the given JSON Schema.
                              Overrides the output schema configured for the function.
        :param allowed_tools: If set, restricts the tools available during this inference request.
                              The list of names should be a subset of the tools configured for the function.
                              Tools provided at inference time in `additional_tools` (if any) are always available.
        :param additional_tools: A list of additional tools to use for the request. Each element should look like {"name": str, "parameters": valid JSON Schema, "description": str}
        :param provider_tools: A list of provider-specific tools to use for the request. Structure matches provider requirements.
        :param tool_choice: If set, overrides the tool choice strategy for the request.
                            It should be one of: "auto", "required", "off", or {"specific": str}. The last option pins the request to a specific tool name.
        :param parallel_tool_calls: If true, the request will allow for multiple tool calls in a single inference request.
        :param tags: If set, adds tags to the inference request.
        :param extra_body: If set, injects extra fields into the provider request body.
        :param extra_headers: If set, injects extra headers into the provider request.
        :param otlp_traces_extra_headers: If set, adds custom headers to OTLP trace exports. Headers will be automatically prefixed with "tensorzero-otlp-traces-extra-header-".
        :param otlp_traces_extra_attributes: If set, attaches custom HTTP headers to OTLP trace exports for this request.
                                             Headers will be automatically prefixed with "tensorzero-otlp-traces-extra-attributes-".
        :param otlp_traces_extra_resources: If set, attaches custom HTTP headers to OTLP trace exports for this request.
                                            Headers will be automatically prefixed with "tensorzero-otlp-traces-extra-resources-".
        :param include_original_response: DEPRECATED. Use `include_raw_response` instead.
        :param include_raw_response: If set, include raw provider-specific response data from all model inferences.
        :param include_raw_usage: If set, include raw provider-specific usage data in the response.
        :return: If stream is false, returns an InferenceResponse.
                 If stream is true, returns an async iterator that yields InferenceChunks as they come in.
        """

    async def feedback(  # type: ignore[override]
        self,
        *,
        metric_name: str,
        value: Any,
        inference_id: Optional[str | UUID | uuid_utils.UUID] = None,
        episode_id: Optional[str | UUID | uuid_utils.UUID] = None,
        dryrun: Optional[bool] = None,
        internal: Optional[bool] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> FeedbackResponse:
        """
        Make a POST request to the /feedback endpoint.

        :param metric_name: The name of the metric to provide feedback for
        :param value: The value of the feedback. It should correspond to the metric type.
        :param inference_id: The inference ID to assign the feedback to.
                             Only use inference IDs that were returned by the TensorZero gateway.
                             Note: You can assign feedback to either an episode or an inference, but not both.
        :param episode_id: The episode ID to use for the request
                           Only use episode IDs that were returned by the TensorZero gateway.
                           Note: You can assign feedback to either an episode or an inference, but not both.
        :param dryrun: If true, the feedback request will be executed but won't be stored to the database (i.e. no-op).
        :param tags: If set, adds tags to the feedback request.
        :return: {"feedback_id": str}
        """

    async def dynamic_evaluation_run(  # type: ignore[override]
        self,
        *,
        variants: Dict[str, str],
        tags: Optional[Dict[str, str]] = None,
        project_name: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> DynamicEvaluationRunResponse:
        """
        DEPRECATED: Use `workflow_evaluation_run` instead.

        Make a POST request to the /dynamic_evaluation_run endpoint.

        :param variants: A dictionary of variant names to variant values.
        :param tags: A dictionary of tags to add to the dynamic evaluation run.
        :param project_name: The name of the project to use for the dynamic evaluation run.
        :param display_name: The display name of the dynamic evaluation run.
        :return: A `DynamicEvaluationRunResponse` instance ({"run_id": str}).
        """

    async def dynamic_evaluation_run_episode(  # type: ignore[override]
        self,
        *,
        run_id: str | UUID | uuid_utils.UUID,
        task_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> DynamicEvaluationRunEpisodeResponse:
        """
        DEPRECATED: Use `workflow_evaluation_run_episode` instead.

        Make a POST request to the /dynamic_evaluation_run/{run_id}/episode endpoint.

        :param run_id: The run ID to use for the dynamic evaluation run.
        :param task_name: The name of the task to use for the dynamic evaluation run.
        :param tags: A dictionary of tags to add to the dynamic evaluation run.
        :return: A `DynamicEvaluationRunEpisodeResponse` instance ({"episode_id": str}).
        """

    async def workflow_evaluation_run(  # type: ignore[override]
        self,
        *,
        variants: Dict[str, str],
        tags: Optional[Dict[str, str]] = None,
        project_name: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> WorkflowEvaluationRunResponse:
        """
        Make a POST request to the /workflow_evaluation_run endpoint.

        :param variants: A dictionary of variant names to variant values.
        :param tags: A dictionary of tags to add to the workflow evaluation run.
        :param project_name: The name of the project to use for the workflow evaluation run.
        :param display_name: The display name of the workflow evaluation run.
        :return: A `WorkflowEvaluationRunResponse` instance ({"run_id": str}).
        """

    async def workflow_evaluation_run_episode(  # type: ignore[override]
        self,
        *,
        run_id: str | UUID | uuid_utils.UUID,
        task_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> WorkflowEvaluationRunEpisodeResponse:
        """
        Make a POST request to the /workflow_evaluation_run/{run_id}/episode endpoint.

        :param run_id: The run ID to use for the workflow evaluation run.
        :param task_name: The name of the task to use for the workflow evaluation run.
        :param tags: A dictionary of tags to add to the workflow evaluation run.
        :return: A `WorkflowEvaluationRunEpisodeResponse` instance ({"episode_id": str}).
        """

    @deprecated("Deprecated since version 2025.11.4; use `create_datapoints` instead.")
    async def create_datapoints_legacy(
        self,
        *,
        dataset_name: str,
        datapoints: Sequence[Union[ChatDatapointInsert, JsonDatapointInsert]],
    ) -> List[UUID]:
        """
        DEPRECATED: Use `create_datapoints` instead.

        Make a POST request to the /datasets/{dataset_name}/datapoints endpoint.

        :param dataset_name: The name of the dataset to insert the datapoints into.
        :param datapoints: A list of datapoints to insert.
        """

    @deprecated("Deprecated since version 2025.11.4; use `create_datapoints` instead.")
    async def bulk_insert_datapoints(
        self,
        *,
        dataset_name: str,
        datapoints: Sequence[Union[ChatDatapointInsert, JsonDatapointInsert]],
    ) -> List[UUID]:
        """
        DEPRECATED: Use `create_datapoints` instead.

        Make a POST request to the /datasets/{dataset_name}/datapoints/bulk endpoint.

        :param dataset_name: The name of the dataset to insert the datapoints into.
        :param datapoints: A list of datapoints to insert.
        """

    @deprecated("Deprecated since 2025.11.4; use `delete_datapoints` instead.")
    async def delete_datapoint(
        self,
        *,
        dataset_name: str,
        datapoint_id: UUID,
    ) -> None:
        """
        Make a DELETE request to the /datasets/{dataset_name}/datapoints/{datapoint_id} endpoint.

        :param dataset_name: The name of the dataset to delete the datapoint from.
        :param datapoint_id: The ID of the datapoint to delete.
        """

    @deprecated("Deprecated since 2025.11.4; use `list_datapoints` instead.")
    async def list_datapoints_legacy(
        self,
        *,
        dataset_name: str,
        function_name: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Datapoint]:
        """
        Make a GET request to the /datasets/{dataset_name}/datapoints endpoint.

        :param dataset_name: The name of the dataset to list the datapoints from.
        :param function_name: The name of the function to list the datapoints from.
        :param limit: The maximum number of datapoints to return.
        :param offset: The offset to start the list from.
        :return: A list of `Datapoint` instances.
        """

    async def list_datapoints(
        self,
        *,
        dataset_name: str,
        request: ListDatapointsRequest,
    ) -> GetDatapointsResponse:
        """
        Lists datapoints in the dataset.

        :param dataset_name: The name of the dataset to list the datapoints from.
        :param request: The request to list the datapoints.
        :return: A `GetDatapointsResponse` containing the datapoints.
        """

    async def get_datapoint(
        self,
        *,
        dataset_name: str,
        datapoint_id: UUID,
    ) -> LegacyDatapoint:
        """
        Make a GET request to the /datasets/{dataset_name}/datapoints/{datapoint_id} endpoint.

        :param dataset_name: The name of the dataset to get the datapoint from.
        :param datapoint_id: The ID of the datapoint to get.
        :return: A `Datapoint` instance.
        """

    async def create_datapoints(
        self,
        *,
        dataset_name: str,
        requests: Sequence[CreateDatapointRequest],
    ) -> CreateDatapointsResponse:
        """
        Creates new datapoints in the dataset.

        :param dataset_name: The name of the dataset to create the datapoints in.
        :param requests: A list of datapoints to create.
        :return: A CreateDatapointsResponse object containing the IDs of the newly-created datapoints.
        """

    async def update_datapoints(
        self,
        *,
        dataset_name: str,
        requests: Sequence[UpdateDatapointRequest],
    ) -> UpdateDatapointsResponse:
        """
        Update one or more datapoints in a dataset.

        :param dataset_name: The name of the dataset containing the datapoints to update.
        :param requests: A sequence of UpdateDatapointRequest objects.
        :return: A `UpdateDatapointsResponse` object.
        """

    async def get_datapoints(
        self,
        *,
        dataset_name: str | None = ...,
        ids: Sequence[str],
    ) -> GetDatapointsResponse:
        """
        Get specific datapoints by their IDs.

        :param dataset_name: Optional dataset name containing the datapoints. Including this improves
            query performance because the dataset is part of the sorting key.
        :param ids: A sequence of datapoint IDs to retrieve. They should be in UUID format.
        :return: A `GetDatapointsResponse` object.
        """

    async def update_datapoints_metadata(
        self,
        *,
        dataset_name: str,
        requests: Sequence[UpdateDatapointMetadataRequest],
    ) -> UpdateDatapointsResponse:
        """
        Update metadata for one or more datapoints.

        :param dataset_name: The name of the dataset containing the datapoints.
        :param requests: A sequence of UpdateDatapointMetadataRequest objects.
        :return: A `UpdateDatapointsResponse` object.
        """

    async def delete_datapoints(
        self,
        *,
        dataset_name: str,
        ids: Sequence[str],
    ) -> DeleteDatapointsResponse:
        """
        Delete multiple datapoints from a dataset.

        :param dataset_name: The name of the dataset to delete datapoints from.
        :param ids: A sequence of datapoint IDs to delete. They should be in UUID format.
        :return: A `DeleteDatapointsResponse` object.
        """

    async def delete_dataset(
        self,
        *,
        dataset_name: str,
    ) -> DeleteDatapointsResponse:
        """
        Delete a dataset and all of its datapoints.

        :param dataset_name: The name of the dataset to delete.
        :return: A `DeleteDatapointsResponse` object.
        """

    async def create_datapoints_from_inferences(
        self,
        *,
        dataset_name: str,
        params: CreateDatapointsFromInferenceRequestParams,
        output_source: Optional[Literal["none", "inference", "demonstration"]] = None,
    ) -> CreateDatapointsResponse:
        """
        Create datapoints from inferences.

        :param dataset_name: The name of the dataset to create datapoints in.
        :param params: The parameters specifying which inferences to convert to datapoints.
        :param output_source: The source of the output to create datapoints from. "none", "inference", or "demonstration".
                             Can also be specified inside `params.output_source`. If both are provided, an error is raised.
        :return: A `CreateDatapointsResponse` object.
        """

    async def get_inferences(
        self,
        *,
        ids: Sequence[str | UUID | uuid_utils.UUID],
        function_name: Optional[str] = None,
        output_source: str = "inference",
    ) -> GetInferencesResponse:
        """
        Get specific inferences by their IDs.

        :param ids: A sequence of inference IDs to retrieve. They should be in UUID format.
        :param function_name: Optional function name to filter by (improves query performance).
        :param output_source: The source of the output ("inference" or "demonstration"). Default: "inference".
        :return: A `GetInferencesResponse` object.
        """

    async def list_inferences(
        self,
        *,
        request: ListInferencesRequest,
    ) -> GetInferencesResponse:
        """
        List inferences with optional filtering, pagination, and sorting.

        :param request: A `ListInferencesRequest` object with filter parameters.
        :return: A `GetInferencesResponse` object.
        """

    async def experimental_list_inferences(
        self,
        *,
        function_name: str,
        variant_name: Optional[str] = None,
        filters: Optional[InferenceFilter] = None,
        output_source: str = "inference",
        order_by: Optional[List[OrderBy]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[StoredInference]:
        """
        Query the Clickhouse database for inferences.
        This function is only available in EmbeddedGateway mode.

        :param function_name: The name of the function to query.
        :param variant_name: The name of the variant to query. Optional
        :param filters: A filter tree to apply to the query. Optional
        :param output_source: The source of the output to query. "inference" or "demonstration"
        :param limit: The maximum number of inferences to return. Optional
        :param offset: The offset to start from. Optional
        :return: A list of `StoredInference` instances.
        """

    async def experimental_render_samples(
        self,
        *,
        stored_samples: Sequence[Union[StoredInference, Datapoint]],
        variants: Dict[str, str],
        concurrency: Optional[int] = None,
    ) -> List[RenderedSample]:
        """
        Render a list of stored samples into a list of rendered stored samples.

        This function performs two main tasks:
        1. Resolves all network resources (e.g., images) in the stored samples.
        2. Prepares all messages into "simple" messages that have been templated for a particular variant.
           To do this, the function needs to know which variant to use for each function that might appear in the data.

        IMPORTANT: For now, this function drops datapoints that are invalid, such as those where templating fails,
        the function has no variant specified, or the process of downloading resources fails.
        In the future, this behavior may be made configurable by the caller.

        :param stored_samples: A list of stored samples to render.
        :param variants: A mapping from function name to variant name.
        :param concurrency: Maximum number of samples to process concurrently. Defaults to 100.
        :return: A list of rendered samples.
        """

    async def experimental_launch_optimization(
        self,
        *,
        train_samples: List[RenderedSample],
        val_samples: Optional[List[RenderedSample]] = None,
        optimization_config: OptimizationConfig,
    ) -> OptimizationJobHandle:
        """
        Launch an optimization job.

        :param train_samples: A list of RenderedSample objects that will be used for training.
        :param val_samples: A list of RenderedSample objects that will be used for validation.
        :param optimization_config: The optimization config.
        :return: A `OptimizerJobHandle` object that can be used to poll the optimization job.
        """
        ...

    async def experimental_poll_optimization(
        self,
        *,
        job_handle: OptimizationJobHandle,
    ) -> OptimizationJobInfo:
        """
        Poll an optimization job.

        :param job_handle: The job handle returned by `experimental_launch_optimization`.
        :return: An `OptimizerStatus` object.
        """
        ...

    async def experimental_run_evaluation(
        self,
        *,
        evaluation_name: str,
        dataset_name: Optional[str] = None,
        datapoint_ids: Optional[List[str]] = None,
        variant_name: Optional[str] = None,
        concurrency: int = 1,
        inference_cache: str = "on",
        internal_dynamic_variant_config: Optional[Dict[str, Any]] = None,
        max_datapoints: Optional[int] = None,
        adaptive_stopping: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> AsyncEvaluationJobHandler:
        """
        Run an evaluation for a specific variant on a dataset or specific datapoints.
        This function is only available in EmbeddedGateway mode.

        :param evaluation_name: The name of the evaluation to run
        :param dataset_name: The name of the dataset to use for evaluation (mutually exclusive with datapoint_ids)
        :param datapoint_ids: Specific datapoint IDs to evaluate (mutually exclusive with dataset_name)
        :param variant_name: The name of the variant to evaluate
        :param concurrency: The number of concurrent evaluations to run
        :param inference_cache: Cache configuration for inference requests ("on", "off", "read_only", or "write_only")
        :param internal_dynamic_variant_config: Optional dynamic variant configuration [INTERNAL: This field is unstable and may change without notice.]
        :param max_datapoints: Maximum number of datapoints to evaluate from the dataset
        :param adaptive_stopping: Optional dict configuring adaptive stopping behavior. Example: {"precision": {"exact_match": 0.2, "llm_judge": 0.15}}. The "precision" field maps evaluator names to CI half-width thresholds.
        :return: An AsyncEvaluationJobHandler for iterating over evaluation results
        """
        ...

    async def close(self) -> None:
        """
        Close the connection to the TensorZero gateway.
        """

    async def __aenter__(self) -> "AsyncTensorZeroGateway": ...
    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None: ...

# Internal helper method
def _start_http_gateway(
    *,
    config_file: Optional[str],
    clickhouse_url: Optional[str],
    postgres_url: Optional[str],
    valkey_url: Optional[str],
    async_setup: bool,
) -> Union[Any, Awaitable[Any]]: ...
@final
class LocalHttpGateway(object):
    base_url: str

    def close(self) -> None: ...

__all__ = [
    "_start_http_gateway",
    "AsyncEvaluationJobHandler",
    "AsyncTensorZeroGateway",
    "BaseTensorZeroGateway",
    "BestOfNSamplingConfig",
    "ChainOfThoughtConfig",
    "ChatCompletionConfig",
    "Config",
    "LegacyDatapoint",
    "DICLConfig",
    "DICLOptimizationConfig",
    "EvaluationJobHandler",
    "FireworksSFTConfig",
    "FunctionConfigChat",
    "FunctionConfigJson",
    "FunctionsConfig",
    "GCPVertexGeminiSFTConfig",
    "GEPAConfig",
    "LocalHttpGateway",
    "MixtureOfNConfig",
    "OpenAIRFTConfig",
    "OpenAISFTConfig",
    "OptimizationJobHandle",
    "OptimizationJobInfo",
    "OptimizationJobStatus",
    "RenderedSample",
    "ResolvedInput",
    "ResolvedInputMessage",
    "TensorZeroGateway",
    "TogetherSFTConfig",
    "VariantsConfig",
]
