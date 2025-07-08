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

from tensorzero import (
    ChatDatapointInsert,
    ChatInferenceOutput,
    ContentBlock,
    DynamicEvaluationRunEpisodeResponse,
    DynamicEvaluationRunResponse,
    ExtraBody,
    FeedbackResponse,
    InferenceChunk,
    InferenceInput,
    InferenceResponse,
    JsonDatapointInsert,
    OptimizationConfig,
)
from tensorzero.internal import ModelInput, ToolCallConfigDatabaseInsert
from tensorzero.types import (
    InferenceFilterTreeNode,
    JsonInferenceOutput,
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
class StoredInference:
    Chat: Type["StoredInference"]
    Json: Type["StoredInference"]

    def __init__(
        self,
        type: str,
        function_name: str,
        variant_name: str,
        input: Any,
        output: Any,
        episode_id: UUID,
        inference_id: UUID,
        tool_params: Optional[Any] = None,
        output_schema: Optional[Any] = None,
        # Dispreferred outputs are lists because there may be several of them in the future.
        dispreferred_outputs: Union[
            List[ChatInferenceOutput], List[JsonInferenceOutput]
        ] = [],
    ) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def function_name(self) -> str: ...
    @property
    def variant_name(self) -> str: ...
    @property
    def input(self) -> ResolvedInput: ...
    @property
    def output(self) -> Any: ...
    @property
    def episode_id(self) -> Optional[UUID]: ...
    @property
    def inference_id(self) -> Optional[UUID]: ...
    @property
    def tool_params(self) -> Optional[Any]: ...
    @property
    def output_schema(self) -> Optional[Any]: ...
    @property
    def type(self) -> str: ...
    @property
    def dispreferred_outputs(
        self,
    ) -> Union[List[ChatInferenceOutput], List[JsonInferenceOutput]]: ...

@final
class RenderedSample:
    function_name: str
    input: ModelInput
    output: Optional[ChatInferenceOutput]
    episode_id: Optional[UUID]
    inference_id: Optional[UUID]
    tool_params: Optional[ToolCallConfigDatabaseInsert]
    output_schema: Optional[Dict[str, Any]]
    dispreferred_outputs: List[ChatInferenceOutput] = []

@final
class OptimizationJobHandle:
    OpenAISFT: Type["OptimizationJobHandle"]
    FireworksSFT: Type["OptimizationJobHandle"]

@final
class OptimizationJobStatus:
    Pending: Type["OptimizationJobStatus"]
    Completed: Type["OptimizationJobStatus"]
    Failed: Type["OptimizationJobStatus"]

@final
class OptimizationJobInfo:
    OpenAISFT: Type["OptimizationJobInfo"]
    FireworksSFT: Type["OptimizationJobInfo"]
    @property
    def message(self) -> str: ...
    @property
    def status(self) -> Type[OptimizationJobStatus]: ...
    @property
    def output(self) -> Optional[Any]: ...
    @property
    def estimated_finish(self) -> Optional[int]: ...

@final
class OpenAISFTConfig:
    def __init__(
        self,
        *,
        model: str,
        batch_size: Optional[int] = None,
        learning_rate_multiplier: Optional[float] = None,
        n_epochs: Optional[int] = None,
        credentials: Optional[str] = None,
        api_base: Optional[str] = None,
        seed: Optional[int] = None,
        suffix: Optional[str] = None,
    ) -> None: ...

@final
class FireworksSFTConfig:
    def __init__(
        self,
        *,
        model: str,
        credentials: Optional[str] = None,
        account_id: str,
        api_base: Optional[str] = None,
    ) -> None: ...

@final
class Datapoint:
    Chat: Type["Datapoint"]
    Json: Type["Datapoint"]

    @property
    def id(self) -> UUID: ...
    @property
    def input(self) -> ResolvedInput: ...
    @property
    def output(self) -> Any: ...
    @property
    def dataset_name(self) -> str: ...
    @property
    def function_name(self) -> str: ...
    @property
    def tool_params(self) -> Optional[Any]: ...
    @property
    def output_schema(self) -> Optional[Any]: ...

@final
class ChatCompletionConfig:
    @property
    def system_template(self) -> Optional[str]: ...
    @property
    def user_template(self) -> Optional[str]: ...
    @property
    def assistant_template(self) -> Optional[str]: ...

@final
class BestOfNSamplingConfig:
    pass

@final
class DiclConfig:
    pass

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
        DiclConfig,
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
    def __getitem__(
        self, key: str
    ) -> Union[FunctionConfigChat, FunctionConfigJson]: ...

@final
class Config:
    @property
    def functions(self) -> FunctionsConfig: ...

class BaseTensorZeroGateway:
    def experimental_get_config(self) -> Config: ...

@final
class TensorZeroGateway(BaseTensorZeroGateway):
    def __init__(self, base_url: str, *, timeout: Optional[float] = None) -> None:
        """
        Initialize the TensorZero client.

        :param base_url: The base URL of the TensorZero gateway. Example: "http://localhost:3000"
        """

    @classmethod
    def build_http(
        cls,
        *,
        gateway_url: str,
        timeout: Optional[float] = None,
        verbose_errors: bool = False,
    ) -> "TensorZeroGateway":
        """
        Initialize the TensorZero client, using the HTTP gateway.
        :param gateway_url: The base URL of the TensorZero gateway. Example: "http://localhost:3000"
        :param timeout: The timeout for the HTTP client in seconds. If not provided, no timeout will be set.
        :param verbose_errors: If true, the client will increase the detail in errors (increasing the risk of leaking sensitive information).
        :return: A `TensorZeroGateway` instance configured to use the HTTP gateway.
        """

    @classmethod
    def build_embedded(
        cls,
        *,
        config_file: Optional[str] = None,
        clickhouse_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> "TensorZeroGateway":
        """
        Build a TensorZeroGateway instance.

        :param config_file: (Optional) The path to the TensorZero configuration file.
        :param clickhouse_url: (Optional) The URL of the ClickHouse database.
        :param timeout: The timeout for embedded gateway request processing, in seconds. If this timeout is hit, any in-progress LLM requests may be aborted. If not provided, no timeout will be set.
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
        tool_choice: Optional[
            Union[Literal["auto", "required", "off"], Dict[Literal["specific"], str]]
        ] = None,
        parallel_tool_calls: Optional[bool] = None,
        internal: Optional[bool] = None,
        tags: Optional[Dict[str, str]] = None,
        credentials: Optional[Dict[str, str]] = None,
        cache_options: Optional[Dict[str, Any]] = None,
        extra_body: Optional[List[ExtraBody | Dict[str, Any]]] = None,
        extra_headers: Optional[List[Dict[str, Any]]] = None,
        include_original_response: Optional[bool] = None,
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
        :param tool_choice: If set, overrides the tool choice strategy for the request.
                            It should be one of: "auto", "required", "off", or {"specific": str}. The last option pins the request to a specific tool name.
        :param parallel_tool_calls: If true, the request will allow for multiple tool calls in a single inference request.
        :param tags: If set, adds tags to the inference request.
        :param extra_body: If set, injects extra fields into the provider request body.
        :param extra_headers: If set, injects extra headers into the provider request.
        :param include_original_response: If set, add an `original_response` field to the response, containing the raw string response from the model.
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
        datapoint_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> DynamicEvaluationRunEpisodeResponse:
        """
        Make a POST request to the /dynamic_evaluation_run_episode endpoint.

        :param run_id: The run ID to use for the dynamic evaluation run.
        :param task_name: The name of the task to use for the dynamic evaluation run.
        :param datapoint_name: The name of the datapoint to use for the dynamic evaluation run.
                    Deprecated: use `task_name` instead.
        :param tags: A dictionary of tags to add to the dynamic evaluation run.
        :return: A `DynamicEvaluationRunEpisodeResponse` instance ({"episode_id": str}).
        """

    def bulk_insert_datapoints(
        self,
        *,
        dataset_name: str,
        datapoints: List[Union[ChatDatapointInsert, JsonDatapointInsert]],
    ) -> List[UUID]:
        """
        Make a POST request to the /datasets/{dataset_name}/datapoints/bulk endpoint.

        :param dataset_name: The name of the dataset to insert the datapoints into.
        :param datapoints: A list of datapoints to insert.
        """

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

    def list_datapoints(
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

    def get_datapoint(
        self,
        *,
        dataset_name: str,
        datapoint_id: UUID,
    ) -> Datapoint:
        """
        Make a GET request to the /datasets/{dataset_name}/datapoints/{datapoint_id} endpoint.

        :param dataset_name: The name of the dataset to get the datapoint from.
        :param datapoint_id: The ID of the datapoint to get.
        :return: A `Datapoint` instance.
        """

    def experimental_list_inferences(
        self,
        *,
        function_name: str,
        variant_name: Optional[str] = None,
        filters: Optional[InferenceFilterTreeNode] = None,
        output_source: str = "inference",
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

    def experimental_render_inferences(
        self,
        *,
        stored_inferences: List[StoredInference],
        variants: Dict[str, str],
    ) -> List[RenderedSample]:
        """
        DEPRECATED: use `experimental_render_samples` instead.
        Render a list of stored samples into a list of rendered stored samples.

        This function performs two main tasks:
        1. Resolves all network resources (e.g., images) in the stored samples.
        2. Prepares all messages into "simple" messages that have been templated for a particular variant.
           To do this, the function needs to know which variant to use for each function that might appear in the data.

        IMPORTANT: For now, this function drops datapoints that are invalid, such as those where templating fails,
        the function has no variant specified, or the process of downloading resources fails.
        In the future, this behavior may be made configurable by the caller.

        :param stored_inferences: A list of stored samples to render.
        :param variants: A mapping from function name to variant name.
        :return: A list of rendered samples.
        """

    def experimental_render_samples(
        self,
        *,
        stored_samples: Sequence[Union[StoredInference, Datapoint]],
        variants: Dict[str, str],
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
    def __init__(self, base_url: str, *, timeout: Optional[float] = None) -> None:
        """
        Initialize the TensorZero client.

        :param base_url: The base URL of the TensorZero gateway. Example: "http://localhost:3000"
        """

    @classmethod
    def build_http(
        cls,
        *,
        gateway_url: str,
        timeout: Optional[float] = None,
        verbose_errors: bool = False,
        async_setup: bool = True,
    ) -> Union[Awaitable["AsyncTensorZeroGateway"], "AsyncTensorZeroGateway"]:
        """
        Initialize the TensorZero client, using the HTTP gateway.
        :param gateway_url: The base URL of the TensorZero gateway. Example: "http://localhost:3000"
        :param timeout: The timeout for the HTTP client in seconds. If not provided, no timeout will be set.
        :param verbose_errors: If true, the client will increase the detail in errors (increasing the risk of leaking sensitive information).
        :param async_setup (Optional): If True, this method will return a `Future` that resolves to an `AsyncTensorZeroGateway` instance. Otherwise, it will block and return an `AsyncTensorZeroGateway` directly.
        :return: An `AsyncTensorZeroGateway` instance configured to use the HTTP gateway.
        """

    @classmethod
    def build_embedded(
        cls,
        *,
        config_file: Optional[str] = None,
        clickhouse_url: Optional[str] = None,
        timeout: Optional[float] = None,
        async_setup: bool = True,
    ) -> Union[Awaitable["AsyncTensorZeroGateway"], "AsyncTensorZeroGateway"]:
        """
        Build an AsyncTensorZeroGateway instance.

        :param config_file: (Optional) The path to the TensorZero configuration file.
        :param clickhouse_url: (Optional) The URL of the ClickHouse database.
        :param timeout: The timeout for embedded gateway request processing, in seconds. If this timeout is hit, any in-progress LLM requests may be aborted. If not provided, no timeout will be set.
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
        tool_choice: Optional[
            Union[Literal["auto", "required", "off"], Dict[Literal["specific"], str]]
        ] = None,
        parallel_tool_calls: Optional[bool] = None,
        internal: Optional[bool] = None,
        tags: Optional[Dict[str, str]] = None,
        credentials: Optional[Dict[str, str]] = None,
        cache_options: Optional[Dict[str, Any]] = None,
        extra_body: Optional[List[ExtraBody | Dict[str, Any]]] = None,
        extra_headers: Optional[List[Dict[str, Any]]] = None,
        include_original_response: Optional[bool] = None,
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
        :param tool_choice: If set, overrides the tool choice strategy for the request.
                            It should be one of: "auto", "required", "off", or {"specific": str}. The last option pins the request to a specific tool name.
        :param parallel_tool_calls: If true, the request will allow for multiple tool calls in a single inference request.
        :param tags: If set, adds tags to the inference request.
        :param extra_body: If set, injects extra fields into the provider request body.
        :param extra_headers: If set, injects extra headers into the provider request.
        :param include_original_response: If set, add an `original_response` field to the response, containing the raw string response from the model.
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
        datapoint_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> DynamicEvaluationRunEpisodeResponse:
        """
        Make a POST request to the /dynamic_evaluation_run_episode endpoint.

        :param run_id: The run ID to use for the dynamic evaluation run.
        :param task_name: The name of the task to use for the dynamic evaluation run.
        :param datapoint_name: The name of the datapoint to use for the dynamic evaluation run.
                    Deprecated: use `task_name` instead.
        :param tags: A dictionary of tags to add to the dynamic evaluation run.
        :return: A `DynamicEvaluationRunEpisodeResponse` instance ({"episode_id": str}).
        """

    async def bulk_insert_datapoints(
        self,
        *,
        dataset_name: str,
        datapoints: List[Union[ChatDatapointInsert, JsonDatapointInsert]],
    ) -> List[UUID]:
        """
        Make a POST request to the /datasets/{dataset_name}/datapoints/bulk endpoint.

        :param dataset_name: The name of the dataset to insert the datapoints into.
        :param datapoints: A list of datapoints to insert.
        """

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

    async def list_datapoints(
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

    async def get_datapoint(
        self,
        *,
        dataset_name: str,
        datapoint_id: UUID,
    ) -> Datapoint:
        """
        Make a GET request to the /datasets/{dataset_name}/datapoints/{datapoint_id} endpoint.

        :param dataset_name: The name of the dataset to get the datapoint from.
        :param datapoint_id: The ID of the datapoint to get.
        :return: A `Datapoint` instance.
        """

    async def experimental_list_inferences(
        self,
        *,
        function_name: str,
        variant_name: Optional[str] = None,
        filters: Optional[InferenceFilterTreeNode] = None,
        output_source: str = "inference",
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

    async def experimental_render_inferences(
        self,
        *,
        stored_inferences: List[StoredInference],
        variants: Dict[str, str],
    ) -> List[RenderedSample]:
        """
        DEPRECATED: use `experimental_render_samples` instead.

        Render a list of stored samples into a list of rendered stored samples.

        This function performs two main tasks:
        1. Resolves all network resources (e.g., images) in the stored samples.
        2. Prepares all messages into "simple" messages that have been templated for a particular variant.
           To do this, the function needs to know which variant to use for each function that might appear in the data.

        IMPORTANT: For now, this function drops datapoints that are invalid, such as those where templating fails,
        the function has no variant specified, or the process of downloading resources fails.
        In the future, this behavior may be made configurable by the caller.

        :param stored_inferences: A list of stored samples to render.
        :param variants: A mapping from function name to variant name.
        :return: A list of rendered samples.
        """

    async def experimental_render_samples(
        self,
        *,
        stored_samples: Sequence[Union[StoredInference, Datapoint]],
        variants: Dict[str, str],
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
    async_setup: bool,
) -> Union[Any, Awaitable[Any]]: ...
@final
class LocalHttpGateway(object):
    base_url: str

    def close(self) -> None: ...

__all__ = [
    "AsyncTensorZeroGateway",
    "BaseTensorZeroGateway",
    "BestOfNSamplingConfig",
    "ChatCompletionConfig",
    "ChainOfThoughtConfig",
    "Config",
    "Datapoint",
    "DiclConfig",
    "FunctionConfigChat",
    "FunctionConfigJson",
    "FunctionsConfig",
    "FireworksSFTConfig",
    "TensorZeroGateway",
    "LocalHttpGateway",
    "MixtureOfNConfig",
    "_start_http_gateway",
    "OpenAISFTConfig",
    "OptimizationJobHandle",
    "OptimizationJobInfo",
    "OptimizationJobStatus",
    "RenderedSample",
    "StoredInference",
    "ResolvedInput",
    "ResolvedInputMessage",
    "VariantsConfig",
]
