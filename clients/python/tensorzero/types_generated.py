class _UnsetType:
    """Sentinel value to distinguish between omitted fields and null values."""

    def __repr__(self):
        return "UNSET"


UNSET = _UnsetType()
"""
Sentinel value to distinguish between omitted and null in API requests.

Usage:
- UNSET: Field is omitted (don't change existing value)
- None: Field is explicitly set to null
- value: Field is set to the provided value
"""


from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal


@dataclass
class ContentBlockChatOutput4:
    """
    Defines the types of content block that can come from a `chat` function
    """

    data: Any
    type: Literal["unknown"] = "unknown"
    model_provider_name: str | None = None


class CreateDatapointsFromInferenceOutputSource(Enum):
    """
    Specifies the source of the output for the datapoint when creating datapoints from inferences.
    - `None`: Do not include any output in the datapoint.
    - `Inference`: Include the original inference output in the datapoint.
    - `Demonstration`: Include the latest demonstration feedback as output in the datapoint.
    """

    none = "none"
    inference = "inference"
    demonstration = "demonstration"


@dataclass
class CreateDatapointsResponse:
    """
    Response from creating datapoints.
    """

    ids: list[str]


@dataclass
class DatapointMetadataUpdate:
    """
    A request to update the metadata of a datapoint.
    """

    name: str | None | _UnsetType = UNSET


@dataclass
class DeleteDatapointsRequest:
    """
    Request to delete datapoints from a dataset.
    """

    ids: list[str]


@dataclass
class DeleteDatapointsResponse:
    """
    Response containing the number of deleted datapoints.
    """

    num_deleted_datapoints: int


class Detail(Enum):
    """
    Detail level for input images (affects fidelity and token cost)
    """

    low = "low"
    high = "high"
    auto = "auto"


@dataclass
class GetDatapointsRequest:
    """
    Request to get specific datapoints by their IDs.
    Used by the `POST /v1/datasets/get_datapoints` endpoint.
    """

    ids: list[str]


@dataclass
class InferenceResponseToolCall:
    """
    An InferenceResponseToolCall is a request by a model to call a Tool
    in the form that we return to the client / ClickHouse
    """

    id: str
    raw_arguments: str
    raw_name: str
    arguments: Any | None = None
    name: str | None = None


@dataclass
class JsonDatapointOutputUpdate:
    """
    A request to update the output of a JSON datapoint.
    We intentionally only accept the `raw` field (in a JSON-serialized string), because datapoints can contain invalid outputs, and it's desirable
    for users to run evals against them.
    """

    raw: str


@dataclass
class ObjectStoragePointer:
    """
    A file stored in an object storage backend, without data.
    This struct can be stored in the database. It's used by `StoredFile` (`StoredInput`).
    Note: `File` supports both `ObjectStorageFilePointer` and `ObjectStorageFile`.
    """

    mime_type: str
    storage_path: dict[str, Any]
    detail: str | None = None
    source_url: str | None = None


@dataclass
class ProviderToolScope1:
    model_name: str
    model_provider_name: str


@dataclass
class RawText:
    """
    Struct that represents raw text content that should be passed directly to the model
    without any template processing or validation
    """

    value: str


class Role(Enum):
    user = "user"
    assistant = "assistant"


@dataclass
class Template:
    arguments: dict[str, Any]
    name: str


@dataclass
class Text:
    """
    InputMessages are validated against the input schema of the Function
    and then templated and transformed into RequestMessages for a particular Variant.
    They might contain tool calls or tool results along with text.
    The abstraction we use to represent this is ContentBlock, which is a union of Text, ToolCall, and ToolResult.
    ContentBlocks are collected into RequestMessages.
    These RequestMessages are collected into a ModelInferenceRequest,
    which should contain all information needed by a ModelProvider to perform the
    inference that is called for.
    """

    text: str


@dataclass
class ThoughtSummaryBlock1:
    text: str
    type: Literal["summary_text"] = "summary_text"


ThoughtSummaryBlock = ThoughtSummaryBlock1


@dataclass
class Tool:
    """
    A Tool object describes how a tool can be dynamically configured by the user.
    """

    description: str
    name: str
    parameters: Any
    strict: bool | None = None


@dataclass
class ToolCall:
    arguments: str
    id: str
    name: str


ToolCallWrapper = ToolCall | InferenceResponseToolCall


@dataclass
class ToolChoice1:
    """
    Most inference providers allow the user to force a tool to be used
    and even specify which tool to be used.

    This enum is used to denote this tool choice.
    """

    specific: str


@dataclass
class ToolResult:
    """
    A ToolResult is the outcome of a ToolCall, which we may want to present back to the model
    """

    id: str
    name: str
    result: str


@dataclass
class Unknown:
    """
    Struct that represents an unknown provider-specific content block.
    We pass this along as-is without any validation or transformation.
    """

    data: Any
    model_provider_name: str | None = None


@dataclass
class UpdateDatapointMetadataRequest:
    """
    A request to update the metadata of a single datapoint.
    """

    id: str
    metadata: DatapointMetadataUpdate


@dataclass
class UpdateDatapointsMetadataRequest:
    """
    Request to update metadata for one or more datapoints in a dataset.
    Used by the `PATCH /v1/datasets/{dataset_id}/datapoints/metadata` endpoint.
    """

    datapoints: list[UpdateDatapointMetadataRequest]


@dataclass
class UpdateDatapointsResponse:
    """
    A response to a request to update one or more datapoints in a dataset.
    """

    ids: list[str]


@dataclass
class UrlFile:
    """
    A file that can be located at a URL
    """

    url: str
    detail: str | None = None
    mime_type: str | None = None


@dataclass
class Base64File:
    """
    A file already encoded as base64
    """

    data: str
    mime_type: str
    detail: Detail | None = None
    source_url: str | None = None


@dataclass
class ContentBlockChatOutput1(Text):
    """
    Defines the types of content block that can come from a `chat` function
    """

    type: Literal["text"] = "text"


@dataclass
class ContentBlockChatOutput2(InferenceResponseToolCall):
    """
    Defines the types of content block that can come from a `chat` function
    """

    type: Literal["tool_call"] = "tool_call"


@dataclass
class File1(UrlFile):
    """
    A file for an inference or a datapoint.
    """

    file_type: Literal["url"] = "url"


@dataclass
class File2(Base64File):
    """
    A file for an inference or a datapoint.
    """

    file_type: Literal["base64"] = "base64"


@dataclass
class File3(ObjectStoragePointer):
    """
    A file for an inference or a datapoint.
    """

    file_type: Literal["object_storage_pointer"] = "object_storage_pointer"


@dataclass
class InputMessageContent1(Text):
    type: Literal["text"] = "text"


@dataclass
class InputMessageContent2(Template):
    type: Literal["template"] = "template"


@dataclass
class InputMessageContent3:
    type: Literal["tool_call"] = "tool_call"


@dataclass
class InputMessageContent4(ToolResult):
    type: Literal["tool_result"] = "tool_result"


@dataclass
class InputMessageContent5(RawText):
    type: Literal["raw_text"] = "raw_text"


@dataclass
class InputMessageContent8(Unknown):
    """
    An unknown content block type, used to allow passing provider-specific
    content blocks (e.g. Anthropic's `redacted_thinking`) in and out
    of TensorZero.
    The `data` field holds the original content block from the provider,
    without any validation or transformation by TensorZero.
    """

    type: Literal["unknown"] = "unknown"


@dataclass
class ObjectStorageError(ObjectStoragePointer):
    """
    A file that we failed to read from object storage.
    This struct can NOT be stored in the database.
    """

    error: str | None = None


@dataclass
class ObjectStorageFile(ObjectStoragePointer):
    """
    A file stored in an object storage backend, with data.
    This struct can NOT be stored in the database.
    Note: `File` supports both `ObjectStorageFilePointer` and `ObjectStorageFile`.
    """

    data: str


@dataclass
class ProviderTool:
    tool: Any
    scope: ProviderToolScope1 | None = None


@dataclass
class Thought:
    """
    Struct that represents a model's reasoning
    """

    field_internal_provider_type: str | None = None
    signature: str | None = None
    summary: list[ThoughtSummaryBlock] | None = None
    text: str | None = None


@dataclass
class ContentBlockChatOutput3(Thought):
    """
    Defines the types of content block that can come from a `chat` function
    """

    type: Literal["thought"] = "thought"


@dataclass
class DynamicToolParams:
    """
    Wire/API representation of dynamic tool parameters for inference requests.

    This type is the **wire format** for tool configurations used in API requests and responses.
    It distinguishes between static tools (configured in the function) and dynamic tools
    (provided at runtime), allowing clients to reference pre-configured tools by name or
    provide new tools on-the-fly.

    # Purpose
    - Accept tool parameters in inference API requests (e.g., `/inference/{function_name}`)
    - Expose tool configurations in API responses for stored inferences
    - Support Python and TypeScript client bindings
    - Allow runtime customization of tool behavior

    # Fields
    - `allowed_tools`: Names of static tools from function config to use (subset selection)
    - `additional_tools`: New tools defined at runtime (not in static config)
    - `tool_choice`: Override the function's default tool choice strategy
    - `parallel_tool_calls`: Override whether parallel tool calls are enabled
    - `provider_tools`: Provider-specific tool configurations (not persisted to database)

    # Key Differences from ToolCallConfigDatabaseInsert
    - **Separate lists**: Maintains distinction between static (`allowed_tools`) and dynamic (`additional_tools`) tools
    - **By reference**: Static tools referenced by name, not duplicated
    - **Has provider_tools**: Can specify provider-specific tool configurations
    - **Has bindings**: Exposed to Python/TypeScript via `pyo3` and `ts_rs`

    # Conversion to Storage Format
    Converting from `DynamicToolParams` to `ToolCallConfigDatabaseInsert` is a **lossy** operation:
    1. Static tools (from `allowed_tools` names) are resolved from function config
    2. Dynamic tools (from `additional_tools`) are included as-is
    3. Both lists are merged into a single `tools_available` list
    4. The distinction between static and dynamic tools is lost
    5. `provider_tools` are dropped (not stored)

    Use `FunctionConfig::dynamic_tool_params_to_database_insert()` for this conversion.

    # Conversion from Storage Format
    Converting from `ToolCallConfigDatabaseInsert` back to `DynamicToolParams` attempts to reconstruct the original:
    1. Tools that match function config tool names → `allowed_tools`
    2. Tools that don't match function config → `additional_tools`
    3. `provider_tools` is set to `None` (cannot be recovered)

    Use `FunctionConfig::database_insert_to_dynamic_tool_params()` for this conversion.

    # Example
    ```rust,ignore
    // API request with dynamic tool params
    let params = DynamicToolParams {
        allowed_tools: Some(vec!["calculator".to_string()]),  // Use only the calculator tool from config
        additional_tools: Some(vec![Tool {  runtime tool  }]),  // Add a new tool
        tool_choice: Some(ToolChoice::Required),
        parallel_tool_calls: Some(true),
        provider_tools: None,
    };

    // Convert to storage format (merge tools, lose distinction)
    let db_insert = function_config
        .dynamic_tool_params_to_database_insert(params, &static_tools)?
        .unwrap_or_default();

    // db_insert.tools_available now contains both the calculator tool (from config)
    // and the runtime tool (from additional_tools), merged together
    ```

    See also: [`ToolCallConfigDatabaseInsert`] for the storage/database format
    """

    additional_tools: list[Tool] | None = None
    allowed_tools: list[str] | None = None
    parallel_tool_calls: bool | None = None
    provider_tools: list[ProviderTool] | None = None
    tool_choice: (
        Literal["none"] | Literal["auto"] | Literal["required"] | ToolChoice1 | None
    ) = None


@dataclass
class File4(ObjectStorageFile):
    """
    A file for an inference or a datapoint.
    """

    file_type: Literal["object_storage"] = "object_storage"


@dataclass
class File5(ObjectStorageError):
    """
    A file for an inference or a datapoint.
    """

    file_type: Literal["object_storage_error"] = "object_storage_error"


File = File1 | File2 | File3 | File4 | File5


@dataclass
class InputMessageContent6(Thought):
    type: Literal["thought"] = "thought"


@dataclass
class InputMessageContent7:
    type: Literal["file"] = "file"


@dataclass
class InputMessage:
    """
    InputMessage and Role are our representation of the input sent by the client
    prior to any processing into LLM representations below.
    `InputMessage` has a custom deserializer that addresses legacy data formats that we used to support (see input_message.rs).
    """

    content: list[
        InputMessageContent1
        | InputMessageContent2
        | InputMessageContent3
        | InputMessageContent4
        | InputMessageContent5
        | InputMessageContent6
        | InputMessageContent7
        | InputMessageContent8
    ]
    role: Role


@dataclass
class Input:
    """
    A request is made that contains an Input
    """

    messages: list[InputMessage] | None = None
    system: str | dict[str, Any] | None = None


@dataclass
class UpdateChatDatapointRequest:
    """
    An update request for a chat datapoint.
    For any fields that are optional in ChatInferenceDatapoint, the request field distinguishes between an omitted field, `null`, and a value:
    - If the field is omitted, it will be left unchanged.
    - If the field is specified as `null`, it will be set to `null`.
    - If the field has a value, it will be set to the provided value.

    In Rust this is modeled as an `Option<Option<T>>`, where `None` means "unchanged" and `Some(None)` means "set to `null`" and `Some(Some(T))` means "set to the provided value".
    """

    id: str
    input: Input | None = None
    metadata: DatapointMetadataUpdate | None = None
    output: (
        list[
            ContentBlockChatOutput1
            | ContentBlockChatOutput2
            | ContentBlockChatOutput3
            | ContentBlockChatOutput4
        ]
        | None
    ) = None
    tags: dict[str, Any] | None = None
    tool_params: DynamicToolParams | None | _UnsetType = UNSET


@dataclass
class UpdateDatapointRequest1(UpdateChatDatapointRequest):
    """
    Request to update a chat datapoint.
    """

    type: Literal["chat"] = "chat"


@dataclass
class UpdateJsonDatapointRequest:
    """
    An update request for a JSON datapoint.
    For any fields that are optional in JsonInferenceDatapoint, the request field distinguishes between an omitted field, `null`, and a value:
    - If the field is omitted, it will be left unchanged.
    - If the field is specified as `null`, it will be set to `null`.
    - If the field has a value, it will be set to the provided value.

    In Rust this is modeled as an `Option<Option<T>>`, where `None` means "unchanged" and `Some(None)` means "set to `null`" and `Some(Some(T))` means "set to the provided value".
    """

    id: str
    input: Input | None = None
    metadata: DatapointMetadataUpdate | None = None
    output: JsonDatapointOutputUpdate | None | _UnsetType = UNSET
    output_schema: Any | None = None
    tags: dict[str, Any] | None = None


@dataclass
class CreateChatDatapointRequest(DynamicToolParams):
    """
    A request to create a chat datapoint.
    """

    function_name: str
    input: Input
    episode_id: str | None = None
    name: str | None = None
    output: (
        list[
            ContentBlockChatOutput1
            | ContentBlockChatOutput2
            | ContentBlockChatOutput3
            | ContentBlockChatOutput4
        ]
        | None
    ) = None
    tags: dict[str, Any] | None = None


@dataclass
class CreateDatapointRequest1(CreateChatDatapointRequest):
    """
    Request to create a chat datapoint.
    """

    type: Literal["chat"] = "chat"


@dataclass
class CreateJsonDatapointRequest:
    """
    A request to create a JSON datapoint.
    """

    function_name: str
    input: Input
    episode_id: str | None = None
    name: str | None = None
    output: JsonDatapointOutputUpdate | None = None
    output_schema: Any | None = None
    tags: dict[str, Any] | None = None


@dataclass
class UpdateDatapointRequest2(UpdateJsonDatapointRequest):
    """
    Request to update a JSON datapoint.
    """

    type: Literal["json"] = "json"


@dataclass
class UpdateDatapointsRequest:
    """
    Request to update one or more datapoints in a dataset.
    """

    datapoints: list[UpdateDatapointRequest1 | UpdateDatapointRequest2]


@dataclass
class CreateDatapointRequest2(CreateJsonDatapointRequest):
    """
    Request to create a JSON datapoint.
    """

    type: Literal["json"] = "json"


@dataclass
class CreateDatapointsRequest:
    """
    Request to create datapoints manually.
    Used by the `POST /v1/datasets/{dataset_id}/datapoints` endpoint.
    """

    datapoints: list[CreateDatapointRequest1 | CreateDatapointRequest2]
