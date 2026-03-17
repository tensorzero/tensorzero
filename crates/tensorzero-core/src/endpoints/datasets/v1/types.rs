use std::collections::HashMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tensorzero_derive::TensorZeroDeserialize;
use tensorzero_derive::export_schema;
use uuid::Uuid;

pub use crate::db::clickhouse::query_builder::{
    DatapointFilter, InferenceFilter, OrderBy, OrderByTerm, OrderDirection, TagFilter, TimeFilter,
};
use crate::db::inferences::InferenceOutputSource;
use crate::endpoints::datasets::Datapoint;
use crate::endpoints::stored_inferences::v1::types::ListInferencesRequest;
use crate::inference::types::{ContentBlockChatOutput, Input};
use crate::serde_util::deserialize_double_option;
use crate::tool::{DynamicToolParams, ProviderTool, Tool, ToolChoice};

/// The property to order datapoints by.
/// This is flattened in the public API inside the `DatapointOrderBy` struct.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema, PartialEq)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(tag = "by", rename_all = "snake_case")]
pub enum DatapointOrderByTerm {
    /// Creation timestamp of the datapoint.
    #[schemars(title = "DatapointOrderByTimestamp")]
    Timestamp,

    /// Relevance score of the search query in the input and output of the datapoint.
    /// Requires a search query (experimental). If it's not provided, we return an error.
    ///
    /// NOTE: Relevance ordering is not yet implemented for Postgres and currently
    /// falls back to id ordering. See TODO(#6441).
    #[schemars(title = "DatapointOrderBySearchRelevance")]
    SearchRelevance,
}

/// Order by clauses for querying datapoints.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema, PartialEq)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[export_schema]
pub struct DatapointOrderBy {
    /// The property to order by.
    #[serde(flatten)]
    pub term: DatapointOrderByTerm,

    /// The ordering direction.
    pub direction: OrderDirection,
}

/// Request to update one or more datapoints in a dataset.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[export_schema]
pub struct UpdateDatapointsRequest {
    /// The datapoints to update.
    pub datapoints: Vec<UpdateDatapointRequest>,
}

/// A tagged request to update a single datapoint in a dataset.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, JsonSchema, Serialize, TensorZeroDeserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[cfg_attr(
    feature = "ts-bindings",
    ts(export, tag = "type", rename_all = "snake_case")
)]
#[export_schema]
pub enum UpdateDatapointRequest {
    /// Request to update a chat datapoint.
    #[schemars(title = "UpdateChatDatapointRequest")]
    Chat(UpdateChatDatapointRequest),
    /// Request to update a JSON datapoint.
    #[schemars(title = "UpdateJsonDatapointRequest")]
    Json(UpdateJsonDatapointRequest),
}

/// An update request for a chat datapoint.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
#[export_schema]
#[schemars(title = "UpdateChatDatapointRequestInternal")]
pub struct UpdateChatDatapointRequest {
    /// The ID of the datapoint to update. Required.
    pub id: Uuid,

    /// Datapoint input. If omitted, it will be left unchanged.
    #[serde(default)]
    pub input: Option<Input>,

    /// Chat datapoint output. If omitted, it will be left unchanged. If specified as `null`, it will be set to
    /// `null`. Otherwise, it will overwrite the existing output (and can be an empty array).
    #[serde(default, deserialize_with = "deserialize_double_option")]
    #[schemars(extend("x-double-option" = true), description = "Chat datapoint output.

If omitted (which uses the default value `OMIT`), it will be left unchanged. If set to `None`, it will be cleared.
Otherwise, it will overwrite the existing output (and can be an empty list).")]
    pub output: Option<Option<Vec<ContentBlockChatOutput>>>,

    /// Datapoint tool parameters.
    #[serde(flatten)]
    pub tool_params: UpdateDynamicToolParamsRequest,

    /// Datapoint tags. If omitted, it will be left unchanged. If empty, it will be cleared. Otherwise,
    /// it will be overwrite the existing tags.
    #[serde(default)]
    #[schemars(description = "Datapoint tags.

If omitted (which uses the default value `OMIT`), it will be left unchanged. If set to `None`, it will be cleared.
Otherwise, it will overwrite the existing tags.")]
    pub tags: Option<HashMap<String, String>>,

    /// Metadata fields to update.
    #[serde(flatten)]
    pub metadata: DatapointMetadataUpdate,
}

/// A request to update the dynamic tool parameters of a datapoint.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
#[export_schema]
pub struct UpdateDynamicToolParamsRequest {
    /// A subset of static tools configured for the function that the inference is explicitly allowed to use.
    /// If omitted, it will be left unchanged. If specified as `null`, it will be cleared (we allow function-configured tools plus additional tools
    /// provided at inference time). If specified as a value, it will be set to the provided value.
    #[serde(default, deserialize_with = "deserialize_double_option")]
    #[schemars(extend("x-double-option" = true), description = "A subset of static tools configured for the function that the inference is explicitly allowed to use.

If omitted (which uses the default value `OMIT`), it will be left unchanged. If set to `None`, it will be cleared (we allow function-configured tools
plus additional tools provided at inference time). If specified as a value, it will be set to the provided value.")]
    pub allowed_tools: Option<Option<Vec<String>>>,

    /// Tools that the user provided at inference time (not in function config), in addition to the function-configured tools, that are also allowed.
    /// Modifying `additional_tools` DOES NOT automatically modify `allowed_tools`; `allowed_tools` must be explicitly updated to include
    /// new tools or exclude removed tools.
    /// If omitted, it will be left unchanged. If specified as a value, it will be set to the provided value.
    pub additional_tools: Option<Vec<Tool>>,

    /// User-specified tool choice strategy.
    /// If omitted, it will be left unchanged. If specified as `null`, we will clear the dynamic tool choice and use function-configured tool choice.
    #[serde(default, deserialize_with = "deserialize_double_option")]
    #[schemars(extend("x-double-option" = true), description = "User-specified tool choice strategy.

If omitted (which uses the default value `OMIT`), it will be left unchanged. If set to `None`, it will be cleared (we will use function-configured
tool choice). If specified as a value, it will be set to the provided value.")]
    pub tool_choice: Option<Option<ToolChoice>>,

    /// Whether to use parallel tool calls in the inference.
    /// If omitted, it will be left unchanged. If specified as `null`, it will be set to `null`. If specified as a value, it will be set to the provided value.
    #[serde(default, deserialize_with = "deserialize_double_option")]
    #[schemars(extend("x-double-option" = true), description = "Whether to use parallel tool calls in the inference.

If omitted (which uses the default value `OMIT`), it will be left unchanged. If set to `None`, it will be cleared (we will use function-configured
parallel tool calls). If specified as a value, it will be set to the provided value.")]
    pub parallel_tool_calls: Option<Option<bool>>,

    /// Provider-specific tool configurations
    /// If omitted, it will be left unchanged. If specified as a value, it will be set to the provided value.
    pub provider_tools: Option<Vec<ProviderTool>>,
}

/// An update request for a JSON datapoint.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
#[export_schema]
#[schemars(title = "UpdateJsonDatapointRequestInternal")]
pub struct UpdateJsonDatapointRequest {
    /// The ID of the datapoint to update. Required.
    pub id: Uuid,

    /// Datapoint input. If omitted, it will be left unchanged.
    #[serde(default)]
    pub input: Option<Input>,

    /// JSON datapoint output. If omitted, it will be left unchanged. If `null`, it will be set to `null`. If specified as a value, it will be set to the provided value.
    #[serde(default, deserialize_with = "deserialize_double_option")]
    #[schemars(extend("x-double-option" = true), description = "JSON datapoint output.
If omitted (which uses the default value `OMIT`), it will be left unchanged. If set to `None`, it will be cleared (represents edge case where
inference succeeded but model didn't output relevant content blocks). Otherwise, it will overwrite the existing output.")]
    pub output: Option<Option<JsonDatapointOutputUpdate>>,

    /// The output schema of the JSON datapoint. If omitted, it will be left unchanged. If specified as `null`, it will be set to `null`. If specified as a value, it will be set to the provided value.
    /// If not provided, the function's output schema will be used.
    #[serde(default)]
    pub output_schema: Option<Value>,

    /// Datapoint tags. If omitted, it will be left unchanged. If empty, it will be cleared. Otherwise,
    /// it will be overwrite the existing tags.
    #[serde(default)]
    pub tags: Option<HashMap<String, String>>,

    /// Metadata fields to update.
    #[serde(flatten)]
    pub metadata: DatapointMetadataUpdate,
}

/// A request to update the output of a JSON datapoint.
///
/// We intentionally only accept the `raw` field, because JSON datapoints can contain invalid or malformed JSON for eval purposes.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, Clone, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[export_schema]
pub struct JsonDatapointOutputUpdate {
    /// The raw output of the datapoint. For valid JSON outputs, this should be a JSON-serialized string.
    ///
    /// This will be parsed and validated against the datapoint's `output_schema`. Valid `raw` values will be parsed and stored as `parsed`, and
    /// invalid `raw` values will be stored as-is, because we allow invalid outputs in datapoints by design.
    pub raw: Option<String>,
}

/// A request to update the metadata of a datapoint.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Default, Deserialize, Clone, PartialEq, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
#[export_schema]
pub struct DatapointMetadataUpdate {
    /// Datapoint name. If omitted, it will be left unchanged. If specified as `null`, it will be set to `null`. If specified as a value, it will be set to the provided value.
    #[serde(default, deserialize_with = "deserialize_double_option")]
    #[schemars(extend("x-double-option" = true), description = "Datapoint name.

If omitted (which uses the default value `OMIT`), it will be left unchanged. If set to `None`, it will be cleared. If specified as a value, it will
be set to the provided value.")]
    pub name: Option<Option<String>>,
}

/// A response to a request to update one or more datapoints in a dataset.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, Clone, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[export_schema]
pub struct UpdateDatapointsResponse {
    /// The IDs of the datapoints that were updated.
    /// These are newly generated IDs for UpdateDatapoint requests, and they are the same IDs for UpdateDatapointMetadata requests.
    pub ids: Vec<Uuid>,
}

/// Request to update metadata for one or more datapoints in a dataset.
/// Used by the `PATCH /v1/datasets/{dataset_id}/datapoints/metadata` endpoint.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[export_schema]
pub struct UpdateDatapointsMetadataRequest {
    /// The datapoints to update metadata for.
    pub datapoints: Vec<UpdateDatapointMetadataRequest>,
}

/// A request to update the metadata of a single datapoint.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
#[export_schema]
pub struct UpdateDatapointMetadataRequest {
    /// The ID of the datapoint to update. Required.
    pub id: Uuid,

    /// Metadata fields to update.
    #[serde(flatten)]
    pub metadata: DatapointMetadataUpdate,
}

/// Request to list datapoints from a dataset with pagination and filters.
/// Used by the `POST /v1/datasets/{dataset_id}/list_datapoints` endpoint.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
#[export_schema]
pub struct ListDatapointsRequest {
    /// Optional function name to filter datapoints by.
    /// If provided, only datapoints from this function will be returned.
    pub function_name: Option<String>,

    /// The maximum number of datapoints to return.
    /// Defaults to 20.
    pub limit: Option<u32>,

    /// The maximum number of datapoints to return. Defaults to 20.
    /// Deprecated: please use `limit`. If `limit` is provided, `page_size` is ignored.
    #[deprecated(since = "2025.11.1", note = "Use `limit` instead")]
    pub page_size: Option<u32>,

    /// The number of datapoints to skip before starting to return results.
    /// Defaults to 0.
    pub offset: Option<u32>,

    /// Optional filter to apply when querying datapoints.
    /// Supports filtering by tags, time, and logical combinations (AND/OR/NOT).
    pub filter: Option<DatapointFilter>,

    /// Optional ordering criteria for the results.
    /// Supports multiple sort criteria (e.g., sort by timestamp then by search relevance).
    pub order_by: Option<Vec<DatapointOrderBy>>,

    /// Text query to filter. Case-insensitive substring search over the datapoints' input and output.
    ///
    /// THIS FEATURE IS EXPERIMENTAL, and we may change or remove it at any time.
    /// We recommend against depending on this feature for critical use cases.
    ///
    /// Important limitations:
    /// - This requires an exact substring match; we do not tokenize this query string.
    /// - This doesn't search for any content in the template itself.
    /// - Quality is based on term frequency > 0, without any relevance scoring.
    /// - There are no performance guarantees (it's best effort only). Today, with no other
    ///   filters, it will perform a full table scan, which may be extremely slow depending
    ///   on the data volume.
    pub search_query_experimental: Option<String>,
}

/// Request to get specific datapoints by their IDs.
/// Used by the `POST /v1/datasets/{dataset_name}/get_datapoints` endpoint.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[export_schema]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GetDatapointsRequest {
    /// The IDs of the datapoints to retrieve. Required.
    pub ids: Vec<Uuid>,
}

/// Response containing the requested datapoints.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[export_schema]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GetDatapointsResponse {
    /// The retrieved datapoints.
    pub datapoints: Vec<Datapoint>,
}

/// Request to create datapoints from inferences.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct CreateDatapointsFromInferenceRequest {
    #[serde(flatten)]
    pub params: CreateDatapointsFromInferenceRequestParams,
}

/// Parameters for creating datapoints from inferences.
/// Can specify either a list of inference IDs or a query to find inferences.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, JsonSchema, Serialize, TensorZeroDeserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[export_schema]
pub enum CreateDatapointsFromInferenceRequestParams {
    /// Create datapoints from specific inference IDs.
    #[schemars(title = "CreateDatapointsFromInferenceRequestParamsInferenceIds")]
    InferenceIds {
        /// The inference IDs to create datapoints from.
        inference_ids: Vec<Uuid>,

        /// When creating the datapoint, this specifies the source of the output for the datapoint.
        /// If not provided, by default we will use the original inference output as the datapoint's output
        /// (equivalent to `inference`).
        #[cfg_attr(feature = "ts-bindings", ts(optional))]
        output_source: Option<InferenceOutputSource>,
    },

    /// Create datapoints from an inference query.
    #[schemars(title = "CreateDatapointsFromInferenceRequestParamsInferenceQuery")]
    InferenceQuery {
        /// Flattened inference query parameters.
        #[serde(flatten)]
        query: Box<ListInferencesRequest>,
    },
}

/// Response from creating datapoints.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[export_schema]
pub struct CreateDatapointsResponse {
    /// The IDs of the newly-generated datapoints.
    pub ids: Vec<Uuid>,
}

/// Request to create datapoints manually.
/// Used by the `POST /v1/datasets/{dataset_id}/datapoints` endpoint.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[export_schema]
pub struct CreateDatapointsRequest {
    /// The datapoints to create.
    pub datapoints: Vec<CreateDatapointRequest>,
}

/// A tagged request to create a single datapoint.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, JsonSchema, Serialize, TensorZeroDeserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[export_schema]
#[cfg_attr(
    feature = "ts-bindings",
    ts(export, tag = "type", rename_all = "snake_case")
)]
pub enum CreateDatapointRequest {
    /// Request to create a chat datapoint.
    #[schemars(title = "CreateDatapointRequestChat")]
    Chat(CreateChatDatapointRequest),
    /// Request to create a JSON datapoint.
    #[schemars(title = "CreateDatapointRequestJson")]
    Json(CreateJsonDatapointRequest),
}

/// A request to create a chat datapoint.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[export_schema]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct CreateChatDatapointRequest {
    /// The function name for this datapoint. Required.
    pub function_name: String,

    /// Episode ID that the datapoint belongs to. Optional.
    #[serde(default)]
    pub episode_id: Option<Uuid>,

    /// Input to the function. Required.
    pub input: Input,

    /// Chat datapoint output. Optional.
    #[serde(default)]
    pub output: Option<Vec<ContentBlockChatOutput>>,

    /// Dynamic tool parameters for the datapoint. Optional.
    /// This is flattened to mirror inference requests.
    #[serde(flatten)]
    pub dynamic_tool_params: DynamicToolParams,

    /// Tags associated with this datapoint. Optional.
    #[serde(default)]
    pub tags: Option<HashMap<String, String>>,

    /// Human-readable name for the datapoint. Optional.
    #[serde(default)]
    pub name: Option<String>,
}

/// A request to create a JSON datapoint.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[export_schema]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct CreateJsonDatapointRequest {
    /// The function name for this datapoint. Required.
    pub function_name: String,

    /// Episode ID that the datapoint belongs to. Optional.
    #[serde(default)]
    pub episode_id: Option<Uuid>,

    /// Input to the function. Required.
    pub input: Input,

    /// JSON datapoint output. Optional.
    pub output: Option<JsonDatapointOutputUpdate>,

    /// The output schema of the JSON datapoint. Optional.
    /// If not provided, the function's output schema will be used. If provided, it will be validated.
    #[serde(default)]
    pub output_schema: Option<Value>,

    /// Tags associated with this datapoint. Optional.
    #[serde(default)]
    pub tags: Option<HashMap<String, String>>,

    /// Human-readable name for the datapoint. Optional.
    #[serde(default)]
    pub name: Option<String>,
}

/// Request to delete datapoints from a dataset.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[export_schema]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct DeleteDatapointsRequest {
    /// The IDs of the datapoints to delete.
    pub ids: Vec<Uuid>,
}

/// Response containing the number of deleted datapoints.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[export_schema]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct DeleteDatapointsResponse {
    /// The number of deleted datapoints.
    pub num_deleted_datapoints: u64,
}

/// Request to list datasets with optional filtering and pagination.
/// Used by the `GET /internal/datasets` endpoint.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
#[export_schema]
pub struct ListDatasetsRequest {
    /// Optional function name to filter datasets by.
    /// If provided, only datasets with datapoints for this function will be returned.
    pub function_name: Option<String>,

    /// The maximum number of datasets to return.
    pub limit: Option<u32>,

    /// The number of datasets to skip before starting to return results.
    pub offset: Option<u32>,
}

/// Metadata for a single dataset.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[export_schema]
pub struct DatasetMetadata {
    /// The name of the dataset.
    pub dataset_name: String,
    /// The total number of datapoints in the dataset.
    pub datapoint_count: u32,
    /// The timestamp of the last update (ISO 8601 format).
    pub last_updated: String,
}

/// Response containing a list of datasets.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[export_schema]
pub struct ListDatasetsResponse {
    /// List of dataset metadata.
    pub datasets: Vec<DatasetMetadata>,
}
