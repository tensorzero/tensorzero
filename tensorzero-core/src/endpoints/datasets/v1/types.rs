use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

pub use crate::db::clickhouse::query_builder::{
    DatapointFilter, InferenceFilter, OrderBy, OrderByTerm, OrderDirection, TagFilter, TimeFilter,
};
use crate::endpoints::datasets::Datapoint;
use crate::inference::types::{ContentBlockChatOutput, Input};
use crate::serde_util::deserialize_double_option;
use crate::tool::DynamicToolParams;

/// Request to update one or more datapoints in a dataset.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct UpdateDatapointsRequest {
    /// The datapoints to update.
    pub datapoints: Vec<UpdateDatapointRequest>,
}

/// A tagged request to update a single datapoint in a dataset.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[serde(tag = "type", rename_all = "snake_case")]
#[ts(export, tag = "type", rename_all = "snake_case")]
pub enum UpdateDatapointRequest {
    /// Request to update a chat datapoint.
    Chat(UpdateChatDatapointRequest),
    /// Request to update a JSON datapoint.
    Json(UpdateJsonDatapointRequest),
}

/// An update request for a chat datapoint.
/// For any fields that are optional in ChatInferenceDatapoint, the request field distinguishes between an omitted field, `null`, and a value:
/// - If the field is omitted, it will be left unchanged.
/// - If the field is specified as `null`, it will be set to `null`.
/// - If the field has a value, it will be set to the provided value.
///
/// In Rust this is modeled as an `Option<Option<T>>`, where `None` means "unchanged" and `Some(None)` means "set to `null`" and `Some(Some(T))` means "set to the provided value".
#[derive(Debug, Serialize, Deserialize, Clone, ts_rs::TS)]
#[ts(export, optional_fields)]
pub struct UpdateChatDatapointRequest {
    /// The ID of the datapoint to update. Required.
    pub id: Uuid,

    /// Datapoint input. If omitted, it will be left unchanged.
    #[serde(default)]
    pub input: Option<Input>,

    /// Chat datapoint output. If omitted, it will be left unchanged. If empty, it will be cleared. Otherwise,
    /// it will overwrite the existing output.
    #[serde(default)]
    pub output: Option<Vec<ContentBlockChatOutput>>,

    /// Datapoint tool parameters. If omitted, it will be left unchanged. If specified as `null`, it will be set to `null`. If specified as a value, it will be set to the provided value.
    #[serde(default, deserialize_with = "deserialize_double_option")]
    pub tool_params: Option<Option<DynamicToolParams>>,

    /// Datapoint tags. If omitted, it will be left unchanged. If empty, it will be cleared. Otherwise,
    /// it will be overwrite the existing tags.
    #[serde(default)]
    pub tags: Option<HashMap<String, String>>,

    /// Metadata fields. If omitted, it will be left unchanged.
    #[serde(default)]
    pub metadata: Option<DatapointMetadataUpdate>,
}

/// An update request for a JSON datapoint.
/// For any fields that are optional in JsonInferenceDatapoint, the request field distinguishes between an omitted field, `null`, and a value:
/// - If the field is omitted, it will be left unchanged.
/// - If the field is specified as `null`, it will be set to `null`.
/// - If the field has a value, it will be set to the provided value.
///
/// In Rust this is modeled as an `Option<Option<T>>`, where `None` means "unchanged" and `Some(None)` means "set to `null`" and `Some(Some(T))` means "set to the provided value".
#[derive(Debug, Serialize, Deserialize, Clone, ts_rs::TS)]
#[ts(export, optional_fields)]
pub struct UpdateJsonDatapointRequest {
    /// The ID of the datapoint to update. Required.
    pub id: Uuid,

    /// Datapoint input. If omitted, it will be left unchanged.
    #[serde(default)]
    pub input: Option<Input>,

    /// JSON datapoint output. If omitted, it will be left unchanged. If `null`, it will be set to `null`. If specified as a value, it will be set to the provided value.
    /// This will be parsed and validated against output_schema, and valid `raw` values will be parsed and stored as `parsed`. Invalid `raw` values will
    /// also be stored, because we allow invalid outputs in datapoints by design.
    #[serde(default, deserialize_with = "deserialize_double_option")]
    pub output: Option<Option<JsonDatapointOutputUpdate>>,

    /// The output schema of the JSON datapoint. If omitted, it will be left unchanged. If specified as `null`, it will be set to `null`. If specified as a value, it will be set to the provided value.
    /// If not provided, the function's output schema will be used.
    #[serde(default)]
    pub output_schema: Option<Value>,

    /// Datapoint tags. If omitted, it will be left unchanged. If empty, it will be cleared. Otherwise,
    /// it will be overwrite the existing tags.
    #[serde(default)]
    pub tags: Option<HashMap<String, String>>,

    /// Metadata fields. If omitted, it will be left unchanged.
    #[serde(default)]
    pub metadata: Option<DatapointMetadataUpdate>,
}

/// A request to update the output of a JSON datapoint.
/// We intentionally only accept the `raw` field (in a JSON-serialized string), because datapoints can contain invalid outputs, and it's desirable
/// for users to run evals against them.
#[derive(Debug, Serialize, Deserialize, Clone, ts_rs::TS)]
#[ts(export)]
pub struct JsonDatapointOutputUpdate {
    /// The raw output of the datapoint. For valid JSON outputs, this should be a JSON-serialized string.
    pub raw: String,
}

/// A request to update the metadata of a datapoint.
#[derive(Debug, Serialize, Deserialize, Clone, ts_rs::TS)]
#[ts(export, optional_fields)]
pub struct DatapointMetadataUpdate {
    /// Datapoint name. If omitted, it will be left unchanged. If specified as `null`, it will be set to `null`. If specified as a value, it will be set to the provided value.
    #[serde(default, deserialize_with = "deserialize_double_option")]
    pub name: Option<Option<String>>,
}

/// A response to a request to update one or more datapoints in a dataset.
#[derive(Debug, Serialize, Deserialize, Clone, ts_rs::TS)]
#[ts(export)]
pub struct UpdateDatapointsResponse {
    /// The IDs of the datapoints that were updated.
    /// These are newly generated IDs for UpdateDatapoint requests, and they are the same IDs for UpdateDatapointMetadata requests.
    pub ids: Vec<Uuid>,
}

/// Request to update metadata for one or more datapoints in a dataset.
/// Used by the `PATCH /v1/datasets/{dataset_id}/datapoints/metadata` endpoint.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct UpdateDatapointsMetadataRequest {
    /// The datapoints to update metadata for.
    pub datapoints: Vec<UpdateDatapointMetadataRequest>,
}

/// A request to update the metadata of a single datapoint.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
pub struct UpdateDatapointMetadataRequest {
    /// The ID of the datapoint to update. Required.
    pub id: Uuid,

    /// Metadata fields to update.
    pub metadata: DatapointMetadataUpdate,
}

/// Request to list datapoints from a dataset with pagination and filters.
/// Used by the `POST /v1/datasets/{dataset_id}/list_datapoints` endpoint.
#[derive(Debug, Default, Serialize, Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
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
}

/// Request to get specific datapoints by their IDs.
/// Used by the `POST /v1/datasets/get_datapoints` endpoint.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct GetDatapointsRequest {
    /// The IDs of the datapoints to retrieve. Required.
    pub ids: Vec<Uuid>,
}

/// Response containing the requested datapoints.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct GetDatapointsResponse {
    /// The retrieved datapoints.
    pub datapoints: Vec<Datapoint>,
}

/// Specifies the source of the output for the datapoint when creating datapoints from inferences.
/// - `None`: Do not include any output in the datapoint.
/// - `Inference`: Include the original inference output in the datapoint.
/// - `Demonstration`: Include the latest demonstration feedback as output in the datapoint.
#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(rename_all = "snake_case")]
pub enum CreateDatapointsFromInferenceOutputSource {
    /// Do not include any output in the datapoint.
    None,
    /// Include the original inference output in the datapoint.
    Inference,
    /// Include the latest demonstration feedback as output in the datapoint.
    Demonstration,
}

/// Request to create datapoints from inferences.
#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export, optional_fields)]
pub struct CreateDatapointsFromInferenceRequest {
    #[serde(flatten)]
    pub params: CreateDatapointsFromInferenceRequestParams,

    /// When creating the datapoint, this specifies the source of the output for the datapoint.
    /// If not provided, by default we will use the original inference output as the datapoint's output
    /// (equivalent to `inference`).
    pub output_source: Option<CreateDatapointsFromInferenceOutputSource>,
}

/// Parameters for creating datapoints from inferences.
/// Can specify either a list of inference IDs or a query to find inferences.
#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CreateDatapointsFromInferenceRequestParams {
    /// Create datapoints from specific inference IDs.
    InferenceIds {
        /// The inference IDs to create datapoints from.
        inference_ids: Vec<Uuid>,
    },

    /// Create datapoints from an inference query.
    InferenceQuery {
        /// The function name to filter inferences by.
        function_name: String,

        /// Variant name to filter inferences by, optional.
        #[ts(optional)]
        variant_name: Option<String>,

        /// Filters to apply when querying inferences, optional.
        #[ts(optional)]
        filters: Option<InferenceFilter>,
    },
}

/// Response from creating datapoints.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct CreateDatapointsResponse {
    /// The IDs of the newly-generated datapoints.
    pub ids: Vec<Uuid>,
}

/// Request to create datapoints manually.
/// Used by the `POST /v1/datasets/{dataset_id}/datapoints` endpoint.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct CreateDatapointsRequest {
    /// The datapoints to create.
    pub datapoints: Vec<CreateDatapointRequest>,
}

/// A tagged request to create a single datapoint.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[derive(ts_rs::TS)]
#[ts(export, tag = "type", rename_all = "snake_case")]
pub enum CreateDatapointRequest {
    /// Request to create a chat datapoint.
    Chat(CreateChatDatapointRequest),
    /// Request to create a JSON datapoint.
    Json(CreateJsonDatapointRequest),
}

/// A request to create a chat datapoint.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
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
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
pub struct CreateJsonDatapointRequest {
    /// The function name for this datapoint. Required.
    pub function_name: String,

    /// Episode ID that the datapoint belongs to. Optional.
    #[serde(default)]
    pub episode_id: Option<Uuid>,

    /// Input to the function. Required.
    pub input: Input,

    /// JSON datapoint output. Optional.
    /// If provided, it will be validated against the output_schema. Invalid raw outputs will be stored as-is (not parsed), because we allow
    /// invalid outputs in datapoints by design.
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
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct DeleteDatapointsRequest {
    /// The IDs of the datapoints to delete.
    pub ids: Vec<Uuid>,
}

/// Response containing the number of deleted datapoints.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct DeleteDatapointsResponse {
    /// The number of deleted datapoints.
    pub num_deleted_datapoints: u64,
}
