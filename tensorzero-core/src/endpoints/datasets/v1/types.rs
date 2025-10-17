use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::inference::types::{ContentBlockChatOutput, Input};
use crate::serde_util::deserialize_double_option;
use crate::tool::ToolCallConfigDatabaseInsert;

#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
/// Request to update one or more datapoints in a dataset.
pub struct UpdateDatapointsRequest {
    /// The datapoints to update.
    pub datapoints: Vec<UpdateDatapointRequest>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, tag = "type", rename_all = "snake_case"))]
/// A tagged request to update a single datapoint in a dataset.
pub enum UpdateDatapointRequest {
    /// Request to update a chat datapoint.
    Chat(UpdateChatDatapointRequest),
    /// Request to update a JSON datapoint.
    Json(UpdateJsonDatapointRequest),
}

#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
/// An update request for a chat datapoint.
/// For any fields that are optional in ChatInferenceDatapoint, the request field distinguishes between an omitted field, `null`, and a value:
/// - If the field is omitted, it will be left unchanged.
/// - If the field is specified as `null`, it will be set to `null`.
/// - If the field has a value, it will be set to the provided value.
///
/// In Rust this is modeled as an `Option<Option<T>>`, where `None` means "unchanged" and `Some(None)` means "set to `null`" and `Some(Some(T))` means "set to the provided value".
pub struct UpdateChatDatapointRequest {
    /// The ID of the datapoint to update. Required.
    pub id: Uuid,

    /// Datapoint input. If omitted or `null`, it will be left unchanged.
    #[serde(default)]
    pub input: Option<Input>,

    /// Chat datapoint output. If omitted or `null`, it will be left unchanged. If empty, it will be cleared. Otherwise,
    /// it will overwrite the existing output.
    #[serde(default)]
    pub output: Option<Vec<ContentBlockChatOutput>>,

    /// Datapoint tool parameters. If omitted, it will be left unchanged. If specified as `null`, it will be set to `null`. If specified as a value, it will be set to the provided value.
    #[serde(default, deserialize_with = "deserialize_double_option")]
    pub tool_params: Option<Option<ToolCallConfigDatabaseInsert>>,

    /// Datapoint tags. If omitted or `null`, it will be left unchanged. If empty, it will be cleared. Otherwise,
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
#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct UpdateJsonDatapointRequest {
    /// The ID of the datapoint to update. Required.
    pub id: Uuid,

    /// Datapoint input. If omitted or `null`, it will be left unchanged.
    #[serde(default)]
    pub input: Option<Input>,

    /// JSON datapoint output. If omitted, it will be left unchanged. If `null`, it will be set to `null`. If specified as a value, it will be set to the provided value.
    /// This will be validated against `output_schema` or the function's output schema.
    #[serde(default, deserialize_with = "deserialize_double_option")]
    pub output: Option<Option<Value>>,

    /// The output schema of the JSON datapoint. If omitted, it will be left unchanged. If specified as `null`, it will be set to `null`. If specified as a value, it will be set to the provided value.
    /// If not provided, the function's output schema will be used.
    #[serde(default)]
    pub output_schema: Option<Value>,

    /// Datapoint tags. If omitted or `null`, it will be left unchanged. If empty, it will be cleared. Otherwise,
    /// it will be overwrite the existing tags.
    #[serde(default)]
    pub tags: Option<HashMap<String, String>>,

    /// Metadata fields. If omitted, it will be left unchanged.
    #[serde(default)]
    pub metadata: Option<DatapointMetadataUpdate>,
}

/// A request to update the metadata of a datapoint.
#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct DatapointMetadataUpdate {
    /// Datapoint name. If omitted, it will be left unchanged. If specified as `null`, it will be set to `null`. If specified as a value, it will be set to the provided value.
    #[serde(default, deserialize_with = "deserialize_double_option")]
    pub name: Option<Option<String>>,
}

/// A response to a request to update one or more datapoints in a dataset.
#[derive(Debug, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct UpdateDatapointsResponse {
    /// The IDs of the datapoints that were updated.
    /// These are newly generated IDs for UpdateDatapoint requests, and they are the same IDs for UpdateDatapointMetadata requests.
    pub ids: Vec<Uuid>,
}
