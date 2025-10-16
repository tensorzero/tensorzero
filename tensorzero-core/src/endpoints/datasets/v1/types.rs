use std::collections::HashMap;

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::inference::types::Input;
use crate::tool::ToolCallConfigDatabaseInsert;

#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
/// Request to update one or more datapoints in a dataset.
pub struct UpdateDatapointsRequest {
    /// The datapoints to update.
    pub datapoints: Vec<UpdateDatapointRequest>,
}

#[derive(Debug, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct UpdateDatapointsResponse {
    pub ids: Vec<Uuid>,
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

impl UpdateDatapointRequest {
    pub fn id(&self) -> Uuid {
        match self {
            UpdateDatapointRequest::Chat(chat) => chat.id,
            UpdateDatapointRequest::Json(json) => json.id,
        }
    }
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
    /// The input of the chat datapoint.
    #[serde(default)]
    pub input: Option<Input>,
    #[serde(default)]
    // #[ts(type = "unknown | null")]
    pub output: Option<Option<Value>>,
    #[serde(default, deserialize_with = "deserialize_option_option_tool_params")]
    // #[ts(type = "ToolCallConfigDatabaseInsert | null")]
    pub tool_params: Option<Option<ToolCallConfigDatabaseInsert>>,
    #[serde(default)]
    // #[ts(type = "Record<string, string> | null")]
    pub tags: Option<Option<HashMap<String, String>>>,
    #[serde(default)]
    // #[ts(optional)]
    pub is_deleted: Option<bool>,
    #[serde(default)]
    // #[ts(type = "string | null")]
    pub name: Option<Option<String>>,
}

#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct UpdateJsonDatapointRequest {
    pub id: Uuid,
    #[serde(default)]
    // #[cfg_attrts(optional, type = "ClientInput")]
    pub input: Option<Input>,
    #[serde(default)]
    // #[ts(optional, type = "unknown | null")]
    pub output: Option<Option<Value>>,
    #[serde(default)]
    // #[ts(optional)]
    pub output_schema: Option<Value>,
    #[serde(default)]
    // #[ts(optional, type = "Record<string, string> | null")]
    pub tags: Option<Option<HashMap<String, String>>>,
    #[serde(default)]
    // #[ts(optional)]
    pub is_deleted: Option<bool>,
    #[serde(default)]
    // #[ts(optional, type = "string | null")]
    pub name: Option<Option<String>>,
}

fn deserialize_option_option_tool_params<'de, D>(
    deserializer: D,
) -> Result<Option<Option<ToolCallConfigDatabaseInsert>>, D::Error>
where
    D: Deserializer<'de>,
{
    let value: Option<Value> = Option::deserialize(deserializer)?;
    match value {
        None => Ok(None),
        Some(Value::Null) => Ok(Some(None)),
        Some(Value::String(s)) => {
            if s.is_empty() {
                return Ok(Some(None));
            }
            let parsed = serde_json::from_str(&s).map_err(serde::de::Error::custom)?;
            Ok(Some(Some(parsed)))
        }
        Some(other) => {
            let parsed = serde_json::from_value(other).map_err(serde::de::Error::custom)?;
            Ok(Some(Some(parsed)))
        }
    }
}
