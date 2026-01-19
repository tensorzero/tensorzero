use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use tensorzero_derive::TensorZeroDeserialize;
use uuid::Uuid;

use crate::config::snapshot::SnapshotHash;
use crate::endpoints::datasets::Datapoint;
use crate::error::Error;
use crate::inference::types::stored_input::StoredInput;
use crate::inference::types::{ContentBlockChatOutput, JsonInferenceOutput, Text};
use crate::serde_util::{
    deserialize_optional_string_or_parsed_json, deserialize_string_or_parsed_json,
    serialize_none_as_empty_map,
};
use crate::stored_inference::{SimpleStoredSampleInfo, StoredOutput, StoredSample};
use crate::tool::{ToolCallConfigDatabaseInsert, deserialize_optional_tool_info};

/// Tagged enum for stored datapoints, used when querying from ClickHouse.
#[derive(Clone, Debug, Serialize, TensorZeroDeserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum StoredDatapoint {
    Chat(StoredChatInferenceDatapoint),
    Json(StoredJsonInferenceDatapoint),
}

/// Storage variant of ChatInferenceDatapoint for database operations (no Python/TypeScript bindings).
/// This type is used for both reading from and writing to ClickHouse.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StoredChatInferenceDatapoint {
    /// Name of the dataset to write to.
    pub dataset_name: String,

    /// Name of the function that generated this datapoint.
    pub function_name: String,

    /// Unique identifier for the datapoint.
    pub id: Uuid,

    /// Episode ID that the datapoint belongs to.
    pub episode_id: Option<Uuid>,

    /// Input type that we directly store in ClickHouse.
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub input: StoredInput,

    /// Output of the function that generated this datapoint. Optional.
    /// TODO(#4405): this should be a new type StoredContentBlockChatOutput that takes the storage ToolCallOutput format.
    #[serde(
        default,
        deserialize_with = "deserialize_optional_string_or_parsed_json"
    )]
    pub output: Option<Vec<ContentBlockChatOutput>>,

    /// Tool parameters used to generate this datapoint. Optional.
    #[serde(flatten, deserialize_with = "deserialize_optional_tool_info")]
    pub tool_params: Option<ToolCallConfigDatabaseInsert>,

    /// Tags associated with this datapoint. Optional.
    #[serde(default, serialize_with = "serialize_none_as_empty_map")]
    pub tags: Option<HashMap<String, String>>,

    /// If true, this datapoint was manually created or edited by the user.
    #[serde(default)]
    pub is_custom: bool,

    /// Source inference ID that generated this datapoint.
    #[serde(default)]
    pub source_inference_id: Option<Uuid>,

    /// Timestamp when the datapoint was marked as stale.
    #[serde(default)]
    pub staled_at: Option<String>,

    /// Human-readable name of the datapoint.
    #[serde(default)]
    pub name: Option<String>,

    /// Hash of the configuration snapshot that created this datapoint. Optional.
    /// This should always be Some when writing (after the feature flag is removed)
    /// but since we also read this type, it will remain an Option.
    #[serde(default)]
    pub snapshot_hash: Option<SnapshotHash>,

    // ================================
    // The following fields are ignored during insert.
    // ================================
    /// If true, this datapoint was deleted.
    /// Do not use - we only use soft deletions via `staled_at`. It will be ignored during insert.
    #[serde(default, skip_serializing)]
    pub is_deleted: bool,

    /// Deprecated, do not use.
    #[serde(default, skip_serializing)]
    pub auxiliary: String,

    /// Timestamp when the datapoint was updated.
    /// Ignored during insert.
    #[serde(skip_serializing)]
    pub updated_at: String,
}

impl std::fmt::Display for StoredChatInferenceDatapoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

/// Storage variant of JsonInferenceDatapoint for database operations (no Python/TypeScript bindings).
/// This type is used for both reading from and writing to ClickHouse.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StoredJsonInferenceDatapoint {
    /// Name of the dataset to write to.
    pub dataset_name: String,

    /// Name of the function that generated this datapoint.
    pub function_name: String,

    /// Unique identifier for the datapoint.
    pub id: Uuid,

    /// Episode ID that the datapoint belongs to.
    pub episode_id: Option<Uuid>,

    /// Input type that we directly store in ClickHouse.
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub input: StoredInput,

    #[serde(
        default,
        deserialize_with = "deserialize_optional_string_or_parsed_json"
    )]
    pub output: Option<JsonInferenceOutput>,

    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub output_schema: serde_json::Value,

    #[serde(default, serialize_with = "serialize_none_as_empty_map")]
    pub tags: Option<HashMap<String, String>>,

    /// If true, this datapoint was manually created or edited by the user.
    #[serde(default)]
    pub is_custom: bool,

    /// Source inference ID that generated this datapoint.
    #[serde(default)]
    pub source_inference_id: Option<Uuid>,

    /// Timestamp when the datapoint was marked as stale.
    #[serde(default)]
    pub staled_at: Option<String>,

    /// Human-readable name of the datapoint.
    #[serde(default)]
    pub name: Option<String>,

    /// Hash of the configuration snapshot that created this datapoint. Optional.
    /// This should always be Some when writing (after the feature flag is removed)
    /// but since we also read this type, it will remain an Option.
    #[serde(default)]
    pub snapshot_hash: Option<SnapshotHash>,

    // ================================
    // The following fields are ignored during insert.
    // ================================
    /// If true, this datapoint was deleted.
    /// Do not use - we only use soft deletions via `staled_at`. It will be ignored during insert.
    #[serde(default, skip_serializing)]
    pub is_deleted: bool,

    /// Deprecated, do not use.
    #[serde(default, skip_serializing)]
    pub auxiliary: String,

    /// Timestamp when the datapoint was updated.
    /// Ignored during insert.
    #[serde(skip_serializing)]
    pub updated_at: String,
}

impl std::fmt::Display for StoredJsonInferenceDatapoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl StoredDatapoint {
    pub fn id(&self) -> Uuid {
        match self {
            StoredDatapoint::Chat(datapoint) => datapoint.id,
            StoredDatapoint::Json(datapoint) => datapoint.id,
        }
    }

    pub fn dataset_name(&self) -> &str {
        match self {
            StoredDatapoint::Chat(datapoint) => &datapoint.dataset_name,
            StoredDatapoint::Json(datapoint) => &datapoint.dataset_name,
        }
    }

    pub fn input(&self) -> &StoredInput {
        match self {
            StoredDatapoint::Chat(datapoint) => &datapoint.input,
            StoredDatapoint::Json(datapoint) => &datapoint.input,
        }
    }

    pub fn tool_call_config(&self) -> Option<&ToolCallConfigDatabaseInsert> {
        match self {
            StoredDatapoint::Chat(datapoint) => datapoint.tool_params.as_ref(),
            StoredDatapoint::Json(_) => None,
        }
    }

    pub fn output_schema(&self) -> Option<&serde_json::Value> {
        match self {
            StoredDatapoint::Chat(_) => None,
            StoredDatapoint::Json(datapoint) => Some(&datapoint.output_schema),
        }
    }

    pub fn name(&self) -> Option<&str> {
        match self {
            StoredDatapoint::Chat(datapoint) => datapoint.name.as_deref(),
            StoredDatapoint::Json(datapoint) => datapoint.name.as_deref(),
        }
    }

    /// Convert to wire type, properly handling tool params by subtracting static tools
    /// TODO(shuyangli): Add parameter to optionally fetch files from object storage
    pub fn into_datapoint(self) -> Result<Datapoint, Error> {
        match self {
            StoredDatapoint::Chat(chat) => Ok(Datapoint::Chat(chat.into_datapoint())),
            StoredDatapoint::Json(json) => Ok(Datapoint::Json(json.into_datapoint())),
        }
    }
}

impl std::fmt::Display for StoredDatapoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoredDatapoint::Chat(datapoint) => write!(f, "{datapoint}"),
            StoredDatapoint::Json(datapoint) => write!(f, "{datapoint}"),
        }
    }
}

impl StoredSample for StoredDatapoint {
    fn function_name(&self) -> &str {
        match self {
            StoredDatapoint::Chat(datapoint) => &datapoint.function_name,
            StoredDatapoint::Json(datapoint) => &datapoint.function_name,
        }
    }

    fn input(&self) -> &StoredInput {
        match self {
            StoredDatapoint::Chat(datapoint) => &datapoint.input,
            StoredDatapoint::Json(datapoint) => &datapoint.input,
        }
    }

    fn input_mut(&mut self) -> &mut StoredInput {
        match self {
            StoredDatapoint::Chat(datapoint) => &mut datapoint.input,
            StoredDatapoint::Json(datapoint) => &mut datapoint.input,
        }
    }

    fn into_input(self) -> StoredInput {
        match self {
            StoredDatapoint::Chat(datapoint) => datapoint.input,
            StoredDatapoint::Json(datapoint) => datapoint.input,
        }
    }

    fn owned_simple_info(self) -> SimpleStoredSampleInfo {
        match self {
            StoredDatapoint::Chat(datapoint) => SimpleStoredSampleInfo {
                function_name: datapoint.function_name,
                input: datapoint.input,
                output: datapoint.output.clone(),
                stored_output: datapoint.output.map(StoredOutput::Chat),
                dispreferred_outputs: Vec::default(),
                tool_params: datapoint.tool_params,
                output_schema: None,
                episode_id: None,
                inference_id: None,
                tags: datapoint.tags.unwrap_or_default(),
            },
            StoredDatapoint::Json(datapoint) => {
                let stored_output = datapoint.output.clone().map(StoredOutput::Json);
                let output = datapoint.output.map(|output| match output.raw {
                    Some(raw) => vec![ContentBlockChatOutput::Text(Text { text: raw })],
                    None => vec![],
                });
                SimpleStoredSampleInfo {
                    function_name: datapoint.function_name,
                    input: datapoint.input,
                    output,
                    stored_output,
                    dispreferred_outputs: Vec::default(),
                    tool_params: None,
                    output_schema: Some(datapoint.output_schema),
                    episode_id: None,
                    inference_id: None,
                    tags: datapoint.tags.unwrap_or_default(),
                }
            }
        }
    }
}
