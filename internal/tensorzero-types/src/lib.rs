//! Wire format types for the TensorZero API.
//!
//! This crate contains the shared types used in API requests and responses.
//! These types are used by both tensorzero-core and autopilot-client.

pub mod content;
pub mod cost;
pub mod error;
pub mod file;
pub mod message;
pub mod role;
pub mod storage;
pub mod tool;

pub(crate) fn deprecation_warning(message: &str) {
    tracing::warn!("Deprecation Warning: {message}");
}

// Re-export all public types at the crate root for convenience
pub use content::{
    Arguments, RawText, System, Template, Text, Thought, ThoughtSummaryBlock, Unknown,
};
pub use cost::{
    CostPointerConfig, UnifiedCostPointerConfig, UninitializedCostConfig,
    UninitializedCostConfigEntry, UninitializedCostRate, UninitializedUnifiedCostConfig,
};
pub use error::TypeError;
pub use file::{
    Base64File, Base64FileMetadata, Detail, File, ObjectStorageError, ObjectStorageFile,
    ObjectStoragePointer, UrlFile,
};
pub use message::{Input, InputMessage, InputMessageContent, TextKind};
pub use role::{
    ASSISTANT_TEXT_TEMPLATE_VAR, Role, SYSTEM_TEXT_TEMPLATE_VAR, USER_TEXT_TEMPLATE_VAR,
};
use serde::{Deserialize, Serialize};
pub use storage::{StorageKind, StoragePath};
use tensorzero_derive::TensorZeroDeserialize;
pub use tool::{InferenceResponseToolCall, ToolCall, ToolCallWrapper, ToolChoice, ToolResult};
use uuid::Uuid;

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize, sqlx::Type)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(rename_all = "snake_case")]
#[sqlx(type_name = "text", rename_all = "snake_case")]
pub enum FunctionType {
    #[default]
    Chat,
    Json,
}

impl FunctionType {
    pub fn inference_table_name(&self) -> &'static str {
        match self {
            FunctionType::Chat => "ChatInference",
            FunctionType::Json => "JsonInference",
        }
    }
}

/// A single resolved object type for a given UUID.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, TensorZeroDeserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub enum ResolvedObject {
    Inference {
        function_name: String,
        function_type: FunctionType,
        variant_name: String,
        episode_id: Uuid,
    },
    Episode,
    BooleanFeedback,
    FloatFeedback,
    CommentFeedback,
    DemonstrationFeedback,
    ChatDatapoint {
        dataset_name: String,
        function_name: String,
    },
    JsonDatapoint {
        dataset_name: String,
        function_name: String,
    },
    ModelInference {
        inference_id: Uuid,
        model_name: String,
        model_provider_name: String,
    },
}

/// Response type for the resolve_uuid endpoint.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct ResolveUuidResponse {
    pub id: Uuid,
    pub object_types: Vec<ResolvedObject>,
}
