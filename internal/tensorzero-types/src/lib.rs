//! Wire format types for the TensorZero API.
//!
//! This crate contains the shared types used in API requests and responses.
//! These types are used by both tensorzero-core and autopilot-client.

pub mod content;
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
pub use error::TypeError;
pub use file::{
    Base64File, Base64FileMetadata, Detail, File, ObjectStorageError, ObjectStorageFile,
    ObjectStoragePointer, UrlFile,
};
pub use message::{Input, InputMessage, InputMessageContent, TextKind};
pub use role::{
    ASSISTANT_TEXT_TEMPLATE_VAR, Role, SYSTEM_TEXT_TEMPLATE_VAR, USER_TEXT_TEMPLATE_VAR,
};
pub use storage::{StorageKind, StoragePath};
pub use tool::{InferenceResponseToolCall, ToolCall, ToolCallWrapper, ToolChoice, ToolResult};
