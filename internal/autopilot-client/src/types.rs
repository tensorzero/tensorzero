//! Wire types for the TensorZero Autopilot API.
//!
//! These types are shared between the client and server.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
// Re-export types from tensorzero-types that InputMessage depends on
pub use tensorzero_types::{
    Base64File, File, InputMessage, InputMessageContent, ObjectStoragePointer, RawText, Role,
    Template, Text, Thought, ToolCallWrapper, ToolResult, Unknown, UrlFile,
};
use uuid::Uuid;

// =============================================================================
// Core Types
// =============================================================================

/// A session representing an autopilot conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: Uuid,
    pub organization_id: String,
    pub workspace_id: String,
    pub deployment_id: Uuid,
    pub tensorzero_version: String,
    pub created_at: DateTime<Utc>,
}

/// An event within a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: Uuid,
    pub payload: EventPayload,
    pub session_id: Uuid,
    pub created_at: DateTime<Utc>,
}

/// The payload of an event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EventPayload {
    Message(InputMessage),
    StatusUpdate { status_update: StatusUpdate },
}

/// Payload for an assistant message event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantMessagePayload {
    pub content: Vec<serde_json::Value>,
}

/// A status update within a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StatusUpdate {
    Text { text: String },
}

// =============================================================================
// Request Types
// =============================================================================

/// Request body for creating an event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateEventRequest {
    pub deployment_id: Uuid,
    pub tensorzero_version: String,
    pub payload: EventPayload,
    /// Used for idempotency when adding events to an existing session.
    ///
    /// When provided (for non-nil `session_id`), the server validates that this ID matches
    /// the most recent `user_message` event in the session. This prevents duplicate events
    /// from being created if a client retries a request that already succeeded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_user_message_event_id: Option<Uuid>,
}

/// Query parameters for listing events.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ListEventsParams {
    /// Maximum number of events to return. Defaults to 20.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    /// Cursor for pagination: return events with id < before.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before: Option<Uuid>,
}

/// Query parameters for listing sessions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ListSessionsParams {
    /// Maximum number of sessions to return. Defaults to 20.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    /// Offset for pagination.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<u32>,
}

/// Query parameters for streaming events.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamEventsParams {
    /// Resume streaming from this event ID (exclusive).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_event_id: Option<Uuid>,
}

// =============================================================================
// Response Types
// =============================================================================

/// Response from creating an event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateEventResponse {
    pub event_id: Uuid,
    pub session_id: Uuid,
}

/// Response from listing events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListEventsResponse {
    pub events: Vec<Event>,
    /// The most recent `user_message` event in this session.
    pub previous_user_message_event_id: Uuid,
}

/// Response from listing sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListSessionsResponse {
    pub sessions: Vec<Session>,
}

// =============================================================================
// Error Types
// =============================================================================

/// Error response from the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

/// Details of an error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetail {
    pub message: String,
}
