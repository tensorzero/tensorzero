//! Wire types for the TensorZero Autopilot API.
//!
//! These types are shared between the client and server.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
// Re-export types from tensorzero-types that InputMessage depends on
use schemars::JsonSchema;
pub use tensorzero_types::{
    Base64File, File, ObjectStoragePointer, RawText, Role, Template, Text, Thought,
    ToolCallWrapper, Unknown, UrlFile,
};
use tensorzero_types::{InputMessage, InputMessageContent};
use uuid::Uuid;

// =============================================================================
// Core Types
// =============================================================================

/// Content block types allowed in autopilot event messages.
/// Restricted to only Text blocks (no ToolCall, File, Template, etc.).
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(
    feature = "ts-bindings",
    ts(export, tag = "type", rename_all = "snake_case")
)]
pub enum EventPayloadMessageContent {
    Text(Text),
}

/// A message payload specific to autopilot events.
/// Content is restricted to Text blocks only.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct EventPayloadMessage {
    pub role: Role,
    pub content: Vec<EventPayloadMessageContent>,
}

impl TryFrom<InputMessage> for EventPayloadMessage {
    type Error = &'static str;

    fn try_from(msg: InputMessage) -> Result<Self, Self::Error> {
        let content = msg
            .content
            .into_iter()
            .map(|c| match c {
                InputMessageContent::Text(text) => Ok(EventPayloadMessageContent::Text(text)),
                _ => Err("EventPayloadMessage only supports Text content blocks"),
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(EventPayloadMessage {
            role: msg.role,
            content,
        })
    }
}

impl From<EventPayloadMessage> for InputMessage {
    fn from(msg: EventPayloadMessage) -> Self {
        InputMessage {
            role: msg.role,
            content: msg
                .content
                .into_iter()
                .map(|c| match c {
                    EventPayloadMessageContent::Text(text) => InputMessageContent::Text(text),
                })
                .collect(),
        }
    }
}

/// A session representing an autopilot conversation.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct Session {
    pub id: Uuid,
    pub organization_id: String,
    pub workspace_id: String,
    pub deployment_id: String,
    pub tensorzero_version: String,
    pub created_at: DateTime<Utc>,
}

/// Internal event type - consumers should use `GatewayEvent` instead.
///
/// Note: TS derive is needed for types that reference this, but we don't export it.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: Uuid,
    pub payload: EventPayload,
    pub session_id: Uuid,
    pub created_at: DateTime<Utc>,
}

/// An event as seen by gateway consumers.
///
/// Uses `GatewayEventPayload` which excludes `NotAvailable` authorization status.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GatewayEvent {
    pub id: Uuid,
    pub payload: GatewayEventPayload,
    pub session_id: Uuid,
    pub created_at: DateTime<Utc>,
}

impl TryFrom<Event> for GatewayEvent {
    type Error = &'static str;

    fn try_from(event: Event) -> Result<Self, Self::Error> {
        Ok(GatewayEvent {
            id: event.id,
            payload: event.payload.try_into()?,
            session_id: event.session_id,
            created_at: event.created_at,
        })
    }
}

/// The UX-relevant status of the Autopilot.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum AutopilotStatus {
    Idle,
    ServerSideProcessing,
    WaitingForToolCallAuthorization,
    WaitingForToolExecution,
    WaitingForRetry,
    Failed,
}

/// Internal stream update type - consumers should use `GatewayStreamUpdate` instead.
///
/// Note: TS derive is needed for types that reference this, but we don't export it.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamUpdate {
    pub event: Event,
    pub status: AutopilotStatus,
}

/// Stream update as seen by gateway consumers.
///
/// Uses `GatewayEvent` which excludes `NotAvailable` authorization status.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GatewayStreamUpdate {
    pub event: GatewayEvent,
    pub status: AutopilotStatus,
}

impl TryFrom<StreamUpdate> for GatewayStreamUpdate {
    type Error = &'static str;

    fn try_from(update: StreamUpdate) -> Result<Self, Self::Error> {
        Ok(GatewayStreamUpdate {
            event: update.event.try_into()?,
            status: update.status,
        })
    }
}

/// Error payload for an event.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventPayloadError {
    pub message: String,
}

/// Status update payload for an event.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventPayloadStatusUpdate {
    pub status_update: StatusUpdate,
}

/// Tool result payload for an event.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventPayloadToolResult {
    pub tool_call_event_id: Uuid,
    pub outcome: ToolOutcome,
}

/// Internal event payload type - consumers should use `GatewayEventPayload` instead.
///
/// Note: TS derive is needed for types that reference this, but we don't export it.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "ts-bindings", ts(tag = "type", rename_all = "snake_case"))]
pub enum EventPayload {
    Message(EventPayloadMessage),
    Error(EventPayloadError),
    StatusUpdate(EventPayloadStatusUpdate),
    ToolCall(EventPayloadToolCall),
    ToolCallAuthorization(EventPayloadToolCallAuthorization),
    ToolResult(EventPayloadToolResult),
    #[serde(other)]
    #[serde(alias = "other")] // legacy name
    Unknown,
}

impl EventPayload {
    /// Returns true if this payload type can be written by API clients.
    /// System-generated types (e.g. AutopilotEventPayloadStatusUpdate) return false.
    pub fn is_client_writable(&self) -> bool {
        matches!(self, EventPayload::Message(msg) if msg.role == Role::User)
            || matches!(
                self,
                EventPayload::ToolCallAuthorization(_) | EventPayload::ToolResult(_)
            )
    }
}

/// Event payload as seen by gateway consumers.
///
/// Uses `GatewayEventPayloadToolCallAuthorization` which excludes `NotAvailable` status.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(
    feature = "ts-bindings",
    ts(export, tag = "type", rename_all = "snake_case")
)]
pub enum GatewayEventPayload {
    Message(EventPayloadMessage),
    Error(EventPayloadError),
    StatusUpdate(EventPayloadStatusUpdate),
    ToolCall(EventPayloadToolCall),
    ToolCallAuthorization(GatewayEventPayloadToolCallAuthorization),
    ToolResult(EventPayloadToolResult),
    #[serde(other)]
    #[serde(alias = "other")] // legacy name
    Unknown,
}

impl TryFrom<EventPayload> for GatewayEventPayload {
    type Error = &'static str;

    fn try_from(payload: EventPayload) -> Result<Self, <Self as TryFrom<EventPayload>>::Error> {
        match payload {
            EventPayload::Message(m) => Ok(GatewayEventPayload::Message(m)),
            EventPayload::Error(e) => Ok(GatewayEventPayload::Error(e)),
            EventPayload::StatusUpdate(s) => Ok(GatewayEventPayload::StatusUpdate(s)),
            EventPayload::ToolCall(t) => Ok(GatewayEventPayload::ToolCall(t)),
            EventPayload::ToolCallAuthorization(auth) => {
                Ok(GatewayEventPayload::ToolCallAuthorization(auth.try_into()?))
            }
            EventPayload::ToolResult(r) => Ok(GatewayEventPayload::ToolResult(r)),
            EventPayload::Unknown => Ok(GatewayEventPayload::Unknown),
        }
    }
}

/// A status update within a session.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(
    feature = "ts-bindings",
    ts(export, tag = "type", rename_all = "snake_case")
)]
pub enum StatusUpdate {
    Text { text: String },
}

// =============================================================================
// Tool Call Types
// =============================================================================

/// Autopilot tool call with side info for tool execution.
///
/// This extends the interface of a standard tool call with bookkeeping information that
/// allows the caller to send over non-llm generated parameters.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventPayloadToolCall {
    /// Name
    pub name: String,
    /// Arguments
    pub arguments: serde_json::Value,
    /// Side info to pass to the tool (hidden from LLM, used for execution context).
    pub side_info: AutopilotSideInfo,
}

/// Side information required for autopilot client tools.
///
/// This should contain all IDs that might be needed as input to a tool
/// that do not need to be generated by LLMs (like the session id or a config hash).
/// We should implement this as a type that has optional or mandatory fields as needed
/// for each kind of tool, then implement TryFrom<AutopilotSideInfo> for each tool's side info type.
/// This can fail if the correct information is not present.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutopilotSideInfo {
    /// The event ID of the ToolCall event (for correlating ToolResult).
    pub tool_call_event_id: Uuid,

    /// The session ID for this autopilot session.
    pub session_id: Uuid,

    /// A hash of the current configuration.
    pub config_snapshot_hash: String,

    /// Settings for optimization workflows run on the gateway by autopilot.
    pub optimization: OptimizationWorkflowSideInfo,
}

/// Side info for optimization workflow tool (hidden from LLM).
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct OptimizationWorkflowSideInfo {
    /// Polling interval in seconds (default: 60).
    #[serde(default = "default_poll_interval_secs")]
    pub poll_interval_secs: u64,

    /// Maximum time to wait for completion in seconds (default: 86400 = 24 hours).
    #[serde(default = "default_max_wait_secs")]
    pub max_wait_secs: u64,
}

impl Default for OptimizationWorkflowSideInfo {
    fn default() -> Self {
        Self {
            poll_interval_secs: default_poll_interval_secs(),
            max_wait_secs: default_max_wait_secs(),
        }
    }
}
fn default_poll_interval_secs() -> u64 {
    60
}

fn default_max_wait_secs() -> u64 {
    86400
}

impl From<AutopilotSideInfo> for OptimizationWorkflowSideInfo {
    fn from(params: AutopilotSideInfo) -> Self {
        // This tool doesn't use the standard autopilot params - it has its own config.
        // Return defaults for polling configuration.
        params.optimization
    }
}

impl AutopilotSideInfo {
    /// Helper for tools that create new datapoints to get bookkeeping info.
    pub fn to_tags(&self) -> HashMap<String, String> {
        let mut tags = HashMap::new();
        tags.insert(
            "tensorzero::autopilot::tool_call_event_id".to_string(),
            self.tool_call_event_id.to_string(),
        );
        tags.insert(
            "tensorzero::autopilot::session_id".to_string(),
            self.session_id.to_string(),
        );
        tags.insert(
            "tensorzero::autopilot::config_snapshot_hash".to_string(),
            self.config_snapshot_hash.clone(),
        );
        tags.insert("tensorzero::autopilot".to_string(), "true".to_string());
        tags
    }
}

/// Implemented so that tools that don't need side info are able to satisfy trait bounds.
impl From<AutopilotSideInfo> for () {
    fn from(_: AutopilotSideInfo) -> Self {}
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutopilotToolResult {
    pub result: String,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolCallDecisionSource {
    Ui,
    Automatic,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventPayloadToolCallAuthorization {
    pub source: ToolCallDecisionSource,
    pub tool_call_event_id: Uuid,
    pub status: ToolCallAuthorizationStatus,
}

/// Tool call authorization payload as seen by gateway consumers.
///
/// Uses `GatewayToolCallAuthorizationStatus` which excludes `NotAvailable`.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayEventPayloadToolCallAuthorization {
    pub source: ToolCallDecisionSource,
    pub tool_call_event_id: Uuid,
    pub status: GatewayToolCallAuthorizationStatus,
}

impl TryFrom<EventPayloadToolCallAuthorization> for GatewayEventPayloadToolCallAuthorization {
    type Error = &'static str;

    fn try_from(auth: EventPayloadToolCallAuthorization) -> Result<Self, Self::Error> {
        Ok(GatewayEventPayloadToolCallAuthorization {
            source: auth.source,
            tool_call_event_id: auth.tool_call_event_id,
            status: auth.status.try_into()?,
        })
    }
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolCallAuthorizationStatus {
    Approved,
    Rejected { reason: String },
    NotAvailable,
}

/// Authorization status for tool calls as seen by gateway consumers.
///
/// This is a narrower type than `ToolCallAuthorizationStatus` that excludes
/// `NotAvailable` since that status is filtered out before reaching consumers.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GatewayToolCallAuthorizationStatus {
    Approved,
    Rejected { reason: String },
}

impl TryFrom<ToolCallAuthorizationStatus> for GatewayToolCallAuthorizationStatus {
    type Error = &'static str;

    fn try_from(status: ToolCallAuthorizationStatus) -> Result<Self, Self::Error> {
        match status {
            ToolCallAuthorizationStatus::Approved => {
                Ok(GatewayToolCallAuthorizationStatus::Approved)
            }
            ToolCallAuthorizationStatus::Rejected { reason } => {
                Ok(GatewayToolCallAuthorizationStatus::Rejected { reason })
            }
            ToolCallAuthorizationStatus::NotAvailable => {
                Err("NotAvailable status should be filtered before conversion")
            }
        }
    }
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolOutcome {
    Success(AutopilotToolResult),
    /// The user rejected the tool call request
    /// Note that this is currently never directly sent by the client - instead,
    /// `ToolCallAuthorizationStatus::Rejected` is sent to the server.
    /// The rejected tool will show in in the events list as `EventPayload::ToolResult`
    /// with `ToolOutcome::Rejected`
    Rejected {
        reason: String,
    },
    Failure {
        /// Structured error data from the tool.
        /// For autopilot tools, this is typically a serialized `AutopilotToolError`
        /// with a `kind` field discriminator (e.g., "ClientError", "Validation").
        error: serde_json::Value,
    },
    Missing,
    #[serde(other)]
    #[serde(alias = "other")] // legacy name
    Unknown,
}

// =============================================================================
// Request Types
// =============================================================================

/// Request body for creating an event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateEventRequest {
    pub deployment_id: String,
    pub tensorzero_version: String,
    pub payload: EventPayload,
    /// Used for idempotency when adding events to an existing session.
    ///
    /// When provided (for non-nil `session_id`), the server validates that this ID matches
    /// the most recent `user_message` event in the session. This prevents duplicate events
    /// from being created if a client retries a create user request that already succeeded.
    /// This should only apply to Message events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_user_message_event_id: Option<Uuid>,
    /// Must be set if the session id is nil and we are starting a new session
    pub config_snapshot_hash: Option<String>,
}

/// Query parameters for listing events.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct ListEventsParams {
    /// Maximum number of events to return. Defaults to 20.
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    /// Cursor for pagination: return events with id < before.
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before: Option<Uuid>,
}

/// Query parameters for listing sessions.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct ListSessionsParams {
    /// Maximum number of sessions to return. Defaults to 20.
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    /// Offset for pagination.
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<u32>,
}

/// Query parameters for streaming events.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct StreamEventsParams {
    /// Resume streaming from this event ID (exclusive).
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_event_id: Option<Uuid>,
}

/// Request body for approving all pending tool calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApproveAllToolCallsRequest {
    pub deployment_id: String,
    pub tensorzero_version: String,
    /// Only approve tool calls with event IDs <= this value.
    /// Prevents race condition where new tool calls arrive after client fetched the list.
    pub last_tool_call_event_id: Uuid,
}

// =============================================================================
// Response Types
// =============================================================================

/// Response from creating an event.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct CreateEventResponse {
    pub event_id: Uuid,
    pub session_id: Uuid,
}

/// Internal response type - consumers should use `GatewayListEventsResponse` instead.
///
/// Note: TS derive is needed for types that reference this, but we don't export it.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListEventsResponse {
    pub events: Vec<Event>,
    /// The most recent `message` event with role `user` in this session.
    pub previous_user_message_event_id: Uuid,
    /// The current status of the Autopilot in this session.
    /// Ignores pagination parameters.
    pub status: AutopilotStatus,
    /// All tool calls in Event history that do not have responses.
    /// These may be duplicates of some of the values in events.
    /// All EventPayloads in these Events should be of type ToolCall.
    #[serde(default)]
    pub pending_tool_calls: Vec<Event>,
}

/// Response from listing events as seen by gateway consumers.
///
/// Uses `GatewayEvent` which excludes `NotAvailable` authorization status.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GatewayListEventsResponse {
    pub events: Vec<GatewayEvent>,
    /// The most recent `message` event with role `user` in this session.
    pub previous_user_message_event_id: Uuid,
    /// The current status of the Autopilot in this session.
    /// Ignores pagination parameters.
    pub status: AutopilotStatus,
    /// All tool calls in Event history that do not have responses.
    /// These may be duplicates of some of the values in events.
    /// All EventPayloads in these Events should be of type ToolCall.
    #[serde(default)]
    pub pending_tool_calls: Vec<GatewayEvent>,
}

/// Response from listing sessions.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct ListSessionsResponse {
    pub sessions: Vec<Session>,
}

/// Response from approving all pending tool calls.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct ApproveAllToolCallsResponse {
    /// Number of tool calls that were approved.
    pub approved_count: u32,
    /// Event IDs of the newly created ToolCallAuthorization events.
    pub event_ids: Vec<Uuid>,
    /// Event IDs of the tool calls that were approved.
    pub tool_call_event_ids: Vec<Uuid>,
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
