//! Wire types for the TensorZero Autopilot API.
//!
//! These types are shared between the client and server.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
// Re-export types from tensorzero-types that InputMessage depends on
use schemars::JsonSchema;
pub use tensorzero_types::{
    Base64File, File, ObjectStoragePointer, RawText, Role, Template, Text, Thought,
    ToolCallWrapper, Unknown, UrlFile,
};
use tensorzero_types::{InputMessage, InputMessageContent, ResolveUuidResponse};
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
    #[serde(default)]
    // EventPayloadMessageMetadata currently has no fields exposed to Typescript,
    // so we need to override the type to satisfy eslint
    #[cfg_attr(feature = "ts-bindings", ts(type = "Record<string, never>"))]
    pub metadata: EventPayloadMessageMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct EventPayloadMessageMetadata {
    /// Attempted lookups for anything that matched a UUID regex
    /// in the parent `EventPayloadMessage`
    // We hide this from the UI, and populate in in the gateway
    // before proxying it to the autopilot server
    #[serde(default)]
    pub resolved_uuids: Vec<ResolveUuidResponse>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub last_event_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub short_summary: Option<String>,
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
    WaitingForUserQuestionsAnswers,
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
    /// Populated by the server from the originating tool call event.
    /// Optional for backwards compatibility until the API is deployed with enrichment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub tool_call_name: Option<String>,
    /// Populated by the server from the originating tool call event.
    /// Optional for backwards compatibility until the API is deployed with enrichment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub tool_call_arguments: Option<serde_json::Value>,
    /// Authorization source (Ui/Automatic/Whitelist). Optional because interrupted
    /// tool results may not have a corresponding authorization event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub tool_call_authorization_source: Option<ToolCallDecisionSource>,
    /// Authorization status (Approved/Rejected/NotAvailable). Optional for same reason.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub tool_call_authorization_status: Option<ToolCallAuthorizationStatus>,
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
    Visualization(EventPayloadVisualization),
    UserQuestions(EventPayloadUserQuestions),
    UserQuestionsAnswers(EventPayloadUserQuestionsAnswers),
    AutoEvalExampleLabeling(EventPayloadAutoEvalExampleLabeling),
    AutoEvalExampleLabelingAnswers(EventPayloadAutoEvalExampleLabelingAnswers),
    #[serde(other)]
    #[serde(alias = "other")] // legacy name
    Unknown,
}

/// Minimal tool result payload for creating events.
/// Omits server-enriched fields like `tool_call_name`, `tool_call_arguments`, etc.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateEventPayloadToolResult {
    pub tool_call_event_id: Uuid,
    pub outcome: ToolOutcome,
}

/// Payload enum restricted to client-writable event types.
///
/// Unlike `EventPayload`, this only includes variants that API clients are allowed
/// to create. Server-enriched fields (e.g. `tool_call_name`, `tool_call_arguments`)
/// are omitted from the inner structs.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "ts-bindings", ts(tag = "type", rename_all = "snake_case"))]
pub enum CreateEventPayload {
    Message(EventPayloadMessage),
    ToolCallAuthorization(CreateEventPayloadToolCallAuthorization),
    ToolResult(CreateEventPayloadToolResult),
    UserQuestionsAnswers(EventPayloadUserQuestionsAnswers),
    AutoEvalExampleLabelingAnswers(CreateEventPayloadAutoEvalExampleLabelingAnswers),
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
    ToolCall(GatewayEventPayloadToolCall),
    ToolCallAuthorization(GatewayEventPayloadToolCallAuthorization),
    ToolResult(EventPayloadToolResult),
    Visualization(EventPayloadVisualization),
    UserQuestions(EventPayloadUserQuestions),
    UserQuestionsAnswers(EventPayloadUserQuestionsAnswers),
    AutoEvalExampleLabeling(EventPayloadAutoEvalExampleLabeling),
    AutoEvalExampleLabelingAnswers(EventPayloadAutoEvalExampleLabelingAnswers),
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
            EventPayload::ToolCall(t) => Ok(GatewayEventPayload::ToolCall(t.into())),
            EventPayload::ToolCallAuthorization(auth) => {
                Ok(GatewayEventPayload::ToolCallAuthorization(auth.try_into()?))
            }
            EventPayload::ToolResult(r) => Ok(GatewayEventPayload::ToolResult(r)),
            EventPayload::Visualization(v) => Ok(GatewayEventPayload::Visualization(v)),
            EventPayload::UserQuestions(q) => Ok(GatewayEventPayload::UserQuestions(q)),
            EventPayload::UserQuestionsAnswers(r) => {
                Ok(GatewayEventPayload::UserQuestionsAnswers(r))
            }
            EventPayload::AutoEvalExampleLabeling(l) => {
                Ok(GatewayEventPayload::AutoEvalExampleLabeling(l))
            }
            EventPayload::AutoEvalExampleLabelingAnswers(a) => {
                Ok(GatewayEventPayload::AutoEvalExampleLabelingAnswers(a))
            }
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

/// Tool call payload as seen by gateway consumers.
/// Includes `requires_approval` which is set by the gateway based on tool whitelist config.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayEventPayloadToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
    pub side_info: AutopilotSideInfo,
    /// Whether this tool call requires manual user approval.
    /// `false` for whitelisted tools (auto-approved by gateway).
    pub requires_approval: bool,
}

impl From<EventPayloadToolCall> for GatewayEventPayloadToolCall {
    fn from(tc: EventPayloadToolCall) -> Self {
        GatewayEventPayloadToolCall {
            name: tc.name,
            arguments: tc.arguments,
            side_info: tc.side_info,
            requires_approval: true, // safe default: require approval
        }
    }
}

/// Side information required for autopilot client tools.
///
/// This should contain all IDs that might be needed as input to a tool
/// that do not need to be generated by LLMs (like the session id or a config hash).
/// We should implement this as a type that has optional or mandatory fields as needed
/// for each kind of tool
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

/// Tool result that can carry either a structured JSON value or a legacy serialized string.
///
/// New producers should use `AutopilotToolResult::typed`. Consumers should use
/// `AutopilotToolResult::value()` which returns the structured value directly
/// or parses the legacy string as a fallback.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AutopilotToolResult {
    Typed {
        result_value: JsonValue,
    },
    #[deprecated = "Use `AutopilotToolResult::typed` instead. This variant exists only for backwards compatibility with old serialized data."]
    Legacy {
        result: String,
    },
}

impl AutopilotToolResult {
    pub fn typed(value: JsonValue) -> Self {
        Self::Typed {
            result_value: value,
        }
    }

    /// Get the structured value, parsing the legacy string as a fallback.
    #[expect(deprecated)]
    pub fn value(&self) -> Result<JsonValue, serde_json::Error> {
        match self {
            Self::Typed { result_value, .. } => Ok(result_value.clone()),
            Self::Legacy { result } => serde_json::from_str(result),
        }
    }
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolCallDecisionSource {
    Ui,
    Automatic,
    Whitelist,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventPayloadToolCallAuthorization {
    pub source: ToolCallDecisionSource,
    pub tool_call_event_id: Uuid,
    pub status: ToolCallAuthorizationStatus,
    /// Populated by the server from the originating tool call event.
    /// Optional for backwards compatibility until the API is deployed with enrichment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub tool_call_name: Option<String>,
    /// Populated by the server from the originating tool call event.
    /// Optional for backwards compatibility until the API is deployed with enrichment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub tool_call_arguments: Option<serde_json::Value>,
}

/// Minimal input payload for creating a tool call authorization event.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateEventPayloadToolCallAuthorization {
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
    /// Populated by the server from the originating tool call event.
    /// Optional for backwards compatibility until the API is deployed with enrichment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub tool_call_name: Option<String>,
    /// Populated by the server from the originating tool call event.
    /// Optional for backwards compatibility until the API is deployed with enrichment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub tool_call_arguments: Option<serde_json::Value>,
}

impl TryFrom<EventPayloadToolCallAuthorization> for GatewayEventPayloadToolCallAuthorization {
    type Error = &'static str;

    fn try_from(auth: EventPayloadToolCallAuthorization) -> Result<Self, Self::Error> {
        Ok(GatewayEventPayloadToolCallAuthorization {
            source: auth.source,
            tool_call_event_id: auth.tool_call_event_id,
            status: auth.status.try_into()?,
            tool_call_name: auth.tool_call_name,
            tool_call_arguments: auth.tool_call_arguments,
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
        error: tensorzero_types::ToolFailure,
    },
    Missing,
    #[serde(other)]
    #[serde(alias = "other")] // legacy name
    Unknown,
}

// =============================================================================
// Visualization Types
// =============================================================================

/// Summary statistics for a variant's performance.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct VariantSummary {
    /// Estimated mean performance.
    pub mean_est: f64,
    /// Lower confidence bound.
    pub cs_lower: f64,
    /// Upper confidence bound.
    pub cs_upper: f64,
    /// Number of observations.
    pub count: u64,
    /// Whether this variant failed during evaluation.
    #[serde(default)]
    pub failed: bool,
}

/// Visualization data for a top-k evaluation.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct TopKEvaluationVisualization {
    /// Map of variant names to their summary statistics.
    pub variant_summaries: std::collections::HashMap<String, VariantSummary>,
    /// Sizes k where we can confidently identify a top-k set.
    /// For example, [2, 5] means there's statistical separation after the 2nd
    /// and 5th ranked variants (sorted by lower confidence bound descending).
    #[serde(default)]
    pub confident_top_k_sizes: Vec<usize>,
    /// Explanation of the results for the user.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub summary_text: Option<String>,
}

/// Types of visualizations that can be displayed.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(
    feature = "ts-bindings",
    ts(export, tag = "type", rename_all = "snake_case")
)]
pub enum VisualizationType {
    /// Top-k evaluation results showing variant performance comparisons.
    TopKEvaluation(TopKEvaluationVisualization),
    /// Unknown visualization type for forward compatibility.
    /// Old clients can gracefully handle new visualization types they don't recognize.
    #[serde(untagged)]
    Unknown(serde_json::Value),
}

/// Visualization payload for an event.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct EventPayloadVisualization {
    /// The ID of the tool execution that generated this visualization.
    /// For client-side tools, this is the ToolCall event ID.
    /// For server-side tools, this is the task ID.
    pub tool_execution_id: Uuid,
    /// The visualization data.
    pub visualization: VisualizationType,
}

// =============================================================================
// Question Types
// =============================================================================

/// Questions payload for an event.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct EventPayloadUserQuestions {
    pub questions: Vec<EventPayloadUserQuestion>,
}

/// A single question to display to the user.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct EventPayloadUserQuestion {
    pub id: Uuid,
    /// Very short label displayed as a chip/tag (max 12 chars). Examples: "Auth method", "Library", "Approach".
    pub header: String,
    /// The complete question to ask the user. Should be clear, specific, and end with a question mark.
    pub question: String,
    #[serde(flatten)]
    pub inner: EventPayloadUserQuestionInner,
}

/// The format of a user question.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub enum EventPayloadUserQuestionInner {
    MultipleChoice(MultipleChoiceQuestion),
    FreeResponse,
}

/// A multiple choice question with options.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct MultipleChoiceQuestion {
    /// Should be 2-4 options.
    pub options: Vec<MultipleChoiceOption>,
    /// Set to true to allow the user to select multiple options instead of just one.
    pub multi_select: bool,
}

/// An option in a multiple choice question.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct MultipleChoiceOption {
    pub id: Uuid,
    /// The display text for this option that the user will see and select. Should be concise (1-5 words).
    pub label: String,
    /// Explanation of what this option means or what will happen if chosen.
    pub description: String,
}

/// User responses payload for an event.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct EventPayloadUserQuestionsAnswers {
    /// Map from question UUID to response.
    pub responses: HashMap<Uuid, UserQuestionAnswer>,
    pub user_questions_event_id: Uuid,
}

/// A user's response to a question.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub enum UserQuestionAnswer {
    MultipleChoice(MultipleChoiceAnswer),
    FreeResponse(FreeResponseAnswer),
    Skipped,
}

/// A user's answer to a multiple choice question.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct MultipleChoiceAnswer {
    /// IDs of the selected options.
    pub selected: Vec<Uuid>,
}

/// A user's free-form text answer.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct FreeResponseAnswer {
    pub text: String,
}

// =============================================================================
// AutoEval Example Labeling Types
// =============================================================================

/// Payload for an autoeval example labeling event.
///
/// Groups labeled examples together, each with rich context blocks
/// (e.g. prompt/response) and associated labeling questions.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct EventPayloadAutoEvalExampleLabeling {
    pub examples: Vec<AutoEvalExampleLabeling>,
}

/// A single example to label, with context and a structured labeling question.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct AutoEvalExampleLabeling {
    /// Rich content blocks providing context (e.g. the prompt and response).
    pub context: Vec<AutoEvalContentBlock>,
    /// The multiple-choice labeling question for this example.
    pub label_question: AutoEvalLabelQuestion,
    /// An optional free-response explanation question.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub explanation_question: Option<AutoEvalExplanationQuestion>,
}

/// A multiple-choice labeling question within an autoeval example.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct AutoEvalLabelQuestion {
    pub id: Uuid,
    pub header: String,
    pub question: String,
    pub options: Vec<MultipleChoiceOption>,
}

/// A free-response explanation question within an autoeval example.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct AutoEvalExplanationQuestion {
    pub id: Uuid,
    pub header: String,
    pub question: String,
}

/// A block of rich content displayed alongside an autoeval example.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(
    feature = "ts-bindings",
    ts(export, tag = "type", rename_all = "snake_case")
)]
pub enum AutoEvalContentBlock {
    /// Rendered as formatted markdown.
    Markdown {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[cfg_attr(feature = "ts-bindings", ts(optional))]
        label: Option<String>,
    },
    /// Rendered as a formatted JSON viewer.
    Json {
        data: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[cfg_attr(feature = "ts-bindings", ts(optional))]
        label: Option<String>,
    },
}

/// Minimal input payload for submitting autoeval example labeling answers.
/// The server enriches this with context from the original labeling event before storing.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct CreateEventPayloadAutoEvalExampleLabelingAnswers {
    /// Map from question UUID to response.
    pub responses: HashMap<Uuid, UserQuestionAnswer>,
    /// The event ID of the original `AutoEvalExampleLabeling` event these answers correspond to.
    pub auto_eval_example_labeling_event_id: Uuid,
}

/// Self-contained read-only payload for labeled autoeval examples.
/// Includes the full context blocks so the UI can render everything
/// without looking up the original labeling event.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct EventPayloadAutoEvalExampleLabelingAnswers {
    pub examples: Vec<AutoEvalLabeledExample>,
    /// The event ID of the original `AutoEvalExampleLabeling` event these answers correspond to.
    pub auto_eval_example_labeling_event_id: Uuid,
}

/// A labeled example with its full context and submitted answers.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct AutoEvalLabeledExample {
    /// Rich content blocks providing context (e.g. the prompt and response).
    pub context: Vec<AutoEvalContentBlock>,
    /// The multiple-choice labeling question for this example.
    pub label_question: AutoEvalLabelQuestion,
    /// The user's answer to the label question.
    pub label_answer: UserQuestionAnswer,
    /// An optional free-response explanation question.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub explanation_question: Option<AutoEvalExplanationQuestion>,
    /// The user's answer to the explanation question, if one was present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub explanation_answer: Option<UserQuestionAnswer>,
}

// =============================================================================
// Request Types
// =============================================================================

/// Request body for creating an event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateEventRequest {
    pub deployment_id: String,
    pub tensorzero_version: String,
    pub payload: CreateEventPayload,
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
    /// All user_questions events that do not have a matching user_questions_answers event.
    #[serde(default)]
    pub pending_user_questions: Vec<Event>,
    /// All auto_eval_example_labeling events that do not have a matching answers event.
    #[serde(default)]
    pub pending_auto_eval_example_labeling: Vec<Event>,
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
    /// All user_questions events that do not have a matching user_questions_answers event.
    #[serde(default)]
    pub pending_user_questions: Vec<GatewayEvent>,
    /// All auto_eval_example_labeling events that do not have a matching answers event.
    #[serde(default)]
    pub pending_auto_eval_example_labeling: Vec<GatewayEvent>,
}

/// Response from listing sessions.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct ListSessionsResponse {
    pub sessions: Vec<Session>,
}

/// Query parameters for listing config writes.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct ListConfigWritesParams {
    /// Maximum number of config writes to return. Defaults to 20.
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    /// Offset for pagination.
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<u32>,
}

/// Internal response type - consumers should use `GatewayListConfigWritesResponse` instead.
///
/// Note: TS derive is needed for types that reference this, but we don't export it.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListConfigWritesResponse {
    pub config_writes: Vec<Event>,
}

/// Response from listing config writes as seen by gateway consumers.
///
/// Uses `GatewayEvent` which excludes `NotAvailable` authorization status.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GatewayListConfigWritesResponse {
    pub config_writes: Vec<GatewayEvent>,
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
// S3 Upload Types
// =============================================================================

/// Request body for initiating an S3 upload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S3UploadRequest {
    pub tool_call_event_id: Uuid,
}

/// Response from initiating an S3 upload, containing temporary credentials.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S3UploadResponse {
    pub bucket: String,
    pub key: String,
    pub region: String,
    pub endpoint: Option<String>,
    pub virtual_hosted_style_request: Option<bool>,
    pub allow_http: Option<bool>,
    // Credentials can be null when running locally
    pub access_key_id: Option<String>,
    pub secret_access_key: Option<String>,
    pub session_token: Option<String>,
    pub credential_expiration: DateTime<Utc>,
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
