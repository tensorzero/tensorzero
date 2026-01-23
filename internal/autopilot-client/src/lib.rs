//! TensorZero Autopilot API client.
//!
//! This crate provides a client for interacting with the TensorZero Autopilot API.
//!
//! # Example
//!
//! ```no_run
//! use autopilot_client::{
//!     AutopilotClient, CreateEventRequest, EventPayload, EventPayloadMessage,
//!     EventPayloadMessageContent, Role, Text,
//! };
//! use uuid::Uuid;
//!
//! # async fn example() -> Result<(), autopilot_client::AutopilotError> {
//! // Create a client
//! let client = AutopilotClient::builder()
//!     .api_key("your-api-key")
//!     .spawn_database_url("postgres://localhost:5432/tensorzero")
//!     .build()
//!     .await?;
//!
//! // Create a new session by sending an event with a nil session ID
//! let response = client.create_event(
//!     Uuid::nil(),
//!     CreateEventRequest {
//!         deployment_id: Uuid::now_v7().to_string(),
//!         tensorzero_version: "2025.1.0".to_string(),
//!         config_snapshot_hash: Some("abc123".to_string()),
//!         payload: EventPayload::Message(EventPayloadMessage {
//!             role: Role::User,
//!             content: vec![EventPayloadMessageContent::Text(Text {
//!                 text: "Hello!".to_string(),
//!             })],
//!         }),
//!         previous_user_message_event_id: None,
//!     },
//! ).await?;
//!
//! println!("Created event {} in session {}", response.event_id, response.session_id);
//! # Ok(())
//! # }
//! ```

use std::collections::HashSet;

mod client;
mod error;
mod reject_missing_tool;
mod types;

pub use client::{
    AutopilotClient, AutopilotClientBuilder, DEFAULT_BASE_URL, DEFAULT_SPAWN_QUEUE_NAME,
};
pub use error::AutopilotError;
pub use reject_missing_tool::reject_missing_tool;
pub use types::{
    ApproveAllToolCallsRequest, ApproveAllToolCallsResponse, AutopilotSideInfo, AutopilotStatus,
    AutopilotToolResult, Base64File, CreateEventRequest, CreateEventResponse, ErrorDetail,
    ErrorResponse, Event, EventPayload, EventPayloadError, EventPayloadMessage,
    EventPayloadMessageContent, EventPayloadStatusUpdate, EventPayloadToolCall,
    EventPayloadToolCallAuthorization, EventPayloadToolResult, File, ListEventsParams,
    ListEventsResponse, ListSessionsParams, ListSessionsResponse, ObjectStoragePointer,
    OptimizationWorkflowSideInfo, RawText, Role, Session, StatusUpdate, StreamEventsParams,
    StreamUpdate, Template, Text, Thought, ToolCallAuthorizationStatus, ToolCallDecisionSource,
    ToolCallWrapper, ToolOutcome, Unknown, UrlFile,
};

/// Collect all available tool names for the autopilot system.
///
/// This function returns the names of all tools available in the TensorZero
/// autopilot deployment. It's used by `AutopilotClient` to filter out unknown
/// tool calls and automatically reject them.
///
/// Note: This list must be kept in sync with the tools registered in
/// `autopilot-tools`. If a tool is added there but not here, it will be
/// auto-rejected (safe but potentially unexpected). If a name is added here
/// but the tool doesn't exist, the tool call will simply fail when executed.
pub fn collect_tool_names() -> HashSet<String> {
    let mut names = HashSet::new();

    // Production tools (from autopilot-tools/src/lib.rs for_each_tool)
    names.insert("inference".to_string());
    names.insert("feedback".to_string());
    names.insert("create_datapoints".to_string());
    names.insert("create_datapoints_from_inferences".to_string());
    names.insert("list_datapoints".to_string());
    names.insert("get_datapoints".to_string());
    names.insert("update_datapoints".to_string());
    names.insert("delete_datapoints".to_string());
    names.insert("launch_optimization_workflow".to_string());
    names.insert("get_latest_feedback_by_metric".to_string());
    names.insert("get_feedback_by_variant".to_string());
    names.insert("run_evaluation".to_string());
    names.insert("get_config".to_string());
    names.insert("write_config".to_string());
    names.insert("list_inferences".to_string());
    names.insert("get_inferences".to_string());
    // Internal tool for auto-rejecting unknown tool calls
    names.insert("__auto_reject_tool_call__".to_string());

    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_tool_names_returns_expected_tools() {
        let names = collect_tool_names();

        // Verify production tools are present
        assert!(
            names.contains("inference"),
            "Expected `inference` tool to be in the set"
        );
        assert!(
            names.contains("feedback"),
            "Expected `feedback` tool to be in the set"
        );
        assert!(
            names.contains("create_datapoints"),
            "Expected `create_datapoints` tool to be in the set"
        );
        assert!(
            names.contains("list_datapoints"),
            "Expected `list_datapoints` tool to be in the set"
        );
        assert!(
            names.contains("run_evaluation"),
            "Expected `run_evaluation` tool to be in the set"
        );
        assert!(
            names.contains("launch_optimization_workflow"),
            "Expected `launch_optimization_workflow` tool to be in the set"
        );

        // Verify internal auto-reject tool is present
        assert!(
            names.contains("__auto_reject_tool_call__"),
            "Expected `__auto_reject_tool_call__` tool to be in the set"
        );
    }

    #[test]
    fn test_collect_tool_names_does_not_contain_unknown_tools() {
        let names = collect_tool_names();

        assert!(
            !names.contains("fake_tool"),
            "Should not contain `fake_tool`"
        );
        assert!(
            !names.contains("unknown_tool"),
            "Should not contain `unknown_tool`"
        );
        assert!(!names.contains(""), "Should not contain empty string");
    }

    #[test]
    fn test_collect_tool_names_count() {
        let names = collect_tool_names();
        // 16 production tools + 1 internal auto-reject tool = 17 total
        assert_eq!(
            names.len(),
            17,
            "Expected 17 tools (16 production + 1 internal)"
        );
    }
}
