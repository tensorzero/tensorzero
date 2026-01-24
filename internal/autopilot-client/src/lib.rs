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
    EventPayloadToolCallAuthorization, EventPayloadToolResult, File, GatewayEvent,
    GatewayEventPayload, GatewayEventPayloadToolCallAuthorization, GatewayListEventsResponse,
    GatewayStreamUpdate, GatewayToolCallAuthorizationStatus, ListEventsParams, ListEventsResponse,
    ListSessionsParams, ListSessionsResponse, ObjectStoragePointer, OptimizationWorkflowSideInfo,
    RawText, Role, Session, StatusUpdate, StreamEventsParams, StreamUpdate, Template, Text,
    Thought, ToolCallAuthorizationStatus, ToolCallDecisionSource, ToolCallWrapper, ToolOutcome,
    Unknown, UrlFile,
};
