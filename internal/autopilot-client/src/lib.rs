//! TensorZero Autopilot API client.
//!
//! This crate provides a client for interacting with the TensorZero Autopilot API.
//!
//! # Example
//!
//! ```no_run
//! use autopilot_client::{
//!     AutopilotClient, CreateEventRequest, EventPayload, InputMessage,
//!     InputMessageContent, Role, Text,
//! };
//! use uuid::Uuid;
//!
//! # async fn example() -> Result<(), autopilot_client::AutopilotError> {
//! // Create a client
//! let client = AutopilotClient::builder()
//!     .api_key("your-api-key")
//!     .build()?;
//!
//! // Create a new session by sending an event with a nil session ID
//! let response = client.create_event(
//!     Uuid::nil(),
//!     CreateEventRequest {
//!         deployment_id: Uuid::new_v4(),
//!         tensorzero_version: "2025.1.0".to_string(),
//!         payload: EventPayload::Message(InputMessage {
//!             role: Role::User,
//!             content: vec![InputMessageContent::Text(Text {
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
mod types;

pub use client::{AutopilotClient, AutopilotClientBuilder, DEFAULT_BASE_URL};
pub use error::AutopilotError;
pub use types::{
    Base64File, CreateEventRequest, CreateEventResponse, ErrorDetail, ErrorResponse, Event,
    EventPayload, File, InputMessage, InputMessageContent, ListEventsParams, ListEventsResponse,
    ListSessionsParams, ListSessionsResponse, ObjectStoragePointer, RawText, Role, Session,
    StatusUpdate, StreamEventsParams, Template, Text, Thought, ToolCall, ToolCallAuthorization,
    ToolCallAuthorizationStatus, ToolCallDecisionSource, ToolCallWrapper, ToolOutcome, ToolResult,
    Unknown, UrlFile,
};
