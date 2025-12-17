//! TensorZero Autopilot API client.
//!
//! This crate provides a client for interacting with the TensorZero Autopilot API.
//!
//! # Example
//!
//! ```no_run
//! use autopilot_client::{AutopilotClient, CreateEventRequest, EventPayload, UserMessagePayload};
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
//!         payload: EventPayload::UserMessage(UserMessagePayload {
//!             content: vec![serde_json::json!({"type": "text", "text": "Hello!"})],
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
    AssistantMessagePayload, CreateEventRequest, CreateEventResponse, ErrorDetail, ErrorResponse,
    Event, EventPayload, ListEventsParams, ListEventsResponse, ListSessionsParams,
    ListSessionsResponse, Session, StatusUpdate, StreamEventsParams, UserMessagePayload,
};
