//! Autopilot API proxy endpoints.
//!
//! These endpoints proxy requests to the TensorZero Autopilot API.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::response::IntoResponse;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use futures::stream::StreamExt;
use tracing::instrument;
use uuid::Uuid;

use autopilot_client::{
    AutopilotClient, CreateEventRequest, CreateEventResponse, Event, ListEventsParams,
    ListEventsResponse, ListSessionsParams, ListSessionsResponse, StreamEventsParams,
};

use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

/// Helper to get the autopilot client or return an error.
fn get_autopilot_client(app_state: &AppStateData) -> Result<Arc<AutopilotClient>, Error> {
    app_state
        .autopilot_client
        .clone()
        .ok_or_else(|| Error::new(ErrorDetails::AutopilotUnavailable))
}

/// Handler for `GET /internal/autopilot/v1/sessions`
///
/// Lists sessions from the Autopilot API.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "autopilot.list_sessions", skip_all)]
pub async fn list_sessions_handler(
    State(app_state): AppState,
    Query(params): Query<ListSessionsParams>,
) -> Result<Json<ListSessionsResponse>, Error> {
    let client = get_autopilot_client(&app_state)?;
    let response = client.list_sessions(params).await?;
    Ok(Json(response))
}

/// Handler for `GET /internal/autopilot/v1/sessions/{session_id}/events`
///
/// Lists events for a session from the Autopilot API.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "autopilot.list_events", skip_all, fields(session_id = %session_id))]
pub async fn list_events_handler(
    State(app_state): AppState,
    Path(session_id): Path<Uuid>,
    Query(params): Query<ListEventsParams>,
) -> Result<Json<ListEventsResponse>, Error> {
    let client = get_autopilot_client(&app_state)?;
    let response = client.list_events(session_id, params).await?;
    Ok(Json(response))
}

/// Handler for `POST /internal/autopilot/v1/sessions/{session_id}/events`
///
/// Creates an event in a session via the Autopilot API.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "autopilot.create_event", skip_all, fields(session_id = %session_id))]
pub async fn create_event_handler(
    State(app_state): AppState,
    Path(session_id): Path<Uuid>,
    StructuredJson(request): StructuredJson<CreateEventRequest>,
) -> Result<Json<CreateEventResponse>, Error> {
    let client = get_autopilot_client(&app_state)?;
    let response = client.create_event(session_id, request).await?;
    Ok(Json(response))
}

/// Handler for `GET /internal/autopilot/v1/sessions/{session_id}/events/stream`
///
/// Streams events for a session via SSE from the Autopilot API.
/// Note: The #[instrument] macro is not used here due to lifetime issues with the SSE stream.
#[axum::debug_handler(state = AppStateData)]
pub async fn stream_events_handler(
    State(app_state): AppState,
    Path(session_id): Path<Uuid>,
    Query(params): Query<StreamEventsParams>,
) -> Result<impl IntoResponse, Error> {
    tracing::info!(session_id = %session_id, "autopilot.stream_events");
    let client = get_autopilot_client(&app_state)?;
    let stream = client.stream_events(session_id, params).await?;

    // Convert the autopilot event stream to SSE events
    let sse_stream =
        stream.map(
            |result: Result<Event, autopilot_client::AutopilotError>| match result {
                Ok(event) => match serde_json::to_string(&event) {
                    Ok(data) => Ok(SseEvent::default().event("event").data(data)),
                    Err(e) => {
                        tracing::error!("Failed to serialize autopilot event: {}", e);
                        Err(Error::new(ErrorDetails::Serialization {
                            message: e.to_string(),
                        }))
                    }
                },
                Err(e) => {
                    tracing::error!("Autopilot stream error: {}", e);
                    Err(Error::from(e))
                }
            },
        );

    Ok(Sse::new(sse_stream).keep_alive(KeepAlive::new()))
}
