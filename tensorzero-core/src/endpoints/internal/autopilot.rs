//! Autopilot API proxy endpoints.
//!
//! These endpoints proxy requests to the TensorZero Autopilot API.
//!
//! This module provides both HTTP handlers and core functions that can be called
//! directly by the embedded client.

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

// =============================================================================
// Core Functions
// =============================================================================
// These functions contain the core logic and can be called directly by the
// embedded client without going through HTTP.

/// Helper to get the autopilot client or return an error.
fn get_autopilot_client(app_state: &AppStateData) -> Result<Arc<AutopilotClient>, Error> {
    app_state
        .autopilot_client
        .clone()
        .ok_or_else(|| Error::new(ErrorDetails::AutopilotUnavailable))
}

/// List sessions from the Autopilot API.
///
/// This is the core function called by both the HTTP handler and embedded client.
pub async fn list_sessions(
    autopilot_client: &AutopilotClient,
    params: ListSessionsParams,
) -> Result<ListSessionsResponse, Error> {
    autopilot_client
        .list_sessions(params)
        .await
        .map_err(Error::from)
}

/// List events for a session from the Autopilot API.
///
/// This is the core function called by both the HTTP handler and embedded client.
pub async fn list_events(
    autopilot_client: &AutopilotClient,
    session_id: Uuid,
    params: ListEventsParams,
) -> Result<ListEventsResponse, Error> {
    autopilot_client
        .list_events(session_id, params)
        .await
        .map_err(Error::from)
}

/// Create an event in a session via the Autopilot API.
///
/// This is the core function called by both the HTTP handler and embedded client.
pub async fn create_event(
    autopilot_client: &AutopilotClient,
    session_id: Uuid,
    request: CreateEventRequest,
) -> Result<CreateEventResponse, Error> {
    autopilot_client
        .create_event(session_id, request)
        .await
        .map_err(Error::from)
}

// =============================================================================
// HTTP Handlers
// =============================================================================

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
    let response = list_sessions(&client, params).await?;
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
    let response = list_events(&client, session_id, params).await?;
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
    let response = create_event(&client, session_id, request).await?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::db::clickhouse::ClickHouseConnectionInfo;
    use crate::db::postgres::PostgresConnectionInfo;
    use crate::http::TensorzeroHttpClient;
    use tokio_util::task::TaskTracker;

    fn make_test_app_state_without_autopilot() -> AppStateData {
        let config = std::sync::Arc::new(Config::default());
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();
        let postgres_connection_info = PostgresConnectionInfo::Disabled;

        AppStateData::new_for_snapshot(
            config,
            http_client,
            clickhouse_connection_info,
            postgres_connection_info,
            TaskTracker::new(),
        )
    }

    #[test]
    fn test_get_autopilot_client_returns_unavailable_when_none() {
        let app_state = make_test_app_state_without_autopilot();
        let error = get_autopilot_client(&app_state).unwrap_err();
        assert_eq!(error.to_string(), "Autopilot credentials unavailable");
    }
}
