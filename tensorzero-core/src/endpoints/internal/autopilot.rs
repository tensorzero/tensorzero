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
use serde::{Deserialize, Serialize};
use tracing::instrument;
use uuid::Uuid;

use autopilot_client::{
    ApproveAllToolCallsRequest, ApproveAllToolCallsResponse, AutopilotClient, CreateEventRequest,
    CreateEventResponse, EventPayload, GatewayListEventsResponse, GatewayStreamUpdate,
    ListEventsParams, ListSessionsParams, ListSessionsResponse, StreamEventsParams,
};

use crate::endpoints::status::TENSORZERO_VERSION;
use crate::error::{DisplayOrDebugGateway, Error, ErrorDetails};
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

/// HTTP request body for creating an event.
///
/// This is the request type used by the HTTP handler. The `deployment_id` is
/// injected from the gateway's app state, so it's not included in this request.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct CreateEventGatewayRequest {
    pub payload: EventPayload,
    /// Used for idempotency when adding events to an existing session.
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[serde(default)]
    pub previous_user_message_event_id: Option<Uuid>,
}

/// HTTP request body for approving all pending tool calls.
///
/// This is the request type used by the HTTP handler. The `deployment_id` and
/// `tensorzero_version` are injected from the gateway's app state.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct ApproveAllToolCallsGatewayRequest {
    /// Only approve tool calls with event IDs <= this value.
    /// Prevents race condition where new tool calls arrive after client fetched the list.
    pub last_tool_call_event_id: Uuid,
}

/// Response for the autopilot status endpoint.
///
/// Indicates whether the autopilot client is configured (i.e., whether
/// `TENSORZERO_AUTOPILOT_API_KEY` is set).
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct AutopilotStatusResponse {
    pub enabled: bool,
}

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
/// Returns `GatewayListEventsResponse` which uses narrower types that exclude
/// `NotAvailable` authorization status.
pub async fn list_events(
    autopilot_client: &AutopilotClient,
    session_id: Uuid,
    params: ListEventsParams,
) -> Result<GatewayListEventsResponse, Error> {
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

/// Approve all pending tool calls for a session via the Autopilot API.
///
/// This is the core function called by both the HTTP handler and embedded client.
pub async fn approve_all_tool_calls(
    autopilot_client: &AutopilotClient,
    session_id: Uuid,
    request: ApproveAllToolCallsRequest,
) -> Result<ApproveAllToolCallsResponse, Error> {
    autopilot_client
        .approve_all_tool_calls(session_id, request)
        .await
        .map_err(Error::from)
}

/// Interrupt an autopilot session via the Autopilot API.
///
/// This interrupts all durable tasks associated with the session (best effort),
/// then interrupts the session via the Autopilot API.
///
/// This is the core function called by both the HTTP handler and embedded client.
pub async fn interrupt_session(
    autopilot_client: &AutopilotClient,
    session_id: Uuid,
) -> Result<(), Error> {
    // Interrupt durable tasks first (best effort - log warning on failure)
    if let Err(e) = autopilot_client
        .interrupt_tasks_for_session(session_id)
        .await
    {
        tracing::warn!(
            session_id = %session_id,
            error = %e,
            "Failed to interrupt durable tasks for session"
        );
    }

    // Then interrupt the session via autopilot API
    autopilot_client
        .interrupt_session(session_id)
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
) -> Result<Json<GatewayListEventsResponse>, Error> {
    let client = get_autopilot_client(&app_state)?;
    let response = list_events(&client, session_id, params).await?;
    Ok(Json(response))
}

/// Handler for `POST /internal/autopilot/v1/sessions/{session_id}/events`
///
/// Creates an event in a session via the Autopilot API.
/// The deployment_id is injected from the gateway's app state.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "autopilot.create_event", skip_all, fields(session_id = %session_id))]
pub async fn create_event_handler(
    State(app_state): AppState,
    Path(session_id): Path<Uuid>,
    StructuredJson(http_request): StructuredJson<CreateEventGatewayRequest>,
) -> Result<Json<CreateEventResponse>, Error> {
    let client = get_autopilot_client(&app_state)?;

    // Get deployment_id from app state
    let deployment_id = app_state
        .deployment_id
        .clone()
        .ok_or_else(|| Error::new(ErrorDetails::AutopilotUnavailable))?;

    // Construct the full request with deployment_id
    // If starting a new session (nil session_id), include the current config hash
    let config_snapshot_hash = if session_id.is_nil() {
        Some(app_state.config.hash.to_string())
    } else {
        None
    };
    let request = CreateEventRequest {
        deployment_id,
        tensorzero_version: TENSORZERO_VERSION.to_string(),
        payload: http_request.payload,
        previous_user_message_event_id: http_request.previous_user_message_event_id,
        config_snapshot_hash,
    };

    let response = create_event(&client, session_id, request).await?;
    Ok(Json(response))
}

/// Handler for `POST /internal/autopilot/v1/sessions/{session_id}/actions/approve_all`
///
/// Approves all pending tool calls for a session via the Autopilot API.
/// The deployment_id and tensorzero_version are injected from the gateway's app state.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "autopilot.approve_all_tool_calls", skip_all, fields(session_id = %session_id))]
pub async fn approve_all_tool_calls_handler(
    State(app_state): AppState,
    Path(session_id): Path<Uuid>,
    StructuredJson(http_request): StructuredJson<ApproveAllToolCallsGatewayRequest>,
) -> Result<Json<ApproveAllToolCallsResponse>, Error> {
    let client = get_autopilot_client(&app_state)?;

    let deployment_id = app_state
        .deployment_id
        .clone()
        .ok_or_else(|| Error::new(ErrorDetails::AutopilotUnavailable))?;

    let request = ApproveAllToolCallsRequest {
        deployment_id,
        tensorzero_version: TENSORZERO_VERSION.to_string(),
        last_tool_call_event_id: http_request.last_tool_call_event_id,
    };

    let response = approve_all_tool_calls(&client, session_id, request).await?;
    Ok(Json(response))
}

/// Handler for `POST /internal/autopilot/v1/sessions/{session_id}/actions/interrupt`
///
/// Interrupts an autopilot session via the Autopilot API.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "autopilot.interrupt_session", skip_all, fields(session_id = %session_id))]
pub async fn interrupt_session_handler(
    State(app_state): AppState,
    Path(session_id): Path<Uuid>,
) -> Result<(), Error> {
    let client = get_autopilot_client(&app_state)?;
    interrupt_session(&client, session_id).await
}

/// Handler for `GET /internal/autopilot/status`
///
/// Returns whether autopilot is configured (i.e., whether `TENSORZERO_AUTOPILOT_API_KEY` is set).
/// This endpoint does not require authentication and does not make any external calls.
#[instrument(name = "autopilot.status", skip_all)]
pub async fn autopilot_status_handler(State(app_state): AppState) -> Json<AutopilotStatusResponse> {
    Json(AutopilotStatusResponse {
        enabled: app_state.autopilot_client.is_some(),
    })
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
    let sse_stream = stream.map(
        |result: Result<GatewayStreamUpdate, autopilot_client::AutopilotError>| match result {
            Ok(event) => match serde_json::to_string(&event) {
                Ok(data) => Ok(SseEvent::default().event("event").data(data)),
                Err(e) => {
                    tracing::error!(
                        "Failed to serialize autopilot event: {}",
                        DisplayOrDebugGateway::new(&e)
                    );
                    Err(Error::new(ErrorDetails::Serialization {
                        message: e.to_string(),
                    }))
                }
            },
            Err(e) => {
                tracing::error!("Autopilot stream error: {}", DisplayOrDebugGateway::new(&e));
                Err(Error::from(e))
            }
        },
    );

    // Close the stream when the server shuts down
    // We do *not* want to wait for the stream to finish when the gateway shuts down,
    // as it may stay open indefinitely (on either end - a browser ui ta might be holding open a connection)
    // Clients using the endpoint should be able to auto-reconnect, so this is fine.
    Ok(
        Sse::new(sse_stream.take_until(app_state.shutdown_token.clone().cancelled_owned()))
            .keep_alive(KeepAlive::new()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::db::clickhouse::ClickHouseConnectionInfo;
    use crate::db::postgres::PostgresConnectionInfo;
    use crate::db::valkey::ValkeyConnectionInfo;
    use crate::http::TensorzeroHttpClient;
    use tokio_util::sync::CancellationToken;
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
            ValkeyConnectionInfo::Disabled,
            TaskTracker::new(),
            CancellationToken::new(),
        )
        .unwrap()
    }

    #[test]
    fn test_get_autopilot_client_returns_unavailable_when_none() {
        let app_state = make_test_app_state_without_autopilot();
        let error = get_autopilot_client(&app_state).unwrap_err();
        assert_eq!(error.to_string(), "Autopilot credentials unavailable");
    }

    #[tokio::test]
    async fn test_autopilot_status_handler_returns_false_when_not_configured() {
        let app_state = make_test_app_state_without_autopilot();
        let response = autopilot_status_handler(State(app_state)).await;
        assert!(
            !response.enabled,
            "Expected enabled to be false when autopilot is not configured"
        );
    }
}
