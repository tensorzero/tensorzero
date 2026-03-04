//! Autopilot API proxy endpoints.
//!
//! These endpoints proxy requests to the TensorZero Autopilot API.
//!
//! This module provides both HTTP handlers and core functions that can be called
//! directly by the embedded client.

use std::sync::{Arc, LazyLock};

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::response::IntoResponse;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use futures::stream::StreamExt;
use regex::Regex;
use serde::{Deserialize, Serialize};
use tracing::instrument;
use uuid::Uuid;

use autopilot_client::{
    ApproveAllToolCallsRequest, ApproveAllToolCallsResponse, AutopilotClient, CreateEventRequest,
    CreateEventResponse, EventPayload, EventPayloadMessageContent, EventPayloadMessageMetadata,
    GatewayListConfigWritesResponse, GatewayListEventsResponse, GatewayStreamUpdate,
    ListConfigWritesParams, ListEventsParams, ListSessionsParams, ListSessionsResponse,
    S3UploadRequest, S3UploadResponse, StreamEventsParams,
};
use tensorzero_types::ResolveUuidResponse;

use crate::db::resolve_uuid::ResolveUuidQueries;
use crate::endpoints::status::TENSORZERO_VERSION;
use crate::error::{DisplayOrDebugGateway, Error, ErrorDetails};
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

// UUID regex: 8-4-4-4-12 hex pattern
static UUID_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    #[expect(clippy::expect_used)]
    Regex::new(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
        .expect("UUID regex should be valid")
});

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

/// HTTP request body for initiating an S3 upload.
///
/// This is the request type used by the HTTP handler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S3InitiateUploadGatewayRequest {
    pub tool_call_event_id: Uuid,
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

/// Extract unique UUIDs from message text content.
fn extract_uuids_from_content(content: &[EventPayloadMessageContent]) -> Vec<Uuid> {
    let mut uuid_set = std::collections::HashSet::new();
    for block in content {
        let EventPayloadMessageContent::Text(text) = block;
        for mat in UUID_REGEX.find_iter(&text.text) {
            if let Ok(id) = mat.as_str().parse::<Uuid>() {
                uuid_set.insert(id);
            }
        }
    }
    let mut uuids: Vec<Uuid> = uuid_set.into_iter().collect();
    uuids.sort();
    uuids
}

/// Resolve a list of UUIDs via the database, returning a `ResolveUuidResponse` for each.
///
/// Resolution failures for individual UUIDs are logged and skipped (best-effort).
async fn resolve_uuids(
    uuids: Vec<Uuid>,
    database: &(dyn ResolveUuidQueries + Sync),
) -> Vec<ResolveUuidResponse> {
    let futures = uuids.into_iter().map(|id| async move {
        match database.resolve_uuid(&id).await {
            Ok(object_types) => Some(ResolveUuidResponse { id, object_types }),
            Err(e) => {
                tracing::warn!(uuid = %id, error = %e, "Failed to resolve UUID in message");
                None
            }
        }
    });

    futures::future::join_all(futures)
        .await
        .into_iter()
        .flatten()
        .collect()
}

/// Extract UUIDs from message text content and resolve each one via the database.
///
/// Returns a `ResolveUuidResponse` for each unique UUID found. Resolution failures
/// for individual UUIDs are logged and skipped (best-effort).
async fn resolve_uuids_in_message(
    content: &[EventPayloadMessageContent],
    app_state: &AppStateData,
) -> Vec<ResolveUuidResponse> {
    let uuids = extract_uuids_from_content(content);

    if uuids.is_empty() {
        return Vec::new();
    }

    let database = app_state.get_delegating_database();

    resolve_uuids(uuids, &database).await
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
///
/// `beta_tools` is forwarded as the `tensorzero-beta-tools` header to the remote server.
pub async fn create_event(
    autopilot_client: &AutopilotClient,
    session_id: Uuid,
    request: CreateEventRequest,
    beta_tools: &[String],
) -> Result<CreateEventResponse, Error> {
    autopilot_client
        .create_event(session_id, request, beta_tools)
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

/// List config writes (write_config tool calls) for a session from the Autopilot API.
///
/// This is the core function called by both the HTTP handler and embedded client.
/// Returns `GatewayListConfigWritesResponse` which uses narrower types that exclude
/// `NotAvailable` authorization status.
pub async fn list_config_writes(
    autopilot_client: &AutopilotClient,
    session_id: Uuid,
    params: ListConfigWritesParams,
) -> Result<GatewayListConfigWritesResponse, Error> {
    autopilot_client
        .list_config_writes(session_id, params)
        .await
        .map_err(Error::from)
}

/// Initiate an S3 upload via the Autopilot API.
///
/// This is the core function called by both the HTTP handler and embedded client.
pub async fn s3_initiate_upload(
    autopilot_client: &AutopilotClient,
    session_id: Uuid,
    request: S3UploadRequest,
) -> Result<S3UploadResponse, Error> {
    autopilot_client
        .s3_initiate_upload(session_id, request)
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
/// The `tensorzero-beta-tools` header, if present, is forwarded to the remote server.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "autopilot.create_event", skip_all, fields(session_id = %session_id))]
pub async fn create_event_handler(
    State(app_state): AppState,
    Path(session_id): Path<Uuid>,
    headers: axum::http::HeaderMap,
    StructuredJson(mut http_request): StructuredJson<CreateEventGatewayRequest>,
) -> Result<Json<CreateEventResponse>, Error> {
    let client = get_autopilot_client(&app_state)?;

    // Extract beta tools from the incoming request header
    let beta_tools: Vec<String> = headers
        .get("tensorzero-beta-tools")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.split(',').map(|t| t.trim().to_string()).collect())
        .unwrap_or_default();

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

    // Enrich Message payloads with resolved UUIDs
    if let EventPayload::Message(msg) = &mut http_request.payload {
        if !msg.metadata.resolved_uuids.is_empty() {
            // We require this to be empty when invoked from the UI,
            // since we want to fill it in ourselves.
            // This avoids the need to duplicate the entire type hierarchy
            return Err(Error::new(ErrorDetails::Serialization {
                message: "`resolved_uuids` must be empty for an incoming event".to_string(),
            }));
        }
        let resolved_uuids = resolve_uuids_in_message(&msg.content, &app_state).await;
        msg.metadata = EventPayloadMessageMetadata { resolved_uuids };
    }

    let request = CreateEventRequest {
        deployment_id,
        tensorzero_version: TENSORZERO_VERSION.to_string(),
        payload: http_request.payload,
        previous_user_message_event_id: http_request.previous_user_message_event_id,
        config_snapshot_hash,
    };

    let response = create_event(&client, session_id, request, &beta_tools).await?;
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

/// Handler for `GET /internal/autopilot/v1/sessions/{session_id}/config-writes`
///
/// Lists config writes (write_config tool calls) for a session from the Autopilot API.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "autopilot.list_config_writes", skip_all, fields(session_id = %session_id))]
pub async fn list_config_writes_handler(
    State(app_state): AppState,
    Path(session_id): Path<Uuid>,
    Query(params): Query<ListConfigWritesParams>,
) -> Result<Json<GatewayListConfigWritesResponse>, Error> {
    let client = get_autopilot_client(&app_state)?;
    let response = list_config_writes(&client, session_id, params).await?;
    Ok(Json(response))
}

/// Handler for `POST /internal/autopilot/v1/sessions/{session_id}/aws/s3_initiate_upload`
///
/// Initiates an S3 upload via the Autopilot API.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "autopilot.s3_initiate_upload", skip_all)]
pub async fn s3_initiate_upload_handler(
    State(app_state): AppState,
    Path(session_id): Path<Uuid>,
    StructuredJson(http_request): StructuredJson<S3InitiateUploadGatewayRequest>,
) -> Result<Json<S3UploadResponse>, Error> {
    let client = get_autopilot_client(&app_state)?;
    let request = S3UploadRequest {
        tool_call_event_id: http_request.tool_call_event_id,
    };
    let response = s3_initiate_upload(&client, session_id, request).await?;
    Ok(Json(response))
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
    use crate::db::delegating_connection::PrimaryDatastore;
    use crate::db::postgres::PostgresConnectionInfo;
    use crate::db::resolve_uuid::ResolvedObject;
    use crate::db::valkey::ValkeyConnectionInfo;
    use crate::http::TensorzeroHttpClient;
    use crate::inference::types::FunctionType;
    use async_trait::async_trait;
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
            ValkeyConnectionInfo::Disabled,
            TaskTracker::new(),
            CancellationToken::new(),
            PrimaryDatastore::ClickHouse,
        )
        .unwrap()
    }

    fn make_text_content(texts: &[&str]) -> Vec<EventPayloadMessageContent> {
        texts
            .iter()
            .map(|t| {
                EventPayloadMessageContent::Text(tensorzero_types::Text {
                    text: t.to_string(),
                })
            })
            .collect()
    }

    /// Mock database that returns configurable results for `resolve_uuid`.
    struct MockResolveUuidDb {
        /// Map from UUID to the resolved object types to return.
        responses: std::collections::HashMap<Uuid, Vec<ResolvedObject>>,
    }

    #[async_trait]
    impl ResolveUuidQueries for MockResolveUuidDb {
        async fn resolve_uuid(&self, id: &Uuid) -> Result<Vec<ResolvedObject>, Error> {
            Ok(self.responses.get(id).cloned().unwrap_or_default())
        }
    }

    /// Mock database that always returns an error for `resolve_uuid`.
    struct FailingResolveUuidDb;

    #[async_trait]
    impl ResolveUuidQueries for FailingResolveUuidDb {
        async fn resolve_uuid(&self, _id: &Uuid) -> Result<Vec<ResolvedObject>, Error> {
            Err(Error::new(ErrorDetails::ClickHouseQuery {
                message: "mock database error".to_string(),
            }))
        }
    }

    // =========================================================================
    // get_autopilot_client tests
    // =========================================================================

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

    // =========================================================================
    // UUID regex tests
    // =========================================================================

    #[test]
    fn test_uuid_regex_matches_valid_uuids() {
        let valid = "550e8400-e29b-41d4-a716-446655440000";
        assert!(
            UUID_REGEX.is_match(valid),
            "Regex should match a valid UUID"
        );
    }

    #[test]
    fn test_uuid_regex_matches_uppercase_uuids() {
        let upper = "550E8400-E29B-41D4-A716-446655440000";
        assert!(
            UUID_REGEX.is_match(upper),
            "Regex should match an uppercase UUID"
        );
    }

    #[test]
    fn test_uuid_regex_does_not_match_short_strings() {
        assert!(
            !UUID_REGEX.is_match("not-a-uuid"),
            "Regex should not match a short non-UUID string"
        );
    }

    #[test]
    fn test_uuid_regex_finds_uuid_embedded_in_text() {
        let text = "Look at inference 550e8400-e29b-41d4-a716-446655440000 for details";
        let matches: Vec<_> = UUID_REGEX.find_iter(text).collect();
        assert_eq!(
            matches.len(),
            1,
            "Should find exactly one UUID in surrounding text"
        );
        assert_eq!(matches[0].as_str(), "550e8400-e29b-41d4-a716-446655440000");
    }

    #[test]
    fn test_uuid_regex_finds_multiple_uuids() {
        let text = "Compare 550e8400-e29b-41d4-a716-446655440000 with aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";
        let matches: Vec<_> = UUID_REGEX.find_iter(text).collect();
        assert_eq!(matches.len(), 2, "Should find two UUIDs in text");
    }

    // =========================================================================
    // extract_uuids_from_content tests
    // =========================================================================

    #[test]
    fn test_extract_uuids_from_content_empty() {
        let content: Vec<EventPayloadMessageContent> = vec![];
        let uuids = extract_uuids_from_content(&content);
        assert!(
            uuids.is_empty(),
            "Should return empty vec for empty content"
        );
    }

    #[test]
    fn test_extract_uuids_from_content_no_uuids() {
        let content = make_text_content(&["Hello, no UUIDs here!"]);
        let uuids = extract_uuids_from_content(&content);
        assert!(
            uuids.is_empty(),
            "Should return empty vec when no UUIDs are present"
        );
    }

    #[test]
    fn test_extract_uuids_from_content_single_uuid() {
        let content = make_text_content(&["Check inference 550e8400-e29b-41d4-a716-446655440000"]);
        let uuids = extract_uuids_from_content(&content);
        assert_eq!(uuids.len(), 1, "Should extract exactly one UUID");
        assert_eq!(
            uuids[0],
            "550e8400-e29b-41d4-a716-446655440000"
                .parse::<Uuid>()
                .unwrap(),
            "Extracted UUID should match the one in text"
        );
    }

    #[test]
    fn test_extract_uuids_from_content_deduplicates() {
        let uuid_str = "550e8400-e29b-41d4-a716-446655440000";
        let content = make_text_content(&[
            &format!("First: {uuid_str}"),
            &format!("Second: {uuid_str}"),
        ]);
        let uuids = extract_uuids_from_content(&content);
        assert_eq!(
            uuids.len(),
            1,
            "Duplicate UUIDs should be deduplicated to one"
        );
    }

    #[test]
    fn test_extract_uuids_from_content_multiple_distinct() {
        let uuid1: Uuid = "550e8400-e29b-41d4-a716-446655440000".parse().unwrap();
        let uuid2: Uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee".parse().unwrap();
        let content = make_text_content(&[&format!("A: {uuid1} B: {uuid2}")]);
        let uuids = extract_uuids_from_content(&content);
        assert_eq!(uuids.len(), 2, "Should extract two distinct UUIDs");
        // Results are sorted
        assert!(
            uuids[0] < uuids[1],
            "UUIDs should be sorted in ascending order"
        );
        assert!(uuids.contains(&uuid1), "Should contain the first UUID");
        assert!(uuids.contains(&uuid2), "Should contain the second UUID");
    }

    #[test]
    fn test_extract_uuids_from_content_across_multiple_blocks() {
        let uuid1: Uuid = "550e8400-e29b-41d4-a716-446655440000".parse().unwrap();
        let uuid2: Uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee".parse().unwrap();
        let content = make_text_content(&[
            &format!("Block one: {uuid1}"),
            &format!("Block two: {uuid2}"),
        ]);
        let uuids = extract_uuids_from_content(&content);
        assert_eq!(
            uuids.len(),
            2,
            "Should extract UUIDs from multiple text blocks"
        );
        assert!(
            uuids.contains(&uuid1),
            "Should contain UUID from first block"
        );
        assert!(
            uuids.contains(&uuid2),
            "Should contain UUID from second block"
        );
    }

    // =========================================================================
    // resolve_uuids tests (with mock database)
    // =========================================================================

    #[tokio::test]
    async fn test_resolve_uuids_with_mock_database() {
        let uuid1: Uuid = "550e8400-e29b-41d4-a716-446655440000".parse().unwrap();
        let uuid2: Uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee".parse().unwrap();

        let mut responses = std::collections::HashMap::new();
        responses.insert(
            uuid1,
            vec![ResolvedObject::Inference {
                function_name: "my_function".to_string(),
                function_type: FunctionType::Chat,
                variant_name: "my_variant".to_string(),
                episode_id: Uuid::nil(),
            }],
        );
        responses.insert(uuid2, vec![ResolvedObject::Episode]);

        let db = MockResolveUuidDb { responses };
        let result = resolve_uuids(vec![uuid1, uuid2], &db).await;

        assert_eq!(result.len(), 2, "Should return two resolved responses");

        let r1 = result
            .iter()
            .find(|r| r.id == uuid1)
            .expect("Should contain uuid1");
        assert_eq!(
            r1.object_types.len(),
            1,
            "uuid1 should have one object type"
        );
        assert!(
            matches!(&r1.object_types[0], ResolvedObject::Inference { function_name, .. } if function_name == "my_function"),
            "uuid1 should resolve to an Inference with function_name `my_function`"
        );

        let r2 = result
            .iter()
            .find(|r| r.id == uuid2)
            .expect("Should contain uuid2");
        assert_eq!(
            r2.object_types.len(),
            1,
            "uuid2 should have one object type"
        );
        assert!(
            matches!(&r2.object_types[0], ResolvedObject::Episode),
            "uuid2 should resolve to an Episode"
        );
    }

    #[tokio::test]
    async fn test_resolve_uuids_unknown_uuid_returns_empty_object_types() {
        let uuid1: Uuid = "550e8400-e29b-41d4-a716-446655440000".parse().unwrap();
        let db = MockResolveUuidDb {
            responses: std::collections::HashMap::new(),
        };
        let result = resolve_uuids(vec![uuid1], &db).await;

        assert_eq!(
            result.len(),
            1,
            "Should return a response even for unknown UUIDs"
        );
        assert_eq!(result[0].id, uuid1, "Returned UUID should match the input");
        assert!(
            result[0].object_types.is_empty(),
            "Unknown UUID should have empty object types"
        );
    }

    #[tokio::test]
    async fn test_resolve_uuids_skips_failed_resolutions() {
        let db = FailingResolveUuidDb;
        let uuid1: Uuid = "550e8400-e29b-41d4-a716-446655440000".parse().unwrap();
        let result = resolve_uuids(vec![uuid1], &db).await;

        assert!(
            result.is_empty(),
            "Should skip UUIDs that fail to resolve (best-effort)"
        );
    }

    #[tokio::test]
    async fn test_resolve_uuids_empty_input() {
        let db = MockResolveUuidDb {
            responses: std::collections::HashMap::new(),
        };
        let result = resolve_uuids(vec![], &db).await;
        assert!(result.is_empty(), "Should return empty vec for no UUIDs");
    }

    // =========================================================================
    // resolve_uuids_in_message integration tests (with disabled database)
    // =========================================================================

    #[tokio::test]
    async fn test_resolve_uuids_in_message_no_uuids() {
        let app_state = make_test_app_state_without_autopilot();
        let content = make_text_content(&["Hello, no UUIDs here!"]);
        let result = resolve_uuids_in_message(&content, &app_state).await;
        assert!(
            result.is_empty(),
            "Should return empty vec when no UUIDs are in the message"
        );
    }

    #[tokio::test]
    async fn test_resolve_uuids_in_message_returns_response_per_uuid() {
        let app_state = make_test_app_state_without_autopilot();
        let uuid1: Uuid = "550e8400-e29b-41d4-a716-446655440000".parse().unwrap();
        let uuid2: Uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee".parse().unwrap();
        let content = make_text_content(&[&format!("See {uuid1} and {uuid2}")]);
        let result = resolve_uuids_in_message(&content, &app_state).await;

        assert_eq!(
            result.len(),
            2,
            "Should return one ResolveUuidResponse per unique UUID"
        );
        let ids: Vec<Uuid> = result.iter().map(|r| r.id).collect();
        assert!(ids.contains(&uuid1), "Should contain uuid1");
        assert!(ids.contains(&uuid2), "Should contain uuid2");
        for r in &result {
            assert!(
                r.object_types.is_empty(),
                "Object types should be empty with disabled database for UUID {}",
                r.id
            );
        }
    }
}
