//! Autopilot API client implementation.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use durable_tools_spawn::{SpawnClient, SpawnOptions};
use futures::stream::{Stream, StreamExt};
use moka::sync::Cache;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use reqwest_sse_stream::RequestBuilderExt;
use secrecy::{ExposeSecret, SecretString};
use sqlx::PgPool;
use url::Url;
use uuid::Uuid;

use crate::StreamUpdate;
use crate::error::AutopilotError;
use crate::reject_missing_tool::reject_missing_tool;
use crate::types::{
    ApproveAllToolCallsRequest, ApproveAllToolCallsResponse, CreateEventRequest,
    CreateEventResponse, ErrorResponse, Event, EventPayload, EventPayloadToolCall,
    GatewayListEventsResponse, GatewayStreamUpdate, ListEventsParams, ListEventsResponse,
    ListSessionsParams, ListSessionsResponse, StreamEventsParams, ToolCallAuthorizationStatus,
};

/// Default base URL for the Autopilot API.
pub const DEFAULT_BASE_URL: &str = "https://api.autopilot.tensorzero.com";
/// Default name for the durable queue used by autopilot
pub const DEFAULT_SPAWN_QUEUE_NAME: &str = "autopilot";

/// Returns the default base URL as a parsed [`Url`].
///
/// This function is infallible because [`DEFAULT_BASE_URL`] is a valid URL.
fn default_base_url() -> Url {
    // SAFETY: DEFAULT_BASE_URL is a compile-time constant that is a valid URL.
    // This is tested in unit tests.
    #[expect(clippy::expect_used)]
    Url::parse(DEFAULT_BASE_URL).expect("DEFAULT_BASE_URL is a valid URL")
}

const DEFAULT_TOOL_CALL_CACHE_CAPACITY: u64 = 256;
const DEFAULT_TOOL_CALL_CACHE_TTL: Duration = Duration::from_secs(60 * 60);

// =============================================================================
// Client Builder
// =============================================================================

/// Builder for creating an [`AutopilotClient`].
pub struct AutopilotClientBuilder {
    base_url: Option<Url>,
    api_key: Option<SecretString>,
    http_client: Option<reqwest::Client>,
    timeout: Option<Duration>,
    spawn_pool: Option<PgPool>,
    spawn_database_url: Option<SecretString>,
    spawn_queue_name: String,
    tool_call_cache_capacity: u64,
    tool_call_cache_ttl: Duration,
    available_tools: Option<HashSet<String>>,
}

impl Default for AutopilotClientBuilder {
    fn default() -> Self {
        Self {
            base_url: None,
            api_key: None,
            http_client: None,
            timeout: None,
            spawn_pool: None,
            spawn_database_url: None,
            spawn_queue_name: DEFAULT_SPAWN_QUEUE_NAME.to_string(),
            tool_call_cache_capacity: DEFAULT_TOOL_CALL_CACHE_CAPACITY,
            tool_call_cache_ttl: DEFAULT_TOOL_CALL_CACHE_TTL,
            available_tools: None,
        }
    }
}

impl AutopilotClientBuilder {
    /// Creates a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the base URL for the API.
    ///
    /// Defaults to `https://api.autopilot.tensorzero.com`.
    pub fn base_url(mut self, url: Url) -> Self {
        self.base_url = Some(url);
        self
    }

    /// Sets the API key for authentication.
    pub fn api_key(mut self, key: impl Into<SecretString>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets a custom HTTP client.
    ///
    /// If not set, a new client will be created.
    pub fn http_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = Some(client);
        self
    }

    /// Sets the timeout for REST API requests.
    ///
    /// This timeout does NOT apply to SSE streaming connections.
    pub fn timeout(mut self, duration: Duration) -> Self {
        self.timeout = Some(duration);
        self
    }

    /// Sets the Postgres pool used for durable tool spawning.
    pub fn spawn_pool(mut self, pool: PgPool) -> Self {
        self.spawn_pool = Some(pool);
        self
    }

    /// Sets the Postgres URL used for durable tool spawning.
    pub fn spawn_database_url(mut self, url: impl Into<SecretString>) -> Self {
        self.spawn_database_url = Some(url.into());
        self
    }

    /// Sets the durable queue name for tool spawning.
    pub fn spawn_queue_name(mut self, name: impl Into<String>) -> Self {
        self.spawn_queue_name = name.into();
        self
    }

    /// Sets the tool call cache capacity.
    pub fn tool_call_cache_capacity(mut self, capacity: u64) -> Self {
        self.tool_call_cache_capacity = capacity;
        self
    }

    /// Sets the tool call cache TTL.
    pub fn tool_call_cache_ttl(mut self, ttl: Duration) -> Self {
        self.tool_call_cache_ttl = ttl;
        self
    }

    /// Sets the set of available tool names for filtering unknown tool calls.
    ///
    /// This is a required field. When tool calls are received for tools not in
    /// this set, they will be automatically rejected with a `NotAvailable` status.
    pub fn available_tools(mut self, tools: HashSet<String>) -> Self {
        self.available_tools = Some(tools);
        self
    }

    /// Builds the [`AutopilotClient`].
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The API key is not set
    /// - The available tools are not set
    /// - The HTTP client cannot be built
    /// - Durable spawn configuration is missing or invalid
    pub async fn build(self) -> Result<AutopilotClient, AutopilotError> {
        let api_key = self
            .api_key
            .ok_or(AutopilotError::MissingConfig("api_key"))?;
        let available_tools = self
            .available_tools
            .ok_or(AutopilotError::MissingConfig("available_tools"))?;

        let base_url = match self.base_url {
            Some(url) => url,
            None => default_base_url(),
        };

        // Build the REST client with timeout
        let http_client = match self.http_client {
            Some(client) => client,
            None => {
                let mut builder = reqwest::Client::builder();
                if let Some(timeout) = self.timeout {
                    builder = builder.timeout(timeout);
                }
                builder.build().map_err(AutopilotError::Request)?
            }
        };

        // Build the SSE client without timeout (for long-lived connections)
        let sse_http_client = reqwest::Client::builder()
            .build()
            .map_err(AutopilotError::Request)?;

        let mut spawn_builder = SpawnClient::builder().queue_name(self.spawn_queue_name);
        spawn_builder = if let Some(pool) = self.spawn_pool {
            spawn_builder.pool(pool)
        } else if let Some(database_url) = self.spawn_database_url {
            spawn_builder.database_url(database_url)
        } else {
            return Err(AutopilotError::MissingConfig(
                "spawn_pool or spawn_database_url",
            ));
        };

        let spawn_client = Arc::new(spawn_builder.build().await?);
        let tool_call_cache = Cache::builder()
            .max_capacity(self.tool_call_cache_capacity)
            .time_to_live(self.tool_call_cache_ttl)
            .build();

        Ok(AutopilotClient {
            http_client,
            sse_http_client,
            base_url,
            api_key,
            spawn_client,
            tool_call_cache,
            available_tools: Arc::new(available_tools),
        })
    }
}

// =============================================================================
// Client
// =============================================================================

/// Client for the TensorZero Autopilot API.
pub struct AutopilotClient {
    http_client: reqwest::Client,
    sse_http_client: reqwest::Client,
    base_url: Url,
    api_key: SecretString,
    spawn_client: Arc<SpawnClient>,
    tool_call_cache: Cache<Uuid, EventPayloadToolCall>,
    /// Set of available tool names for filtering unknown tool calls.
    available_tools: Arc<HashSet<String>>,
}

impl fmt::Debug for AutopilotClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AutopilotClient")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl AutopilotClient {
    /// Creates a new builder for constructing an [`AutopilotClient`].
    pub fn builder() -> AutopilotClientBuilder {
        AutopilotClientBuilder::new()
    }

    /// Returns the base URL of the API.
    pub fn base_url(&self) -> &Url {
        &self.base_url
    }

    fn cache_tool_call_event(&self, event: &Event) {
        if let EventPayload::ToolCall(tool_call) = &event.payload {
            self.tool_call_cache.insert(event.id, tool_call.clone());
        }
    }

    fn cache_tool_call_events(&self, events: &[Event]) {
        for event in events {
            self.cache_tool_call_event(event);
        }
    }

    /// Check if a tool call is for an unknown tool.
    fn is_unknown_tool(&self, tool_call: &EventPayloadToolCall) -> bool {
        !self.available_tools.contains(&tool_call.name)
    }

    /// Check if an event should be filtered from responses.
    ///
    /// Returns `true` for:
    /// - `ToolCallAuthorization` events with `NotAvailable` status
    /// - `ToolCall` events for unknown tools (not in `available_tools`)
    fn should_filter_event(event: &Event, available_tools: &HashSet<String>) -> bool {
        match &event.payload {
            EventPayload::ToolCallAuthorization(auth)
                if matches!(auth.status, ToolCallAuthorizationStatus::NotAvailable) =>
            {
                true
            }
            EventPayload::ToolCall(tool_call) if !available_tools.contains(&tool_call.name) => true,
            _ => false,
        }
    }

    /// Spawns a durable task to execute an approved tool call.
    ///
    /// This is called after a `ToolCallAuthorization` event with `Approved` status
    /// is successfully created. It retrieves the original tool call details and
    /// spawns the corresponding durable task for execution.
    ///
    /// # Tool Call Lookup
    ///
    /// The function first checks the local cache for the tool call. If not found,
    /// it fetches the tool call event directly from the API using `get_event`.
    ///
    /// # Side Info
    ///
    /// The spawned task receives the side info that was stored in the ToolCall event.
    /// This allows callers (e.g., autopilot sessions) to propagate tool-specific
    /// configuration to the tool executor.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Fetching the tool call event fails
    /// - The event is not a tool call
    /// - Tool call arguments cannot be parsed as JSON
    /// - Spawning the durable task fails
    async fn handle_tool_call_authorization(
        &self,
        session_id: Uuid,
        tool_call_event_id: Uuid,
    ) -> Result<(), AutopilotError> {
        // Check cache first, otherwise fetch the tool call event directly
        let autopilot_tool_call = match self.tool_call_cache.get(&tool_call_event_id) {
            Some(tc) => tc,
            None => {
                let event = self.get_event(session_id, tool_call_event_id).await?;
                match event.payload {
                    EventPayload::ToolCall(tc) => tc,
                    _ => return Err(AutopilotError::ToolCallNotFound(tool_call_event_id)),
                }
            }
        };

        let tool_name = autopilot_tool_call.name.clone();
        let llm_params = autopilot_tool_call.arguments.clone();

        // Use the side_info from the ToolCall event (propagated from caller)

        let episode_id = Uuid::now_v7();
        self.spawn_client
            .spawn_tool_by_name(
                &tool_name,
                llm_params,
                serde_json::to_value(autopilot_tool_call.side_info)?,
                episode_id,
                SpawnOptions::default(),
            )
            .await?;

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Health Endpoints
    // -------------------------------------------------------------------------

    /// Checks the API status.
    ///
    /// Returns `Ok(())` if the API is reachable.
    pub async fn status(&self) -> Result<(), AutopilotError> {
        let url = self.base_url.join("/status")?;
        let response = self.http_client.get(url).send().await?;
        self.check_response(response).await?;
        Ok(())
    }

    /// Checks the API health.
    ///
    /// Returns detailed health information.
    pub async fn health(&self) -> Result<serde_json::Value, AutopilotError> {
        let url = self.base_url.join("/health")?;
        let response = self.http_client.get(url).send().await?;
        let response = self.check_response(response).await?;
        let body = response.json().await?;
        Ok(body)
    }

    // -------------------------------------------------------------------------
    // Session Endpoints
    // -------------------------------------------------------------------------

    /// Lists sessions.
    pub async fn list_sessions(
        &self,
        params: ListSessionsParams,
    ) -> Result<ListSessionsResponse, AutopilotError> {
        let url = self.base_url.join("/v1/sessions")?;
        let response = self
            .http_client
            .get(url)
            .headers(self.auth_headers())
            .query(&params)
            .send()
            .await?;
        let response = self.check_response(response).await?;
        let body = response.json().await?;
        Ok(body)
    }

    /// Interrupts a session.
    pub async fn interrupt_session(&self, session_id: Uuid) -> Result<(), AutopilotError> {
        let url = self
            .base_url
            .join(&format!("/v1/sessions/{session_id}/actions/interrupt"))?;
        let response = self
            .http_client
            .post(url)
            .headers(self.auth_headers())
            .send()
            .await?;
        self.check_response(response).await?;
        Ok(())
    }

    /// Interrupt all durable tasks associated with a session ID.
    ///
    /// This interrupts any running durable tasks (and their recursive children)
    /// that were spawned for the given session.
    ///
    /// # Returns
    ///
    /// Returns the number of tasks that were interrupted.
    ///
    /// # Errors
    ///
    /// Returns an error if the interruption fails.
    pub async fn interrupt_tasks_for_session(
        &self,
        session_id: Uuid,
    ) -> Result<u64, AutopilotError> {
        self.spawn_client
            .interrupt_tasks_by_session_id(session_id)
            .await
            .map_err(AutopilotError::from)
    }

    // -------------------------------------------------------------------------
    // Event Endpoints
    // -------------------------------------------------------------------------

    /// Lists events for a session.
    ///
    /// Unknown tool calls (those not in `available_tools`) are filtered from
    /// both `events` and `pending_tool_calls`, and automatically rejected via a durable task.
    /// `ToolCallAuthorization::NotAvailable` events are also filtered from results.
    ///
    /// Returns `GatewayListEventsResponse` which uses `GatewayEvent` - a narrower type
    /// that excludes `NotAvailable` authorization status.
    pub async fn list_events(
        &self,
        session_id: Uuid,
        params: ListEventsParams,
    ) -> Result<GatewayListEventsResponse, AutopilotError> {
        let url = self
            .base_url
            .join(&format!("/v1/sessions/{session_id}/events"))?;
        let response = self
            .http_client
            .get(url)
            .headers(self.auth_headers())
            .query(&params)
            .send()
            .await?;
        let response = self.check_response(response).await?;
        let body: ListEventsResponse = response.json().await?;
        self.cache_tool_call_events(&body.events);
        self.cache_tool_call_events(&body.pending_tool_calls);

        // Filter out unknown tool calls and NotAvailable authorizations, then convert to gateway types.
        // We don't reject unknown tool calls from events - only from pending_tool_calls below.
        let mut filtered_events = Vec::new();
        for event in body.events {
            if Self::should_filter_event(&event, &self.available_tools) {
                continue;
            }
            // Conversion should succeed since we filtered out NotAvailable events
            let gateway_event = event.try_into().map_err(|e| {
                AutopilotError::Internal(format!("Event conversion failed after filtering: {e}"))
            })?;
            filtered_events.push(gateway_event);
        }

        // Filter pending tool calls and reject unknown ones
        let mut filtered_pending_tool_calls = Vec::new();
        for event in body.pending_tool_calls {
            // Reject unknown tool calls (but not other filtered events like NotAvailable authorizations)
            if let EventPayload::ToolCall(tool_call) = &event.payload
                && self.is_unknown_tool(tool_call)
            {
                reject_missing_tool(&self.spawn_client, tool_call).await?;
            }
            if Self::should_filter_event(&event, &self.available_tools) {
                continue;
            }
            // Conversion should succeed since we filtered out NotAvailable events
            let gateway_event = event.try_into().map_err(|e| {
                AutopilotError::Internal(format!("Event conversion failed after filtering: {e}"))
            })?;
            filtered_pending_tool_calls.push(gateway_event);
        }

        Ok(GatewayListEventsResponse {
            events: filtered_events,
            previous_user_message_event_id: body.previous_user_message_event_id,
            status: body.status,
            pending_tool_calls: filtered_pending_tool_calls,
        })
    }

    /// Gets a single event by ID.
    pub async fn get_event(
        &self,
        session_id: Uuid,
        event_id: Uuid,
    ) -> Result<Event, AutopilotError> {
        let url = self
            .base_url
            .join(&format!("/v1/sessions/{session_id}/events/{event_id}"))?;
        let response = self
            .http_client
            .get(url)
            .headers(self.auth_headers())
            .send()
            .await?;
        let response = self.check_response(response).await?;
        let event: Event = response.json().await?;
        self.cache_tool_call_event(&event);
        Ok(event)
    }

    /// Creates an event in a session.
    ///
    /// Use `Uuid::nil()` as the `session_id` to create a new session.
    pub async fn create_event(
        &self,
        session_id: Uuid,
        request: CreateEventRequest,
    ) -> Result<CreateEventResponse, AutopilotError> {
        let tool_call_event_id = match &request.payload {
            EventPayload::ToolCallAuthorization(auth) => match auth.status {
                ToolCallAuthorizationStatus::Approved => Some(auth.tool_call_event_id),
                // Don't start the tool if rejected or not available
                ToolCallAuthorizationStatus::Rejected { .. }
                | ToolCallAuthorizationStatus::NotAvailable => None,
            },
            _ => None,
        };
        if session_id == Uuid::nil() && request.config_snapshot_hash.is_none() {
            return Err(AutopilotError::Http {
                status_code: 400,
                message: "Config snapshot hash must be set if a new session is being started"
                    .to_string(),
            });
        }
        if session_id != Uuid::nil() && request.config_snapshot_hash.is_some() {
            return Err(AutopilotError::Http {
                status_code: 400,
                message:
                    "Config snapshot hash must not be set if an existing session is being used"
                        .to_string(),
            });
        }

        let url = self
            .base_url
            .join(&format!("/v1/sessions/{session_id}/events"))?;
        let response = self
            .http_client
            .post(url)
            .headers(self.auth_headers())
            .json(&request)
            .send()
            .await?;
        let response = self.check_response(response).await?;
        let body: CreateEventResponse = response.json().await?;

        if let Some(tool_call_event_id) = tool_call_event_id {
            self.handle_tool_call_authorization(session_id, tool_call_event_id)
                .await?;
        }

        Ok(body)
    }

    /// Approves all pending tool calls for a session up to a specified event ID.
    ///
    /// This atomically approves all pending tool calls with event IDs <= `last_tool_call_event_id`.
    /// The `last_tool_call_event_id` parameter prevents race conditions where new tool calls
    /// arrive after the client fetched the list.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The session is not found (404)
    /// - The caller doesn't have access to the session (403)
    /// - The deployment_id doesn't match the session (400)
    /// - The last_tool_call_event_id is not found in the session (404)
    pub async fn approve_all_tool_calls(
        &self,
        session_id: Uuid,
        request: ApproveAllToolCallsRequest,
    ) -> Result<ApproveAllToolCallsResponse, AutopilotError> {
        let url = self
            .base_url
            .join(&format!("/v1/sessions/{session_id}/actions/approve-all"))?;
        let response = self
            .http_client
            .post(url)
            .headers(self.auth_headers())
            .json(&request)
            .send()
            .await?;
        let response = self.check_response(response).await?;
        let body: ApproveAllToolCallsResponse = response.json().await?;

        // Spawn durable tasks for each approved tool call
        let mut spawn_errors: Vec<(Uuid, Box<AutopilotError>)> = Vec::new();
        for tool_call_event_id in &body.tool_call_event_ids {
            if let Err(e) = self
                .handle_tool_call_authorization(session_id, *tool_call_event_id)
                .await
            {
                tracing::error!(
                    session_id = %session_id,
                    tool_call_event_id = %tool_call_event_id,
                    error = %e,
                    "Failed to spawn task for approved tool call"
                );
                spawn_errors.push((*tool_call_event_id, Box::new(e)));
            }
        }

        if !spawn_errors.is_empty() {
            return Err(AutopilotError::PartialSpawnFailure {
                response: body,
                errors: spawn_errors,
            });
        }

        Ok(body)
    }

    /// Streams events for a session using Server-Sent Events.
    ///
    /// Returns a stream of events. The stream will remain open until:
    /// - The client drops the stream
    /// - The server closes the connection
    /// - An error occurs
    ///
    /// Use `params.last_event_id` to resume from a specific event.
    ///
    /// # Errors
    ///
    /// Returns `AutopilotError::Http` if the server returns an error status code
    /// (e.g., 401 Unauthorized, 404 Not Found). This is checked before returning
    /// the stream, so connection errors are caught immediately.
    /// Returns a stream of `GatewayStreamUpdate` - a narrower type that excludes
    /// `NotAvailable` authorization status.
    pub async fn stream_events(
        &self,
        session_id: Uuid,
        params: StreamEventsParams,
    ) -> Result<
        impl Stream<Item = Result<GatewayStreamUpdate, AutopilotError>> + use<>,
        AutopilotError,
    > {
        let mut url = self
            .base_url
            .join(&format!("/v1/sessions/{session_id}/events/stream"))?;
        if let Some(last_event_id) = params.last_event_id {
            url.query_pairs_mut()
                .append_pair("last_event_id", &last_event_id.to_string());
        }

        // Wait for connection to be established or fail.
        let event_source = self
            .sse_http_client
            .get(url)
            .headers(self.auth_headers())
            .eventsource()
            .await;

        let mut event_source = match event_source {
            Ok(event_source) => event_source,
            Err(e) => return Err(Self::convert_sse_error(e).await),
        };

        // The first event should be Open on success, or an error on failure.
        match event_source.next().await {
            Some(Ok(reqwest_sse_stream::Event::Open)) => {
                // Connection established successfully
            }
            Some(Err(e)) => {
                // Convert SSE error to appropriate AutopilotError
                return Err(Self::convert_sse_error(e).await);
            }
            Some(Ok(reqwest_sse_stream::Event::Message(_))) => {
                return Err(AutopilotError::Sse(
                    "Received message before connection was established".to_string(),
                ));
            }
            None => {
                return Err(AutopilotError::Sse(
                    "Connection closed unexpectedly".to_string(),
                ));
            }
        }

        // Connection is good, return the stream
        let cache = self.tool_call_cache.clone();
        let available_tools = self.available_tools.clone();
        let spawn_client = self.spawn_client.clone();

        let stream = event_source.filter_map(move |result| {
            let cache = cache.clone();
            let available_tools = available_tools.clone();
            let spawn_client = spawn_client.clone();
            async move {
                match result {
                    Ok(reqwest_sse_stream::Event::Open) => None,
                    Ok(reqwest_sse_stream::Event::Message(sse)) => {
                        if sse.event.as_str() == "event" {
                            match serde_json::from_str::<StreamUpdate>(&sse.data) {
                                Ok(update) => {
                                    // Cache tool calls for later lookup
                                    if let EventPayload::ToolCall(tool_call) = &update.event.payload
                                    {
                                        cache.insert(update.event.id, tool_call.clone());

                                        // Reject unknown tool calls
                                        if !available_tools.contains(&tool_call.name)
                                            && let Err(e) =
                                                reject_missing_tool(&spawn_client, tool_call).await
                                        {
                                            return Some(Err(e));
                                        }
                                    }

                                    // Filter out NotAvailable authorizations and unknown tool calls
                                    if Self::should_filter_event(&update.event, &available_tools) {
                                        return None;
                                    }

                                    // Convert to gateway type - should succeed since we filtered NotAvailable above
                                    match update.try_into() {
                                        Ok(gateway_update) => Some(Ok(gateway_update)),
                                        Err(e) => Some(Err(AutopilotError::Internal(format!(
                                            "StreamUpdate conversion failed after filtering: {e}"
                                        )))),
                                    }
                                }
                                Err(e) => Some(Err(AutopilotError::Json(e))),
                            }
                        } else {
                            None
                        }
                    }
                    Err(e) => Some(Err(Self::convert_sse_error(e).await)),
                }
            }
        });

        Ok(stream)
    }

    /// Converts an SSE error to the appropriate AutopilotError.
    /// HTTP errors are converted to AutopilotError::Http for consistency.
    async fn convert_sse_error(e: reqwest_sse_stream::ReqwestSseStreamError) -> AutopilotError {
        match e {
            reqwest_sse_stream::ReqwestSseStreamError::ReqwestError(e) if e.is_status() => {
                AutopilotError::Http {
                    status_code: e.status().map(|s| s.as_u16()).unwrap_or(0),
                    message: e
                        .status()
                        .and_then(|s| s.canonical_reason())
                        .unwrap_or("Unknown error")
                        .to_string(),
                }
            }
            reqwest_sse_stream::ReqwestSseStreamError::InvalidStatusCode(status, response) => {
                AutopilotError::Http {
                    status_code: status.as_u16(),
                    message: response
                        .text()
                        .await
                        .unwrap_or_else(|e| format!("<Could not read response body>: {e}")),
                }
            }
            other => AutopilotError::Sse(other.to_string()),
        }
    }

    // -------------------------------------------------------------------------
    // Helper Methods
    // -------------------------------------------------------------------------

    /// Creates the authorization headers.
    fn auth_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        let auth_value = format!("Bearer {}", self.api_key.expose_secret());
        if let Ok(value) = HeaderValue::from_str(&auth_value) {
            headers.insert(AUTHORIZATION, value);
        }
        headers
    }

    /// Checks the response status and extracts error details if needed.
    async fn check_response(
        &self,
        response: reqwest::Response,
    ) -> Result<reqwest::Response, AutopilotError> {
        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }

        let status_code = status.as_u16();
        let message = match response.json::<ErrorResponse>().await {
            Ok(error_response) => error_response.error.message,
            Err(_) => status
                .canonical_reason()
                .unwrap_or("Unknown error")
                .to_string(),
        };

        Err(AutopilotError::Http {
            status_code,
            message,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_is_unknown_tool_with_known_tool() {
        let mut tools = HashSet::new();
        tools.insert("inference".to_string());
        tools.insert("feedback".to_string());
        let available_tools: Arc<HashSet<String>> = Arc::new(tools);

        let tool_name = "inference".to_string();

        let is_unknown = !available_tools.contains(&tool_name);
        assert!(!is_unknown, "Known tool should not be marked as unknown");
    }

    #[test]
    fn test_is_unknown_tool_with_unknown_tool() {
        let mut tools = HashSet::new();
        tools.insert("inference".to_string());
        tools.insert("feedback".to_string());
        let available_tools: Arc<HashSet<String>> = Arc::new(tools);

        let tool_name = "fake_nonexistent_tool".to_string();

        let is_unknown = !available_tools.contains(&tool_name);
        assert!(is_unknown, "Unknown tool should be marked as unknown");
    }

    #[test]
    fn test_default_base_url_is_valid() {
        // Test that DEFAULT_BASE_URL can be parsed
        let url: Url = DEFAULT_BASE_URL
            .parse()
            .expect("DEFAULT_BASE_URL should be a valid URL");
        assert_eq!(url.scheme(), "https", "Should use HTTPS");
        assert!(url.host_str().is_some(), "Should have a host");
    }

    #[test]
    fn test_build_fails_without_available_tools() {
        let result = futures::executor::block_on(
            AutopilotClient::builder()
                .api_key("test_key")
                .spawn_database_url("postgres://localhost/test")
                .build(),
        );

        assert!(
            matches!(
                result,
                Err(AutopilotError::MissingConfig("available_tools"))
            ),
            "Building without available_tools should fail with MissingConfig error"
        );
    }
}
