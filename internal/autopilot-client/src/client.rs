//! Autopilot API client implementation.

use std::fmt;
use std::time::Duration;

use durable_tools_spawn::SpawnClient;
use futures::stream::{Stream, StreamExt};
use moka::sync::Cache;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use reqwest_eventsource::{Event as SseEvent, EventSource};
use secrecy::{ExposeSecret, SecretString};
use serde_json::Value as JsonValue;
use sqlx::PgPool;
use tracing::warn;
use url::Url;
use uuid::Uuid;

use crate::error::AutopilotError;
use crate::types::{
    CreateEventRequest, CreateEventResponse, ErrorResponse, Event, EventPayload, ListEventsParams,
    ListEventsResponse, ListSessionsParams, ListSessionsResponse, StreamEventsParams, ToolCall,
    ToolCallAuthorizationStatus,
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

    /// Builds the [`AutopilotClient`].
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The API key is not set
    /// - The HTTP client cannot be built
    /// - Durable spawn configuration is missing or invalid
    pub async fn build(self) -> Result<AutopilotClient, AutopilotError> {
        let api_key = self
            .api_key
            .ok_or(AutopilotError::MissingConfig("api_key"))?;

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

        let spawn_client = spawn_builder.build().await?;
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
    spawn_client: SpawnClient,
    tool_call_cache: Cache<Uuid, ToolCall>,
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

    async fn handle_tool_call_approval(
        &self,
        session_id: Uuid,
        deployment_id: Uuid,
        tool_call_event_id: Uuid,
    ) {
        let mut tool_call = self.tool_call_cache.get(&tool_call_event_id);

        if tool_call.is_none() {
            match self
                .list_events(session_id, ListEventsParams::default())
                .await
            {
                Ok(response) => {
                    tool_call = response.pending_tool_calls.iter().find_map(|event| {
                        if event.id == tool_call_event_id {
                            match &event.payload {
                                EventPayload::ToolCall(tool_call) => {
                                    self.tool_call_cache.insert(event.id, tool_call.clone());
                                    Some(tool_call.clone())
                                }
                                _ => None,
                            }
                        } else {
                            None
                        }
                    });
                }
                Err(err) => {
                    warn!(
                        tool_call_event_id = %tool_call_event_id,
                        "Failed to fetch tool call approval details: {err}"
                    );
                    return;
                }
            }
        }

        let tool_call = match tool_call {
            Some(tool_call) => tool_call,
            None => {
                warn!(
                    tool_call_event_id = %tool_call_event_id,
                    "Tool call approval received without matching pending tool call"
                );
                return;
            }
        };

        let tool_call_id = tool_call.id.clone();
        let tool_name = tool_call.name.clone();
        let tool_arguments = tool_call.arguments.clone();

        let llm_params = match serde_json::from_str::<JsonValue>(&tool_arguments) {
            Ok(value) => value,
            Err(err) => {
                warn!(
                    tool_call_event_id = %tool_call_event_id,
                    tool_name = %tool_name,
                    "Failed to parse tool call arguments as JSON: {err}"
                );
                JsonValue::String(tool_arguments.clone())
            }
        };

        let side_info = serde_json::json!({
            "tool_call_event_id": tool_call_event_id,
            "tool_call_id": tool_call_id,
            "session_id": session_id,
            "deployment_id": deployment_id,
            "inner": null,
        });

        let episode_id = Uuid::now_v7();
        if let Err(err) = self
            .spawn_client
            .spawn_tool_by_name(&tool_name, llm_params, side_info, episode_id)
            .await
        {
            warn!(
                tool_call_event_id = %tool_call_event_id,
                tool_name = %tool_name,
                "Failed to spawn tool call: {err}"
            );
        }
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

    // -------------------------------------------------------------------------
    // Event Endpoints
    // -------------------------------------------------------------------------

    /// Lists events for a session.
    pub async fn list_events(
        &self,
        session_id: Uuid,
        params: ListEventsParams,
    ) -> Result<ListEventsResponse, AutopilotError> {
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
        Ok(body)
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
                // Don't start the tool if rejected
                ToolCallAuthorizationStatus::Rejected { .. } => None,
            },
            _ => None,
        };
        let deployment_id = request.deployment_id;

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
            self.handle_tool_call_approval(session_id, deployment_id, tool_call_event_id)
                .await;
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
    pub async fn stream_events(
        &self,
        session_id: Uuid,
        params: StreamEventsParams,
    ) -> Result<impl Stream<Item = Result<Event, AutopilotError>> + use<>, AutopilotError> {
        let mut url = self
            .base_url
            .join(&format!("/v1/sessions/{session_id}/events/stream"))?;
        if let Some(last_event_id) = params.last_event_id {
            url.query_pairs_mut()
                .append_pair("last_event_id", &last_event_id.to_string());
        }

        let request = self.sse_http_client.get(url).headers(self.auth_headers());

        let mut event_source =
            EventSource::new(request).map_err(|e| AutopilotError::Sse(e.to_string()))?;

        // Wait for connection to be established or fail.
        // The first event should be Open on success, or an error on failure.
        match event_source.next().await {
            Some(Ok(SseEvent::Open)) => {
                // Connection established successfully
            }
            Some(Err(e)) => {
                // Convert SSE error to appropriate AutopilotError
                return Err(Self::convert_sse_error(e));
            }
            Some(Ok(SseEvent::Message(_))) => {
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
        let stream = event_source.filter_map(move |result| {
            let cache = cache.clone();
            async move {
                match result {
                    Ok(SseEvent::Open) => None,
                    Ok(SseEvent::Message(message)) => {
                        if message.event == "event" {
                            match serde_json::from_str::<Event>(&message.data) {
                                Ok(event) => {
                                    if let EventPayload::ToolCall(tool_call) = &event.payload {
                                        cache.insert(event.id, tool_call.clone());
                                    }
                                    Some(Ok(event))
                                }
                                Err(e) => Some(Err(AutopilotError::Json(e))),
                            }
                        } else {
                            None
                        }
                    }
                    Err(e) => Some(Err(AutopilotError::Sse(e.to_string()))),
                }
            }
        });

        Ok(stream)
    }

    /// Converts an SSE error to the appropriate AutopilotError.
    /// HTTP errors are converted to AutopilotError::Http for consistency.
    fn convert_sse_error(e: reqwest_eventsource::Error) -> AutopilotError {
        use reqwest_eventsource::Error as SseError;
        match e {
            SseError::InvalidStatusCode(status, _response) => AutopilotError::Http {
                status_code: status.as_u16(),
                message: status
                    .canonical_reason()
                    .unwrap_or("Unknown error")
                    .to_string(),
            },
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
