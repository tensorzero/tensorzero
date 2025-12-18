//! Autopilot API client implementation.

use std::time::Duration;

use futures::stream::{Stream, StreamExt};
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use reqwest_eventsource::{Event as SseEvent, EventSource};
use secrecy::{ExposeSecret, SecretString};
use url::Url;
use uuid::Uuid;

use crate::error::AutopilotError;
use crate::types::{
    CreateEventRequest, CreateEventResponse, ErrorResponse, Event, ListEventsParams,
    ListEventsResponse, ListSessionsParams, ListSessionsResponse, StreamEventsParams,
};

/// Default base URL for the Autopilot API.
pub const DEFAULT_BASE_URL: &str = "https://api.autopilot.tensorzero.com";

/// Returns the default base URL as a parsed [`Url`].
///
/// This function is infallible because [`DEFAULT_BASE_URL`] is a valid URL.
fn default_base_url() -> Url {
    // SAFETY: DEFAULT_BASE_URL is a compile-time constant that is a valid URL.
    // This is tested in unit tests.
    #[expect(clippy::expect_used)]
    Url::parse(DEFAULT_BASE_URL).expect("DEFAULT_BASE_URL is a valid URL")
}

// =============================================================================
// Client Builder
// =============================================================================

/// Builder for creating an [`AutopilotClient`].
#[derive(Default)]
pub struct AutopilotClientBuilder {
    base_url: Option<Url>,
    api_key: Option<SecretString>,
    http_client: Option<reqwest::Client>,
    timeout: Option<Duration>,
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

    /// Builds the [`AutopilotClient`].
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The API key is not set
    /// - The HTTP client cannot be built
    pub fn build(self) -> Result<AutopilotClient, AutopilotError> {
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

        Ok(AutopilotClient {
            http_client,
            sse_http_client,
            base_url,
            api_key,
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
        let body = response.json().await?;
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
        let body = response.json().await?;
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
    ) -> Result<impl Stream<Item = Result<Event, AutopilotError>>, AutopilotError> {
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
        let stream = event_source.filter_map(|result| async move {
            match result {
                Ok(SseEvent::Open) => None,
                Ok(SseEvent::Message(message)) => {
                    if message.event == "event" {
                        match serde_json::from_str::<Event>(&message.data) {
                            Ok(event) => Some(Ok(event)),
                            Err(e) => Some(Err(AutopilotError::Json(e))),
                        }
                    } else {
                        None
                    }
                }
                Err(e) => Some(Err(AutopilotError::Sse(e.to_string()))),
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
