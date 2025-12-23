//! Inference client trait and implementations.
//!
//! This module provides an abstraction over TensorZero inference and autopilot
//! operations, allowing tools to call these without directly depending on the
//! concrete client type.

use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::Arc;
use tensorzero::{
    Client, ClientBuilder, ClientBuilderError, ClientBuilderMode, ClientInferenceParams,
    ClientMode, InferenceOutput, InferenceResponse, TensorZeroError,
};
use url::Url;
use uuid::Uuid;

// Re-export autopilot types for use by tools
pub use autopilot_client::{
    CreateEventRequest, CreateEventResponse, EventPayload, ListEventsParams, ListEventsResponse,
    ListSessionsParams, ListSessionsResponse, ToolOutcome,
};

#[cfg(test)]
use mockall::automock;

/// Error type for inference and autopilot operations.
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    /// Error from the TensorZero client.
    #[error(transparent)]
    TensorZero(#[from] TensorZeroError),

    /// Streaming inference was returned but is not supported.
    #[error("Streaming inference not supported in tool context")]
    StreamingNotSupported,

    /// Autopilot client is not configured.
    #[error("Autopilot client not configured")]
    AutopilotUnavailable,

    /// Error from the Autopilot API.
    #[error("Autopilot error: {0}")]
    Autopilot(#[from] autopilot_client::AutopilotError),
}

/// Trait for inference and autopilot clients, enabling mocking in tests via mockall.
///
/// This trait abstracts over the TensorZero client, allowing tools to
/// call inference and autopilot operations without directly depending on
/// the concrete client type.
#[async_trait]
#[cfg_attr(test, automock)]
pub trait InferenceClient: Send + Sync + 'static {
    /// Run inference with the given parameters.
    ///
    /// Returns the inference response on success. Streaming inference
    /// is not supported and will return an error.
    async fn inference(
        &self,
        params: ClientInferenceParams,
    ) -> Result<InferenceResponse, InferenceError>;

    /// Create an event in an autopilot session.
    ///
    /// Use `Uuid::nil()` as `session_id` to create a new session.
    async fn create_autopilot_event(
        &self,
        session_id: Uuid,
        request: CreateEventRequest,
    ) -> Result<CreateEventResponse, InferenceError>;

    /// List events in an autopilot session.
    async fn list_autopilot_events(
        &self,
        session_id: Uuid,
        params: ListEventsParams,
    ) -> Result<ListEventsResponse, InferenceError>;

    /// List autopilot sessions.
    async fn list_autopilot_sessions(
        &self,
        params: ListSessionsParams,
    ) -> Result<ListSessionsResponse, InferenceError>;
}

/// Implementation of `InferenceClient` for the real TensorZero client.
#[async_trait]
impl InferenceClient for Client {
    async fn inference(
        &self,
        params: ClientInferenceParams,
    ) -> Result<InferenceResponse, InferenceError> {
        match Client::inference(self, params).await? {
            InferenceOutput::NonStreaming(response) => Ok(response),
            InferenceOutput::Streaming(_) => Err(InferenceError::StreamingNotSupported),
        }
    }

    async fn create_autopilot_event(
        &self,
        session_id: Uuid,
        request: CreateEventRequest,
    ) -> Result<CreateEventResponse, InferenceError> {
        match self.mode() {
            ClientMode::HTTPGateway(http) => {
                // HTTP mode: call the internal endpoint
                let url = http
                    .base_url
                    .join(&format!(
                        "internal/autopilot/v1/sessions/{session_id}/events"
                    ))
                    .map_err(|e: url::ParseError| {
                        InferenceError::Autopilot(autopilot_client::AutopilotError::InvalidUrl(e))
                    })?;

                let response = http
                    .http_client
                    .post(url)
                    .json(&request)
                    .send()
                    .await
                    .map_err(|e| {
                        InferenceError::Autopilot(autopilot_client::AutopilotError::Request(e))
                    })?;

                if !response.status().is_success() {
                    let status = response.status().as_u16();
                    let text = response.text().await.unwrap_or_default();
                    return Err(InferenceError::Autopilot(
                        autopilot_client::AutopilotError::Http {
                            status_code: status,
                            message: text,
                        },
                    ));
                }

                response.json().await.map_err(|e| {
                    InferenceError::Autopilot(autopilot_client::AutopilotError::Request(e))
                })
            }
            ClientMode::EmbeddedGateway {
                gateway,
                timeout: _,
            } => {
                // Embedded mode: call the core function directly
                let autopilot_client = gateway
                    .handle
                    .app_state
                    .autopilot_client
                    .as_ref()
                    .ok_or(InferenceError::AutopilotUnavailable)?;

                tensorzero_core::endpoints::internal::autopilot::create_event(
                    autopilot_client,
                    session_id,
                    request,
                )
                .await
                .map_err(|e| {
                    InferenceError::TensorZero(TensorZeroError::Other { source: e.into() })
                })
            }
        }
    }

    async fn list_autopilot_events(
        &self,
        session_id: Uuid,
        params: ListEventsParams,
    ) -> Result<ListEventsResponse, InferenceError> {
        match self.mode() {
            ClientMode::HTTPGateway(http) => {
                let mut url = http
                    .base_url
                    .join(&format!(
                        "internal/autopilot/v1/sessions/{session_id}/events"
                    ))
                    .map_err(|e: url::ParseError| {
                        InferenceError::Autopilot(autopilot_client::AutopilotError::InvalidUrl(e))
                    })?;

                // Add query params
                if let Some(limit) = params.limit {
                    url.query_pairs_mut()
                        .append_pair("limit", &limit.to_string());
                }
                if let Some(before) = params.before {
                    url.query_pairs_mut()
                        .append_pair("before", &before.to_string());
                }

                let response = http.http_client.get(url).send().await.map_err(|e| {
                    InferenceError::Autopilot(autopilot_client::AutopilotError::Request(e))
                })?;

                if !response.status().is_success() {
                    let status = response.status().as_u16();
                    let text = response.text().await.unwrap_or_default();
                    return Err(InferenceError::Autopilot(
                        autopilot_client::AutopilotError::Http {
                            status_code: status,
                            message: text,
                        },
                    ));
                }

                response.json().await.map_err(|e| {
                    InferenceError::Autopilot(autopilot_client::AutopilotError::Request(e))
                })
            }
            ClientMode::EmbeddedGateway {
                gateway,
                timeout: _,
            } => {
                let autopilot_client = gateway
                    .handle
                    .app_state
                    .autopilot_client
                    .as_ref()
                    .ok_or(InferenceError::AutopilotUnavailable)?;

                tensorzero_core::endpoints::internal::autopilot::list_events(
                    autopilot_client,
                    session_id,
                    params,
                )
                .await
                .map_err(|e| {
                    InferenceError::TensorZero(TensorZeroError::Other { source: e.into() })
                })
            }
        }
    }

    async fn list_autopilot_sessions(
        &self,
        params: ListSessionsParams,
    ) -> Result<ListSessionsResponse, InferenceError> {
        match self.mode() {
            ClientMode::HTTPGateway(http) => {
                let mut url = http
                    .base_url
                    .join("internal/autopilot/v1/sessions")
                    .map_err(|e: url::ParseError| {
                        InferenceError::Autopilot(autopilot_client::AutopilotError::InvalidUrl(e))
                    })?;

                // Add query params
                if let Some(limit) = params.limit {
                    url.query_pairs_mut()
                        .append_pair("limit", &limit.to_string());
                }
                if let Some(offset) = params.offset {
                    url.query_pairs_mut()
                        .append_pair("offset", &offset.to_string());
                }

                let response = http.http_client.get(url).send().await.map_err(|e| {
                    InferenceError::Autopilot(autopilot_client::AutopilotError::Request(e))
                })?;

                if !response.status().is_success() {
                    let status = response.status().as_u16();
                    let text = response.text().await.unwrap_or_default();
                    return Err(InferenceError::Autopilot(
                        autopilot_client::AutopilotError::Http {
                            status_code: status,
                            message: text,
                        },
                    ));
                }

                response.json().await.map_err(|e| {
                    InferenceError::Autopilot(autopilot_client::AutopilotError::Request(e))
                })
            }
            ClientMode::EmbeddedGateway {
                gateway,
                timeout: _,
            } => {
                let autopilot_client = gateway
                    .handle
                    .app_state
                    .autopilot_client
                    .as_ref()
                    .ok_or(InferenceError::AutopilotUnavailable)?;

                tensorzero_core::endpoints::internal::autopilot::list_sessions(
                    autopilot_client,
                    params,
                )
                .await
                .map_err(|e| {
                    InferenceError::TensorZero(TensorZeroError::Other { source: e.into() })
                })
            }
        }
    }
}

/// Create an inference client from an existing TensorZero `Client`.
///
/// This wraps the client in an `Arc<dyn InferenceClient>` for use with
/// `ToolExecutorBuilder`.
///
/// # Example
///
/// ```ignore
/// use tensorzero::{ClientBuilder, ClientBuilderMode};
/// use durable_tools::inference::from_client;
///
/// let client = ClientBuilder::new(ClientBuilderMode::HTTPGateway {
///     url: "http://localhost:3000".parse()?,
/// }).build_http()?;
/// let inference_client = from_client(client);
/// ```
pub fn from_client(client: Client) -> Arc<dyn InferenceClient> {
    Arc::new(client)
}

/// Create an inference client for HTTP gateway mode.
///
/// This connects to a TensorZero gateway server at the specified URL.
///
/// # Errors
///
/// Returns a `ClientBuilderError` if the HTTP client cannot be built.
///
/// # Example
///
/// ```ignore
/// use durable_tools::inference::http_gateway_client;
/// use url::Url;
///
/// let client = http_gateway_client(Url::parse("http://localhost:3000")?)?;
/// ```
pub fn http_gateway_client(url: Url) -> Result<Arc<dyn InferenceClient>, ClientBuilderError> {
    let client = ClientBuilder::new(ClientBuilderMode::HTTPGateway { url }).build_http()?;
    Ok(Arc::new(client))
}

/// Create an inference client for embedded gateway mode.
///
/// This runs the TensorZero gateway embedded in the application,
/// without requiring an external gateway server.
///
/// # Arguments
///
/// * `config_file` - Path to the TensorZero config file (tensorzero.toml)
/// * `clickhouse_url` - Optional ClickHouse URL for observability
/// * `postgres_url` - Optional Postgres URL for experimentation
///
/// # Errors
///
/// Returns a `ClientBuilderError` if the embedded gateway client cannot be built
/// (e.g., invalid config file, connection failures, or invalid credentials).
///
/// # Example
///
/// ```ignore
/// use durable_tools::inference::embedded_gateway_client;
///
/// let client = embedded_gateway_client(
///     Some("tensorzero.toml".into()),
///     Some("http://localhost:8123".into()),
///     None,
/// ).await?;
/// ```
pub async fn embedded_gateway_client(
    config_file: Option<PathBuf>,
    clickhouse_url: Option<String>,
    postgres_url: Option<String>,
) -> Result<Arc<dyn InferenceClient>, ClientBuilderError> {
    let client = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file,
        clickhouse_url,
        postgres_url,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: false,
    })
    .build()
    .await?;
    Ok(Arc::new(client))
}
