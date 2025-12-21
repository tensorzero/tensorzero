//! Inference client trait and implementations.
//!
//! This module provides an abstraction over TensorZero inference,
//! allowing tools to call inference and enabling mocking in tests.

use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::Arc;
use tensorzero::{
    Client, ClientBuilder, ClientBuilderError, ClientBuilderMode, ClientInferenceParams,
    InferenceOutput, InferenceResponse, TensorZeroError,
};
use url::Url;

#[cfg(test)]
use mockall::automock;

/// Error type for inference operations.
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    /// Error from the TensorZero client.
    #[error(transparent)]
    TensorZero(#[from] TensorZeroError),

    /// Streaming inference was returned but is not supported.
    #[error("Streaming inference not supported in tool context")]
    StreamingNotSupported,
}

/// Trait for inference clients, enabling mocking in tests via mockall.
///
/// This trait abstracts over the TensorZero client, allowing tools to
/// call inference without directly depending on the concrete client type.
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
