//! TensorZero client trait and implementations.
//!
//! This module provides an abstraction over TensorZero inference and autopilot
//! operations, allowing tools to call these without directly depending on the
//! concrete client type.

mod client_ext;
mod embedded;

use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::Arc;
pub use tensorzero::{
    ActionInput, Client, ClientBuilder, ClientBuilderError, ClientBuilderMode,
    ClientInferenceParams, CreateDatapointRequest, CreateDatapointsFromInferenceRequestParams,
    CreateDatapointsResponse, DeleteDatapointsResponse, GetDatapointsResponse, InferenceResponse,
    ListDatapointsRequest, TensorZeroError, UpdateDatapointRequest, UpdateDatapointsResponse,
};
pub use tensorzero_core::config::snapshot::SnapshotHash;
use tensorzero_optimizers::endpoints::LaunchOptimizationWorkflowParams;
use url::Url;
use uuid::Uuid;

// Re-export client implementations
pub use embedded::EmbeddedClient;

// Re-export autopilot types for use by tools
pub use autopilot_client::{
    CreateEventRequest, CreateEventResponse, EventPayload, ListEventsParams, ListEventsResponse,
    ListSessionsParams, ListSessionsResponse, ToolOutcome,
};

#[cfg(test)]
use mockall::automock;

/// Error type for TensorZero client operations.
#[derive(Debug, thiserror::Error)]
pub enum TensorZeroClientError {
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

/// Trait for TensorZero client operations, enabling mocking in tests via mockall.
///
/// This trait abstracts over the TensorZero client, allowing tools to
/// call inference and autopilot operations without directly depending on
/// the concrete client type.
#[async_trait]
#[cfg_attr(test, automock)]
pub trait TensorZeroClient: Send + Sync + 'static {
    /// Run inference with the given parameters.
    ///
    /// Returns the inference response on success. Streaming inference
    /// is not supported and will return an error.
    async fn inference(
        &self,
        params: ClientInferenceParams,
    ) -> Result<InferenceResponse, TensorZeroClientError>;

    /// Create an event in an autopilot session.
    ///
    /// Use `Uuid::nil()` as `session_id` to create a new session.
    async fn create_autopilot_event(
        &self,
        session_id: Uuid,
        request: CreateEventRequest,
    ) -> Result<CreateEventResponse, TensorZeroClientError>;

    /// List events in an autopilot session.
    async fn list_autopilot_events(
        &self,
        session_id: Uuid,
        params: ListEventsParams,
    ) -> Result<ListEventsResponse, TensorZeroClientError>;

    /// List autopilot sessions.
    async fn list_autopilot_sessions(
        &self,
        params: ListSessionsParams,
    ) -> Result<ListSessionsResponse, TensorZeroClientError>;

    /// Run inference with a historical config snapshot.
    ///
    /// This uses the action endpoint to run inference with a specific config version,
    /// enabling reproducibility by using the exact configuration that was active
    /// at a previous point in time.
    ///
    /// Returns the inference response on success. Streaming inference
    /// is not supported and will return an error.
    async fn action(
        &self,
        snapshot_hash: SnapshotHash,
        input: ActionInput,
    ) -> Result<InferenceResponse, TensorZeroClientError>;

    // ========== Datapoint CRUD Operations ==========

    /// Create datapoints in a dataset.
    async fn create_datapoints(
        &self,
        dataset_name: String,
        datapoints: Vec<CreateDatapointRequest>,
    ) -> Result<CreateDatapointsResponse, TensorZeroClientError>;

    /// Create datapoints from existing inferences.
    async fn create_datapoints_from_inferences(
        &self,
        dataset_name: String,
        params: CreateDatapointsFromInferenceRequestParams,
    ) -> Result<CreateDatapointsResponse, TensorZeroClientError>;

    /// List datapoints in a dataset with filtering and pagination.
    async fn list_datapoints(
        &self,
        dataset_name: String,
        request: ListDatapointsRequest,
    ) -> Result<GetDatapointsResponse, TensorZeroClientError>;

    /// Get specific datapoints by their IDs.
    async fn get_datapoints(
        &self,
        dataset_name: Option<String>,
        ids: Vec<Uuid>,
    ) -> Result<GetDatapointsResponse, TensorZeroClientError>;

    /// Update existing datapoints.
    async fn update_datapoints(
        &self,
        dataset_name: String,
        datapoints: Vec<UpdateDatapointRequest>,
    ) -> Result<UpdateDatapointsResponse, TensorZeroClientError>;

    /// Delete datapoints by ID.
    async fn delete_datapoints(
        &self,
        dataset_name: String,
        ids: Vec<Uuid>,
    ) -> Result<DeleteDatapointsResponse, TensorZeroClientError>;

    // ========== Optimization Operations ==========

    /// Launch an optimization workflow.
    ///
    /// Returns an encoded job handle that can be used to poll the optimization status.
    async fn launch_optimization_workflow(
        &self,
        params: LaunchOptimizationWorkflowParams,
    ) -> Result<String, TensorZeroClientError>;
}

/// Create a TensorZero client from an existing TensorZero `Client`.
///
/// This wraps the client in an `Arc<dyn TensorZeroClient>` for use with
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
/// let tz_client = from_client(client);
/// ```
pub fn from_client(client: Client) -> Arc<dyn TensorZeroClient> {
    Arc::new(client)
}

/// Create a TensorZero client for HTTP gateway mode.
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
pub fn http_gateway_client(url: Url) -> Result<Arc<dyn TensorZeroClient>, ClientBuilderError> {
    let client = ClientBuilder::new(ClientBuilderMode::HTTPGateway { url }).build_http()?;
    Ok(Arc::new(client))
}

/// Create a TensorZero client for embedded gateway mode.
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
) -> Result<Arc<dyn TensorZeroClient>, ClientBuilderError> {
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
