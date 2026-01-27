//! TensorZero client trait and implementations.
//!
//! This module provides an abstraction over TensorZero inference and autopilot
//! operations, allowing tools to call these without directly depending on the
//! concrete client type.

mod client_ext;
mod embedded;

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
pub use tensorzero::{
    Client, ClientBuilder, ClientBuilderError, ClientBuilderMode, ClientInferenceParams,
    CreateDatapointRequest, CreateDatapointsFromInferenceRequestParams, CreateDatapointsResponse,
    DeleteDatapointsResponse, FeedbackParams, FeedbackResponse, GetConfigResponse,
    GetDatapointsResponse, InferenceResponse, ListDatapointsRequest, ListDatasetsRequest,
    ListDatasetsResponse, PostgresConfig, TensorZeroError, UpdateDatapointRequest,
    UpdateDatapointsResponse, WriteConfigRequest, WriteConfigResponse,
};
use tensorzero::{GetInferencesRequest, GetInferencesResponse, ListInferencesRequest};
pub use tensorzero_core::cache::CacheEnabledMode;
pub use tensorzero_core::config::snapshot::SnapshotHash;
use tensorzero_core::db::feedback::FeedbackByVariant;
use tensorzero_core::endpoints::feedback::internal::LatestFeedbackIdByMetricResponse;
pub use tensorzero_core::optimization::OptimizationJobHandle;
pub use tensorzero_core::optimization::OptimizationJobInfo;
use tensorzero_optimizers::endpoints::LaunchOptimizationWorkflowParams;
use url::Url;
use uuid::Uuid;

// Re-export client implementations
pub use embedded::EmbeddedClient;

// Re-export autopilot types for use by tools
pub use autopilot_client::{
    CreateEventResponse, EventPayload, EventPayloadToolResult, GatewayListEventsResponse,
    ListEventsParams, ListSessionsParams, ListSessionsResponse, ToolOutcome,
};
pub use tensorzero_core::endpoints::internal::autopilot::CreateEventGatewayRequest;

// Re-export action types from crate::action
pub use crate::action::{ActionInput, ActionInputInfo, ActionResponse};

// Re-export evaluation types from crate::run_evaluation
pub use crate::run_evaluation::{
    DatapointResult, EvaluatorStats, RunEvaluationParams, RunEvaluationResponse,
};

#[cfg(any(test, feature = "test-support"))]
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

    /// Operation not supported in this client mode.
    #[error("Operation not supported: {0}")]
    NotSupported(String),

    /// Evaluation error.
    #[error("Evaluation error: {0}")]
    Evaluation(String),
}

/// Trait for TensorZero client operations, enabling mocking in tests via mockall.
///
/// This trait abstracts over the TensorZero client, allowing tools to
/// call inference and autopilot operations without directly depending on
/// the concrete client type.
#[async_trait]
#[cfg_attr(any(test, feature = "test-support"), automock)]
pub trait TensorZeroClient: Send + Sync + 'static {
    /// Run inference with the given parameters.
    ///
    /// Returns the inference response on success. Streaming inference
    /// is not supported and will return an error.
    async fn inference(
        &self,
        params: ClientInferenceParams,
    ) -> Result<InferenceResponse, TensorZeroClientError>;

    // ========== Feedback Operations ==========

    /// Submit feedback for an inference or episode.
    ///
    /// Feedback can be a comment, demonstration, or a metric value (float or boolean).
    /// The `metric_name` field in `FeedbackParams` determines the feedback type.
    async fn feedback(
        &self,
        params: FeedbackParams,
    ) -> Result<FeedbackResponse, TensorZeroClientError>;

    /// Get the latest feedback ID for each metric for a target.
    async fn get_latest_feedback_id_by_metric(
        &self,
        target_id: Uuid,
    ) -> Result<LatestFeedbackIdByMetricResponse, TensorZeroClientError>;

    /// Get feedback statistics by variant for a function and metric.
    ///
    /// Returns mean, variance, and count for each variant. This is useful for
    /// analyzing variant performance without requiring an HTTP endpoint.
    ///
    /// Note: This method only works in embedded mode (no HTTP endpoint available).
    async fn get_feedback_by_variant(
        &self,
        metric_name: String,
        function_name: String,
        variant_names: Option<Vec<String>>,
    ) -> Result<Vec<FeedbackByVariant>, TensorZeroClientError>;

    /// Create an event in an autopilot session.
    ///
    /// Use `Uuid::nil()` as `session_id` to create a new session.
    /// The deployment_id is injected from the gateway's app state.
    async fn create_autopilot_event(
        &self,
        session_id: Uuid,
        request: CreateEventGatewayRequest,
    ) -> Result<CreateEventResponse, TensorZeroClientError>;

    /// List events in an autopilot session.
    ///
    /// Returns `GatewayListEventsResponse` which uses narrower types that exclude
    /// `NotAvailable` authorization status.
    async fn list_autopilot_events(
        &self,
        session_id: Uuid,
        params: ListEventsParams,
    ) -> Result<GatewayListEventsResponse, TensorZeroClientError>;

    /// List autopilot sessions.
    async fn list_autopilot_sessions(
        &self,
        params: ListSessionsParams,
    ) -> Result<ListSessionsResponse, TensorZeroClientError>;

    /// Execute an action with a historical config snapshot.
    ///
    /// This uses the action endpoint to run inference, feedback, or evaluations
    /// with a specific config version, enabling reproducibility by using the exact
    /// configuration that was active at a previous point in time.
    ///
    /// Returns the appropriate response type based on the action input.
    /// Streaming inference is not supported and will return an error.
    async fn action(
        &self,
        snapshot_hash: SnapshotHash,
        input: ActionInput,
    ) -> Result<ActionResponse, TensorZeroClientError>;

    /// Get a config snapshot by hash, or the live config if no hash is provided.
    async fn get_config_snapshot(
        &self,
        hash: Option<String>,
    ) -> Result<GetConfigResponse, TensorZeroClientError>;

    /// Write a config snapshot to storage.
    async fn write_config(
        &self,
        request: WriteConfigRequest,
    ) -> Result<WriteConfigResponse, TensorZeroClientError>;

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

    /// List all datasets with optional filtering and pagination.
    async fn list_datasets(
        &self,
        request: ListDatasetsRequest,
    ) -> Result<ListDatasetsResponse, TensorZeroClientError>;

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

    // ========== Inference Query Operations ==========

    /// List inferences with filtering and pagination.
    async fn list_inferences(
        &self,
        request: ListInferencesRequest,
    ) -> Result<GetInferencesResponse, TensorZeroClientError>;

    /// Get specific inferences by their IDs.
    async fn get_inferences(
        &self,
        request: GetInferencesRequest,
    ) -> Result<GetInferencesResponse, TensorZeroClientError>;

    // ========== Optimization Operations ==========

    /// Launch an optimization workflow.
    ///
    /// Returns a job handle that can be used to poll the optimization status.
    async fn launch_optimization_workflow(
        &self,
        params: LaunchOptimizationWorkflowParams,
    ) -> Result<OptimizationJobHandle, TensorZeroClientError>;

    /// Poll an optimization workflow for its current status.
    ///
    /// Returns the current status of the optimization job (Pending, Completed, or Failed).
    async fn poll_optimization(
        &self,
        job_handle: &OptimizationJobHandle,
    ) -> Result<OptimizationJobInfo, TensorZeroClientError>;

    // ========== Evaluation Operations ==========

    /// Run an evaluation on a dataset or set of datapoints.
    ///
    /// This runs inference on each datapoint using the specified variant,
    /// then runs the configured evaluators on the results.
    ///
    /// Returns summary statistics for each evaluator.
    ///
    /// Note: This operation is only supported in embedded gateway mode.
    /// HTTP gateway mode will return a `NotSupported` error.
    async fn run_evaluation(
        &self,
        params: RunEvaluationParams,
    ) -> Result<RunEvaluationResponse, TensorZeroClientError>;
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
///     None,
/// ).await?;
/// ```
pub async fn embedded_gateway_client(
    config_file: Option<PathBuf>,
    clickhouse_url: Option<String>,
    postgres_config: Option<String>,
    valkey_url: Option<String>,
) -> Result<Arc<dyn TensorZeroClient>, ClientBuilderError> {
    let client = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file,
        clickhouse_url,
        postgres_config: postgres_config.map(PostgresConfig::Url),
        valkey_url,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: false,
    })
    .build()
    .await?;
    Ok(Arc::new(client))
}
