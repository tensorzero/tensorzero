//! TensorZero client trait and implementations.
//!
//! This module provides an abstraction over TensorZero inference and autopilot
//! operations, allowing tools to call these without directly depending on the
//! concrete client type.

mod client_ext;
mod embedded;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
pub use tensorzero::{
    ActionInput, Client, ClientBuilder, ClientBuilderError, ClientBuilderMode,
    ClientInferenceParams, CreateDatapointRequest, CreateDatapointsFromInferenceRequestParams,
    CreateDatapointsResponse, DeleteDatapointsResponse, FeedbackParams, FeedbackResponse,
    GetConfigResponse, GetDatapointsResponse, InferenceResponse, ListDatapointsRequest,
    PostgresConfig, TensorZeroError, UpdateDatapointRequest, UpdateDatapointsResponse,
    WriteConfigRequest, WriteConfigResponse,
};
use tensorzero::{GetInferencesResponse, ListInferencesRequest};
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
    CreateEventResponse, EventPayload, ListEventsParams, ListEventsResponse, ListSessionsParams,
    ListSessionsResponse, ToolOutcome,
};
pub use tensorzero_core::endpoints::internal::autopilot::CreateEventGatewayRequest;

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

    /// Operation not supported in this client mode.
    #[error("Operation not supported: {0}")]
    NotSupported(String),

    /// Evaluation error.
    #[error("Evaluation error: {0}")]
    Evaluation(String),
}

// Note: These evaluation types are specific to durable-tools and cannot be replaced with
// the HTTP wire types from gateway/src/routes/evaluations.rs. The HTTP endpoint uses SSE
// streaming with per-datapoint events, while these types provide an aggregated response
// suitable for tool use cases. Additionally, RunEvaluationParams takes only evaluation_name
// (looking up config internally), while the HTTP endpoint requires the caller to pass in
// the resolved evaluation_config and function_config.

/// Parameters for running an evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunEvaluationParams {
    /// Name of the evaluation to run.
    pub evaluation_name: String,
    /// Name of the dataset to run on.
    /// Either dataset_name or datapoint_ids must be provided, but not both.
    pub dataset_name: Option<String>,
    /// Specific datapoint IDs to evaluate.
    /// Either dataset_name or datapoint_ids must be provided, but not both.
    pub datapoint_ids: Option<Vec<Uuid>>,
    /// Name of the variant to evaluate.
    pub variant_name: String,
    /// Number of concurrent requests to make.
    pub concurrency: usize,
    /// Cache configuration for inference requests.
    pub inference_cache: CacheEnabledMode,
    /// Maximum number of datapoints to evaluate from the dataset.
    pub max_datapoints: Option<u32>,
    /// Precision targets for adaptive stopping.
    /// Maps evaluator names to target confidence interval half-widths.
    /// When the CI half-width for an evaluator falls below its target,
    /// evaluation may stop early for that evaluator.
    #[serde(default)]
    pub precision_targets: HashMap<String, f32>,
}

/// Statistics for a single evaluator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluatorStatsResponse {
    /// Mean value of the evaluator.
    pub mean: f32,
    /// Standard error of the evaluator.
    pub stderr: f32,
    /// Number of samples.
    pub count: usize,
}

/// Response from running an evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunEvaluationResponse {
    /// Unique identifier for this evaluation run.
    pub evaluation_run_id: Uuid,
    /// Number of datapoints evaluated.
    pub num_datapoints: usize,
    /// Number of successful evaluations.
    pub num_successes: usize,
    /// Number of errors.
    pub num_errors: usize,
    /// Per-evaluator statistics.
    pub stats: HashMap<String, EvaluatorStatsResponse>,
}

// Re-export topk types from evaluations crate
pub use evaluations::betting_confidence_sequences::{
    MeanBettingConfidenceSequence, WealthProcessGridPoints, WealthProcesses,
};
pub use evaluations::topk::{
    GlobalStoppingReason, ScoringFunctionType, TopKTaskOutput, TopKTaskParams, VariantStatus,
};

/// Parameters for running a top-k evaluation.
///
/// This wraps [`TopKTaskParams`] but uses names for evaluation and function configs,
/// which are resolved at runtime from the gateway config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunTopKEvaluationParams {
    /// Name of the evaluation to run.
    pub evaluation_name: String,
    /// Name of the dataset to run on.
    pub dataset_name: String,
    /// List of variant names to compare.
    pub variant_names: Vec<String>,
    /// Minimum k for top-k identification.
    pub k_min: u32,
    /// Maximum k for top-k identification.
    pub k_max: u32,
    /// Tolerance for performance equivalence (epsilon).
    #[serde(default)]
    pub epsilon: Option<f64>,
    /// Maximum number of datapoints to process.
    #[serde(default)]
    pub max_datapoints: Option<usize>,
    /// Batch size for processing.
    #[serde(default)]
    pub batch_size: Option<usize>,
    /// Failure rate threshold for variants.
    /// Variants exceeding this threshold are marked as Failed.
    #[serde(default = "default_failure_threshold")]
    pub variant_failure_threshold: f64,
    /// Failure rate threshold for evaluators.
    /// The run terminates if any evaluator exceeds this threshold.
    #[serde(default = "default_failure_threshold")]
    pub evaluator_failure_threshold: f64,
    /// Number of concurrent requests.
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    /// Cache mode for inference.
    #[serde(default)]
    pub inference_cache: CacheEnabledMode,
    /// Scoring function type for ranking variants.
    pub scoring_function: ScoringFunctionType,
}

fn default_failure_threshold() -> f64 {
    0.05
}

fn default_concurrency() -> usize {
    5
}

/// Response from a top-k evaluation.
///
/// This is a wrapper around [`TopKTaskOutput`] for API consistency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunTopKEvaluationResponse {
    /// The full output from the top-k evaluation task.
    #[serde(flatten)]
    pub output: TopKTaskOutput,
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

    /// Run a top-k evaluation to identify the best-performing variants.
    ///
    /// This runs an adaptive evaluation algorithm that evaluates multiple variants
    /// against a dataset, stopping when it can confidently identify the top-k variants
    /// (for some k in [k_min, k_max]).
    ///
    /// The evaluation uses betting confidence sequences for anytime-valid inference,
    /// allowing early stopping when sufficient confidence is reached.
    ///
    /// Note: This operation is only supported in embedded gateway mode.
    /// HTTP gateway mode will return a `NotSupported` error.
    async fn run_topk_evaluation(
        &self,
        params: RunTopKEvaluationParams,
    ) -> Result<RunTopKEvaluationResponse, TensorZeroClientError>;
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
    postgres_config: Option<String>,
) -> Result<Arc<dyn TensorZeroClient>, ClientBuilderError> {
    let client = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file,
        clickhouse_url,
        postgres_config: postgres_config.map(PostgresConfig::Url),
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: false,
    })
    .build()
    .await?;
    Ok(Arc::new(client))
}

/// Poll for top-k task completion.
///
/// This is a shared helper used by both embedded and HTTP client implementations.
pub(super) async fn poll_topk_task(
    pool: &sqlx::PgPool,
    queue_name: &str,
    task_id: Uuid,
) -> Result<evaluations::topk::TopKTaskOutput, TensorZeroClientError> {
    use sqlx::{AssertSqlSafe, query_as};
    use std::time::Duration;

    let timeout = Duration::from_secs(3600); // 1 hour timeout
    let start = std::time::Instant::now();

    loop {
        if start.elapsed() > timeout {
            return Err(TensorZeroClientError::Evaluation(
                "Top-k evaluation timed out".to_string(),
            ));
        }

        // Check task state
        let query = format!("SELECT state FROM durable.t_{queue_name} WHERE task_id = $1");
        let state: Option<(String,)> = query_as(AssertSqlSafe(query))
            .bind(task_id)
            .fetch_optional(pool)
            .await
            .map_err(|e| {
                TensorZeroClientError::Evaluation(format!("Failed to query task state: {e}"))
            })?;

        if let Some((state,)) = state {
            if state == "completed" {
                break;
            } else if state == "failed" {
                // Get error message
                let query =
                    format!("SELECT failed_error FROM durable.t_{queue_name} WHERE task_id = $1");
                let error: Option<(Option<String>,)> = query_as(AssertSqlSafe(query))
                    .bind(task_id)
                    .fetch_optional(pool)
                    .await
                    .ok()
                    .flatten();
                let error_msg = error
                    .and_then(|(e,)| e)
                    .unwrap_or_else(|| "Unknown error".to_string());
                return Err(TensorZeroClientError::Evaluation(format!(
                    "Top-k task failed: {error_msg}"
                )));
            }
        }

        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    // Get the task result
    let query = format!("SELECT completed_payload FROM durable.t_{queue_name} WHERE task_id = $1");
    let result: Option<(Option<serde_json::Value>,)> = query_as(AssertSqlSafe(query))
        .bind(task_id)
        .fetch_optional(pool)
        .await
        .map_err(|e| {
            TensorZeroClientError::Evaluation(format!("Failed to query task result: {e}"))
        })?;

    let output = result
        .and_then(|(payload,)| payload)
        .ok_or_else(|| TensorZeroClientError::Evaluation("No task output found".to_string()))?;

    serde_json::from_value(output).map_err(|e| {
        TensorZeroClientError::Evaluation(format!("Failed to deserialize output: {e}"))
    })
}
