//! Public API types used for the TensorZero evaluations crate.
//! These types are constructed from tensorzero-optimizers, the Python client, and the Node client.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tensorzero_core::{
    cache::CacheEnabledMode,
    client::{
        ClientInferenceParams, FeedbackParams, FeedbackResponse, InferenceOutput, TensorZeroError,
    },
    config::UninitializedVariantInfo,
    db::BatchWriterHandle,
    db::delegating_connection::DelegatingDatabaseQueries,
    error::Error,
    evaluations::{EvaluationFunctionConfig, EvaluatorConfig},
    inference::types::storage::StoragePath,
    inference::types::stored_input::StoragePathResolver,
    utils::gateway::AppStateData,
};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::EvaluationUpdate;

/// Trait for executing inference and feedback requests during evaluations.
///
/// This abstraction allows evaluations to run both:
/// - Inside the gateway (via direct handler calls)
/// - Outside the gateway (via HTTP client or embedded gateway)
#[async_trait]
pub trait EvaluationsInferenceExecutor: Send + Sync {
    /// Execute an inference request and return the result.
    async fn inference(
        &self,
        params: ClientInferenceParams,
    ) -> Result<InferenceOutput, TensorZeroError>;

    /// Submit feedback for an inference.
    async fn feedback(&self, params: FeedbackParams) -> Result<FeedbackResponse, TensorZeroError>;

    /// Resolve a storage path to get the actual data (for file inputs stored in object storage).
    async fn resolve_storage_path(&self, storage_path: StoragePath) -> Result<String, Error>;
}

/// Wrapper around `Client` that implements `EvaluationsInferenceExecutor`.
/// Use this for CLI tools or when calling evaluations from outside the gateway.
pub struct ClientInferenceExecutor {
    client: tensorzero_core::client::Client,
}

impl ClientInferenceExecutor {
    pub fn new(client: tensorzero_core::client::Client) -> Self {
        Self { client }
    }
}

#[async_trait]
impl EvaluationsInferenceExecutor for ClientInferenceExecutor {
    async fn inference(
        &self,
        params: ClientInferenceParams,
    ) -> Result<InferenceOutput, TensorZeroError> {
        self.client.inference(params).await
    }

    async fn feedback(&self, params: FeedbackParams) -> Result<FeedbackResponse, TensorZeroError> {
        self.client.feedback(params).await
    }

    async fn resolve_storage_path(&self, storage_path: StoragePath) -> Result<String, Error> {
        self.client.resolve(storage_path).await
    }
}

/// Executor that calls gateway handlers directly without HTTP overhead.
/// This is used when running evaluations from within the gateway or embedded mode.
pub struct AppStateInferenceExecutor {
    app_state: AppStateData,
}

impl AppStateInferenceExecutor {
    pub fn new(app_state: AppStateData) -> Self {
        Self { app_state }
    }
}

#[async_trait]
impl EvaluationsInferenceExecutor for AppStateInferenceExecutor {
    async fn inference(
        &self,
        params: ClientInferenceParams,
    ) -> Result<InferenceOutput, TensorZeroError> {
        use tensorzero_core::endpoints::inference::Params as InferenceParams;

        // Convert ClientInferenceParams to endpoint Params
        let endpoint_params: InferenceParams = params
            .try_into()
            .map_err(|e: Error| TensorZeroError::Other { source: e.into() })?;

        // Call the inference endpoint directly
        let result = Box::pin(tensorzero_core::endpoints::inference::inference(
            self.app_state.config.clone(),
            &self.app_state.http_client,
            self.app_state.clickhouse_connection_info.clone(),
            self.app_state.postgres_connection_info.clone(),
            self.app_state.cache_manager.clone(),
            self.app_state.deferred_tasks.clone(),
            self.app_state.rate_limiting_manager.clone(),
            self.app_state.primary_datastore,
            endpoint_params,
            None, // No API key for internal calls
        ))
        .await
        .map_err(|e| TensorZeroError::Other { source: e.into() })?;

        Ok(result.output)
    }

    async fn feedback(&self, params: FeedbackParams) -> Result<FeedbackResponse, TensorZeroError> {
        // Call the feedback endpoint directly
        tensorzero_core::endpoints::feedback::feedback(self.app_state.clone(), params, None)
            .await
            .map_err(|e| TensorZeroError::Other { source: e.into() })
    }

    async fn resolve_storage_path(&self, storage_path: StoragePath) -> Result<String, Error> {
        // Use the object storage endpoint directly
        let response = tensorzero_core::endpoints::object_storage::get_object(
            self.app_state.config.object_store_info.as_ref(),
            storage_path,
        )
        .await?;
        Ok(response.data)
    }
}

/// Wrapper type that implements `StoragePathResolver` using an `EvaluationsInferenceExecutor`.
/// This is needed because we can't implement a foreign trait for `Arc<dyn EvaluationsInferenceExecutor>`.
pub struct ExecutorStorageResolver(pub Arc<dyn EvaluationsInferenceExecutor>);

impl tensorzero_core::inference::types::stored_input::StoragePathResolver
    for ExecutorStorageResolver
{
    async fn resolve(&self, storage_path: StoragePath) -> Result<String, Error> {
        self.0.resolve_storage_path(storage_path).await
    }
}

/// Specifies which variant to use for evaluation.
/// Either a variant name from the config, or a dynamic variant configuration.
#[derive(Clone, Debug)]
pub enum EvaluationVariant {
    /// Use a variant by name from the config file
    Name(String),
    /// Use a dynamically provided variant configuration
    Info(Box<UninitializedVariantInfo>),
}

/// Parameters for running an evaluation using run_evaluation_core
/// This struct encapsulates all the necessary components for evaluation execution
pub struct EvaluationCoreArgs {
    /// Executor for making inference requests during evaluation.
    /// Use `ClientInferenceExecutor` for CLI/external usage, or
    /// `GatewayInferenceExecutor` when running inside the gateway.
    pub inference_executor: Arc<dyn EvaluationsInferenceExecutor>,

    /// Database connection for dataset and evaluation queries
    pub db: Arc<dyn DelegatingDatabaseQueries>,

    /// Name of the function to evaluate
    pub function_name: String,

    /// Function configuration for output schema validation
    pub function_config: EvaluationFunctionConfig,

    /// Evaluator configurations (pre-resolved by caller)
    pub evaluators: HashMap<String, EvaluatorConfig>,

    /// Evaluation name for metric naming. `None` for standalone evaluators (top-level naming).
    pub evaluation_name: Option<String>,

    /// Unique identifier for this evaluation run
    pub evaluation_run_id: Uuid,

    /// Name of the dataset to run on.
    /// Either dataset_name or datapoint_ids must be provided, but not both.
    pub dataset_name: Option<String>,

    /// Specific datapoint IDs to evaluate.
    /// Either dataset_name or datapoint_ids must be provided, but not both.
    pub datapoint_ids: Option<Vec<Uuid>>,

    /// Variant to use for evaluation.
    /// Either a variant name from the config file, or a dynamic variant configuration.
    pub variant: EvaluationVariant,

    /// Number of concurrent requests to make.
    pub concurrency: usize,

    /// Cache configuration for inference requests
    pub inference_cache: CacheEnabledMode,

    /// Additional tags to apply to all inferences made during the evaluation.
    /// These tags will be added to each inference, with internal evaluation tags
    /// taking precedence in case of conflicts.
    pub tags: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunInfo {
    pub evaluation_run_id: Uuid,
    pub evaluation_name: String,
    pub num_datapoints: usize,
}

/// Result from running an evaluation that supports streaming
pub struct EvaluationStreamResult {
    pub receiver: mpsc::Receiver<EvaluationUpdate>,
    pub run_info: RunInfo,
    /// Evaluator configurations, for computing summary stats after the run completes.
    pub evaluators: HashMap<String, EvaluatorConfig>,
    /// Join handles for batch writers (ClickHouse, Postgres, etc.).
    /// The caller may want to wait for these to finish.
    pub batcher_join_handles: Vec<BatchWriterHandle>,
}

/// Parameters for running an evaluation using the app state directly.
/// This is used by the gateway handler and embedded mode in durable-tools.
pub struct RunEvaluationWithAppStateParams {
    /// Name of the function to evaluate
    pub function_name: String,

    /// Function configuration for output schema validation
    pub function_config: EvaluationFunctionConfig,

    /// Evaluator configurations (pre-resolved by caller)
    pub evaluators: HashMap<String, EvaluatorConfig>,

    /// Evaluation name for metric naming. `None` for standalone evaluators (top-level naming).
    pub evaluation_name: Option<String>,

    /// Name of the dataset to run on.
    /// Either dataset_name or datapoint_ids must be provided, but not both.
    pub dataset_name: Option<String>,

    /// Specific datapoint IDs to evaluate.
    /// Either dataset_name or datapoint_ids must be provided, but not both.
    pub datapoint_ids: Option<Vec<Uuid>>,

    /// Variant to use for evaluation.
    pub variant: EvaluationVariant,

    /// Number of concurrent requests to make.
    pub concurrency: usize,

    /// Cache configuration for inference requests
    pub cache_mode: CacheEnabledMode,

    /// Maximum number of datapoints to evaluate
    pub max_datapoints: Option<u32>,

    /// Precision targets for adaptive stopping.
    /// Maps evaluator names to target confidence interval half-widths.
    pub precision_targets: HashMap<String, f32>,

    /// Additional tags to apply to all inferences made during the evaluation.
    /// These tags will be added to each inference, with internal evaluation tags
    /// taking precedence in case of conflicts.
    pub tags: HashMap<String, String>,
}
