//! Evaluation streaming endpoint.
//!
//! This module provides an SSE endpoint for running evaluations and streaming results.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use axum::extract::State;
use axum::response::IntoResponse;
use axum::response::sse::{Event, KeepAlive, Sse};
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;
use uuid::Uuid;

use evaluations::stats::{EvaluationError, EvaluationInfo, EvaluationUpdate};
use evaluations::{
    EvaluationCoreArgs, EvaluationVariant, EvaluationsInferenceExecutor,
    run_evaluation_core_streaming,
};
use tensorzero_core::cache::CacheEnabledMode;
use tensorzero_core::client::{
    ClientInferenceParams, FeedbackParams, FeedbackResponse, InferenceOutput,
};
use tensorzero_core::config::UninitializedVariantInfo;
use tensorzero_core::db::clickhouse::ClickHouseConnectionInfo;
use tensorzero_core::error::{Error, ErrorDetails};
use tensorzero_core::evaluations::{EvaluationConfig, EvaluationFunctionConfig};
use tensorzero_core::inference::types::storage::StoragePath;
use tensorzero_core::utils::gateway::{AppState, AppStateData, StructuredJson};

// =============================================================================
// GatewayInferenceExecutor
// =============================================================================

/// Executor that calls gateway handlers directly without HTTP overhead.
/// This is used when running evaluations from within the gateway.
pub struct GatewayInferenceExecutor {
    app_state: AppStateData,
}

impl GatewayInferenceExecutor {
    pub fn new(app_state: AppStateData) -> Self {
        Self { app_state }
    }
}

#[async_trait]
impl EvaluationsInferenceExecutor for GatewayInferenceExecutor {
    async fn inference(
        &self,
        params: ClientInferenceParams,
    ) -> Result<InferenceOutput, tensorzero_core::client::TensorZeroError> {
        use tensorzero_core::client::TensorZeroError;
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
            self.app_state.deferred_tasks.clone(),
            endpoint_params,
            None, // No API key for internal calls
        ))
        .await
        .map_err(|e| TensorZeroError::Other { source: e.into() })?;

        Ok(result.output)
    }

    async fn feedback(
        &self,
        params: FeedbackParams,
    ) -> Result<FeedbackResponse, tensorzero_core::client::TensorZeroError> {
        use tensorzero_core::client::TensorZeroError;

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

// =============================================================================
// Request/Response Types
// =============================================================================

/// Request body for running an evaluation.
#[derive(Debug, Clone, Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
pub struct RunEvaluationRequest {
    /// The evaluation configuration (serialized)
    pub evaluation_config: EvaluationConfig,
    /// The function configuration for output schema validation
    pub function_config: EvaluationFunctionConfig,
    /// Name of the evaluation
    pub evaluation_name: String,
    /// Name of the dataset to evaluate (optional, either this or datapoint_ids must be provided)
    #[serde(default)]
    pub dataset_name: Option<String>,
    /// Specific datapoint IDs to evaluate (optional)
    #[serde(default)]
    pub datapoint_ids: Option<Vec<Uuid>>,
    /// Variant to use: either a variant name or a dynamic variant configuration
    #[serde(default)]
    pub variant_name: Option<String>,
    /// Dynamic variant configuration (alternative to variant_name)
    #[serde(default)]
    pub internal_dynamic_variant_config: Option<UninitializedVariantInfo>,
    /// Number of concurrent requests
    #[serde(default = "default_concurrency")]
    pub concurrency: u32,
    /// Cache mode for inference requests
    #[serde(default = "default_inference_cache")]
    pub inference_cache: String,
    /// Maximum number of datapoints to evaluate
    #[serde(default)]
    pub max_datapoints: Option<u32>,
    /// Per-evaluator precision targets for adaptive stopping
    #[serde(default)]
    pub precision_targets: Option<HashMap<String, f64>>,
}

fn default_concurrency() -> u32 {
    5
}

fn default_inference_cache() -> String {
    "on".to_string()
}

/// SSE event types for evaluation streaming.
#[derive(Debug, Clone, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EvaluationRunEvent {
    Start(EvaluationRunStartEvent),
    Success(EvaluationRunSuccessEvent),
    Error(EvaluationRunErrorEvent),
    FatalError(EvaluationRunFatalErrorEvent),
    Complete(EvaluationRunCompleteEvent),
}

#[derive(Debug, Clone, Serialize, ts_rs::TS)]
#[ts(export, optional_fields)]
pub struct EvaluationRunStartEvent {
    pub evaluation_run_id: Uuid,
    pub num_datapoints: usize,
    pub evaluation_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct EvaluationRunSuccessEvent {
    pub evaluation_run_id: Uuid,
    pub datapoint: Value,
    pub response: Value,
    pub evaluations: HashMap<String, Option<Value>>,
    pub evaluator_errors: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct EvaluationRunErrorEvent {
    pub evaluation_run_id: Uuid,
    pub datapoint_id: Uuid,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, ts_rs::TS)]
#[ts(export, optional_fields)]
pub struct EvaluationRunFatalErrorEvent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluation_run_id: Option<Uuid>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct EvaluationRunCompleteEvent {
    pub evaluation_run_id: Uuid,
}

impl EvaluationRunSuccessEvent {
    fn try_from_info(evaluation_run_id: Uuid, info: EvaluationInfo) -> Result<Self, Error> {
        let datapoint = serde_json::to_value(info.datapoint).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize datapoint: {e}"),
            })
        })?;
        let response = serde_json::to_value(info.response).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize inference response: {e}"),
            })
        })?;

        Ok(Self {
            evaluation_run_id,
            datapoint,
            response,
            evaluations: info.evaluations,
            evaluator_errors: info.evaluator_errors,
        })
    }
}

// =============================================================================
// Stream Creation
// =============================================================================

struct EvaluationStreamParams {
    evaluation_run_id: Uuid,
    num_datapoints: usize,
    evaluation_name: String,
    dataset_name: Option<String>,
    variant_name: Option<String>,
    receiver: mpsc::Receiver<EvaluationUpdate>,
    clickhouse_client: ClickHouseConnectionInfo,
}

fn create_evaluation_stream(
    params: EvaluationStreamParams,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let EvaluationStreamParams {
        evaluation_run_id,
        num_datapoints,
        evaluation_name,
        dataset_name,
        variant_name,
        mut receiver,
        clickhouse_client,
    } = params;

    async_stream::stream! {
        // Send start event
        let start_event = EvaluationRunEvent::Start(EvaluationRunStartEvent {
            evaluation_run_id,
            num_datapoints,
            evaluation_name,
            dataset_name,
            variant_name,
        });

        match serde_json::to_string(&start_event) {
            Ok(data) => yield Ok(Event::default().event("event").data(data)),
            Err(e) => {
                let error_event = EvaluationRunEvent::FatalError(EvaluationRunFatalErrorEvent {
                    evaluation_run_id: Some(evaluation_run_id),
                    message: format!("Failed to serialize start event: {e}"),
                });
                if let Ok(data) = serde_json::to_string(&error_event) {
                    yield Ok(Event::default().event("event").data(data));
                }
                return;
            }
        }

        // Stream evaluation updates
        while let Some(update) = receiver.recv().await {
            let event = match update {
                EvaluationUpdate::RunInfo(_) => continue, // Already sent start event
                EvaluationUpdate::Success(info) => {
                    match EvaluationRunSuccessEvent::try_from_info(evaluation_run_id, info) {
                        Ok(success_event) => EvaluationRunEvent::Success(success_event),
                        Err(e) => EvaluationRunEvent::Error(EvaluationRunErrorEvent {
                            evaluation_run_id,
                            datapoint_id: Uuid::nil(),
                            message: format!("Failed to serialize success event: {e}"),
                        }),
                    }
                }
                EvaluationUpdate::Error(EvaluationError { datapoint_id, message }) => {
                    EvaluationRunEvent::Error(EvaluationRunErrorEvent {
                        evaluation_run_id,
                        datapoint_id,
                        message,
                    })
                }
            };

            match serde_json::to_string(&event) {
                Ok(data) => yield Ok(Event::default().event("event").data(data)),
                Err(e) => {
                    tracing::error!("Failed to serialize evaluation event: {e}");
                    continue;
                }
            }
        }

        // Wait for ClickHouse batch writer to finish
        let join_handle = clickhouse_client.batcher_join_handle();
        drop(clickhouse_client);

        if let Some(handle) = join_handle
            && let Err(e) = handle.await
        {
            let error_event = EvaluationRunEvent::FatalError(EvaluationRunFatalErrorEvent {
                evaluation_run_id: Some(evaluation_run_id),
                message: format!("Error waiting for ClickHouse batch writer: {e}"),
            });
            if let Ok(data) = serde_json::to_string(&error_event) {
                yield Ok(Event::default().event("event").data(data));
            }
            return;
        }

        // Send complete event
        let complete_event = EvaluationRunEvent::Complete(EvaluationRunCompleteEvent {
            evaluation_run_id,
        });
        if let Ok(data) = serde_json::to_string(&complete_event) {
            yield Ok(Event::default().event("event").data(data));
        }
    }
}

// =============================================================================
// HTTP Handler
// =============================================================================

/// Handler for `POST /internal/evaluations/run`
///
/// Runs an evaluation and streams results via SSE.
#[axum::debug_handler(state = AppStateData)]
pub async fn run_evaluation_handler(
    State(app_state): AppState,
    StructuredJson(request): StructuredJson<RunEvaluationRequest>,
) -> Result<impl IntoResponse, Error> {
    tracing::info!(
        evaluation_name = %request.evaluation_name,
        dataset_name = ?request.dataset_name,
        variant_name = ?request.variant_name,
        concurrency = %request.concurrency,
        "Starting evaluation run"
    );

    // Parse cache mode
    let cache_mode = match request.inference_cache.as_str() {
        "on" => CacheEnabledMode::On,
        "off" => CacheEnabledMode::Off,
        "read_only" => CacheEnabledMode::ReadOnly,
        "write_only" => CacheEnabledMode::WriteOnly,
        other => {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: format!("Invalid inference cache setting '{other}'"),
            }));
        }
    };

    // Parse concurrency
    let concurrency = usize::try_from(request.concurrency).map_err(|_| {
        Error::new(ErrorDetails::InvalidRequest {
            message: format!(
                "Concurrency {} is larger than supported on this platform",
                request.concurrency
            ),
        })
    })?;

    // Determine the variant
    let variant = match (
        request.variant_name.clone(),
        request.internal_dynamic_variant_config,
    ) {
        (Some(name), None) => EvaluationVariant::Name(name),
        (None, Some(config)) => EvaluationVariant::Info(Box::new(config)),
        (Some(_), Some(_)) => {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "Cannot provide both variant_name and internal_dynamic_variant_config"
                    .into(),
            }));
        }
        (None, None) => {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message:
                    "Exactly one of variant_name or internal_dynamic_variant_config must be provided"
                        .into(),
            }));
        }
    };

    // Create a new ClickHouse client for the evaluation (with independent batch writer)
    let clickhouse_client = app_state
        .clickhouse_connection_info
        .recreate()
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::ClickHouseConnection {
                message: format!("Failed to create ClickHouse client for evaluation: {e}"),
            })
        })?;

    // Create GatewayInferenceExecutor to call handlers directly without HTTP overhead
    let inference_executor = Arc::new(GatewayInferenceExecutor::new(app_state.clone()));

    // Extract function name from evaluation config
    let EvaluationConfig::Inference(ref inference_eval_config) = request.evaluation_config;
    let function_name = inference_eval_config.function_name.clone();

    // Build function configs table
    let mut function_configs = evaluations::EvaluationFunctionConfigTable::new();
    function_configs.insert(function_name, request.function_config);
    let function_configs = Arc::new(function_configs);

    // Generate a new evaluation run ID
    let evaluation_run_id = Uuid::now_v7();

    // Parse precision_targets
    let precision_targets: HashMap<String, f32> = request
        .precision_targets
        .unwrap_or_default()
        .into_iter()
        .map(|(k, v)| (k, v as f32))
        .collect();

    // Get variant name for the start event
    let variant_name_for_event = match &variant {
        EvaluationVariant::Name(name) => Some(name.clone()),
        EvaluationVariant::Info(_) => None,
    };

    let dataset_name_for_event = request.dataset_name.clone();
    let evaluation_name_for_event = request.evaluation_name.clone();

    // Build the core args
    let core_args = EvaluationCoreArgs {
        inference_executor,
        clickhouse_client: clickhouse_client.clone(),
        evaluation_config: Arc::new(request.evaluation_config),
        function_configs,
        dataset_name: request.dataset_name,
        datapoint_ids: request.datapoint_ids,
        variant,
        evaluation_name: request.evaluation_name,
        evaluation_run_id,
        inference_cache: cache_mode,
        concurrency,
    };

    // Start the evaluation
    let result =
        run_evaluation_core_streaming(core_args, request.max_datapoints, precision_targets)
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::EvaluationRun {
                    message: format!("Failed to start evaluation run: {e}"),
                })
            })?;

    let receiver = result.receiver;
    let num_datapoints = result.run_info.num_datapoints;

    // Create the SSE stream
    let sse_stream = create_evaluation_stream(EvaluationStreamParams {
        evaluation_run_id,
        num_datapoints,
        evaluation_name: evaluation_name_for_event,
        dataset_name: dataset_name_for_event,
        variant_name: variant_name_for_event,
        receiver,
        clickhouse_client,
    });

    Ok(Sse::new(sse_stream).keep_alive(KeepAlive::new()))
}
