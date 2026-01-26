//! Evaluation running logic and types.
//!
//! This module provides the types and runner for evaluations used by both the action
//! endpoint and embedded/SDK clients.
//!
//! These types differ from the HTTP wire types in `gateway/src/routes/evaluations.rs`:
//! the HTTP endpoint uses SSE streaming with per-datapoint events, while these types
//! provide an aggregated response suitable for programmatic use.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use evaluations::run_evaluation_with_app_state;
use evaluations::stats::{EvaluationStats, EvaluationUpdate};
use evaluations::types::{EvaluationVariant, RunEvaluationWithAppStateParams};
use tensorzero_core::cache::CacheEnabledMode;
use tensorzero_core::evaluations::{EvaluationConfig, EvaluationFunctionConfig};
use tensorzero_core::utils::gateway::AppStateData;

pub use evaluations::stats::EvaluatorStats;

// ============================================================================
// Errors
// ============================================================================

/// Error type for evaluation operations.
#[derive(Debug, Clone)]
pub enum RunEvaluationError {
    /// Validation error (invalid parameters, missing config). Maps to HTTP 400.
    Validation(String),
    /// Runtime error (execution failures). Maps to HTTP 500.
    Runtime(String),
}

impl std::fmt::Display for RunEvaluationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunEvaluationError::Validation(msg) => write!(f, "{msg}"),
            RunEvaluationError::Runtime(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for RunEvaluationError {}

// ============================================================================
// Types
// ============================================================================

fn default_concurrency() -> usize {
    10
}

fn default_inference_cache() -> CacheEnabledMode {
    CacheEnabledMode::On
}

/// Parameters for running an evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunEvaluationParams {
    /// Name of the evaluation to run (must be defined in config).
    pub evaluation_name: String,
    /// Name of the dataset to run on.
    /// Either `dataset_name` or `datapoint_ids` must be provided, but not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_name: Option<String>,
    /// Specific datapoint IDs to evaluate.
    /// Either `dataset_name` or `datapoint_ids` must be provided, but not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub datapoint_ids: Option<Vec<Uuid>>,
    /// Name of the variant to evaluate.
    pub variant_name: String,
    /// Number of concurrent inference requests (default: 10).
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    /// Cache configuration for inference requests (default: On).
    #[serde(default = "default_inference_cache")]
    pub inference_cache: CacheEnabledMode,
    /// Maximum number of datapoints to evaluate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_datapoints: Option<u32>,
    /// Precision targets for adaptive stopping (evaluator name -> target CI half-width).
    #[serde(default)]
    pub precision_targets: HashMap<String, f32>,
    /// Whether to include per-datapoint results in the response.
    #[serde(default)]
    pub include_datapoint_results: bool,
    /// Additional tags to apply to all inferences made during the evaluation.
    /// These tags will be added to each inference, with internal evaluation tags
    /// taking precedence in case of conflicts.
    #[serde(default)]
    pub tags: HashMap<String, String>,
}

/// Result for a single datapoint evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatapointResult {
    /// ID of the datapoint that was evaluated.
    pub datapoint_id: Uuid,
    /// Whether the evaluation succeeded (inference + at least one evaluator ran).
    pub success: bool,
    /// Per-evaluator scores (None if that evaluator failed or returned non-numeric).
    #[serde(default)]
    pub evaluations: HashMap<String, Option<f64>>,
    /// Per-evaluator error messages for evaluators that failed on this datapoint.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub evaluator_errors: HashMap<String, String>,
    /// Error message if the entire datapoint evaluation failed (e.g., inference error).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
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
    pub stats: HashMap<String, EvaluatorStats>,
    /// Per-datapoint results (only populated if `include_datapoint_results` was true).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub datapoint_results: Option<Vec<DatapointResult>>,
}

// ============================================================================
// Runner
// ============================================================================

/// Runs an evaluation: looks up configs, executes inference + evaluators, collects results.
pub async fn run_evaluation(
    app_state: AppStateData,
    params: &RunEvaluationParams,
) -> Result<RunEvaluationResponse, RunEvaluationError> {
    // Validate concurrency
    if params.concurrency == 0 {
        return Err(RunEvaluationError::Validation(
            "Concurrency must be greater than 0".to_string(),
        ));
    }

    // Validate max_datapoints if provided
    if params.max_datapoints == Some(0) {
        return Err(RunEvaluationError::Validation(
            "max_datapoints must be greater than 0".to_string(),
        ));
    }

    // Validate precision_targets values
    for (evaluator_name, target) in &params.precision_targets {
        if !target.is_finite() || *target <= 0.0 {
            return Err(RunEvaluationError::Validation(format!(
                "precision_target for `{evaluator_name}` must be a positive finite number, got {target}"
            )));
        }
    }

    // Validate exactly one of dataset_name or datapoint_ids is provided
    let has_dataset = params.dataset_name.is_some();
    let has_datapoints = params
        .datapoint_ids
        .as_ref()
        .is_some_and(|ids| !ids.is_empty());
    if has_dataset && has_datapoints {
        return Err(RunEvaluationError::Validation(
            "Cannot provide both dataset_name and datapoint_ids".to_string(),
        ));
    }
    if !has_dataset && !has_datapoints {
        return Err(RunEvaluationError::Validation(
            "Must provide either dataset_name or datapoint_ids".to_string(),
        ));
    }

    // Validate max_datapoints cannot be used with datapoint_ids
    if has_datapoints && params.max_datapoints.is_some() {
        return Err(RunEvaluationError::Validation(
            "Cannot use max_datapoints with datapoint_ids".to_string(),
        ));
    }

    let evaluation_config = app_state
        .config
        .evaluations
        .get(&params.evaluation_name)
        .ok_or_else(|| {
            RunEvaluationError::Validation(format!(
                "Evaluation `{}` not found in config",
                params.evaluation_name
            ))
        })?
        .clone();

    let EvaluationConfig::Inference(ref inference_config) = *evaluation_config;
    let function_config = app_state
        .config
        .functions
        .get(&inference_config.function_name)
        .map(|f| EvaluationFunctionConfig::from(f.as_ref()))
        .ok_or_else(|| {
            RunEvaluationError::Validation(format!(
                "Function `{}` not found in config",
                inference_config.function_name
            ))
        })?;

    let run_params = RunEvaluationWithAppStateParams {
        evaluation_config: (*evaluation_config).clone(),
        function_config,
        evaluation_name: params.evaluation_name.clone(),
        dataset_name: params.dataset_name.clone(),
        datapoint_ids: params.datapoint_ids.clone(),
        variant: EvaluationVariant::Name(params.variant_name.clone()),
        concurrency: params.concurrency,
        cache_mode: params.inference_cache,
        max_datapoints: params.max_datapoints,
        precision_targets: params.precision_targets.clone(),
        tags: params.tags.clone(),
    };

    let result = run_evaluation_with_app_state(app_state, run_params)
        .await
        .map_err(|e| RunEvaluationError::Runtime(format!("Evaluation failed: {e}")))?;

    collect_results(result, params.include_datapoint_results).await
}

/// Collects evaluation results from the stream and computes statistics.
async fn collect_results(
    result: evaluations::EvaluationStreamResult,
    include_datapoint_results: bool,
) -> Result<RunEvaluationResponse, RunEvaluationError> {
    let evaluation_run_id = result.run_info.evaluation_run_id;
    let num_datapoints = result.run_info.num_datapoints;

    let EvaluationConfig::Inference(ref inference_config) = *result.evaluation_config;

    // Collect updates from the stream (OutputFormat::Jsonl skips progress bar rendering)
    let mut stats_collector =
        EvaluationStats::new(evaluations::OutputFormat::Jsonl, num_datapoints);
    let mut sink = std::io::sink();
    let mut receiver = result.receiver;

    while let Some(update) = receiver.recv().await {
        if !matches!(update, EvaluationUpdate::RunInfo(_)) {
            let _ = stats_collector.push(update, &mut sink);
        }
    }

    let stats = stats_collector.compute_stats(&inference_config.evaluators);

    let datapoint_results = if include_datapoint_results {
        Some(build_datapoint_results(&stats_collector))
    } else {
        None
    };

    // Wait for ClickHouse writes to complete
    if let Some(handle) = result.batcher_join_handle {
        handle.await.map_err(|e| {
            RunEvaluationError::Runtime(format!("ClickHouse batch writer failed: {e}"))
        })?;
    }

    Ok(RunEvaluationResponse {
        evaluation_run_id,
        num_datapoints,
        num_successes: stats_collector.evaluation_infos.len(),
        num_errors: stats_collector.evaluation_errors.len(),
        stats,
        datapoint_results,
    })
}

/// Converts evaluation stats into per-datapoint results.
fn build_datapoint_results(stats: &EvaluationStats) -> Vec<DatapointResult> {
    let mut results =
        Vec::with_capacity(stats.evaluation_infos.len() + stats.evaluation_errors.len());

    for info in &stats.evaluation_infos {
        let evaluations = info
            .evaluations
            .iter()
            .map(|(name, value)| {
                let score = value.as_ref().and_then(|v| {
                    v.as_f64()
                        .or_else(|| v.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
                });
                (name.clone(), score)
            })
            .collect();

        results.push(DatapointResult {
            datapoint_id: info.datapoint.id(),
            success: true,
            evaluations,
            evaluator_errors: info.evaluator_errors.clone(),
            error: None,
        });
    }

    for error in &stats.evaluation_errors {
        results.push(DatapointResult {
            datapoint_id: error.datapoint_id,
            success: false,
            evaluations: HashMap::new(),
            evaluator_errors: HashMap::new(),
            error: Some(error.message.clone()),
        });
    }

    results
}
