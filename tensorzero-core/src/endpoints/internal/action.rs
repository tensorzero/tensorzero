//! Action endpoint types and helpers for executing inference or feedback with historical config snapshots.
//!
//! This module provides type definitions and config loading utilities for the action endpoint.
//! The action dispatch logic lives in the gateway.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tensorzero_derive::TensorZeroDeserialize;
use uuid::Uuid;

use crate::cache::CacheEnabledMode;
use crate::client::client_inference_params::ClientInferenceParams;
use crate::config::snapshot::SnapshotHash;
use crate::config::{Config, RuntimeOverlay};
use crate::db::ConfigQueries;
use crate::endpoints::feedback::{FeedbackResponse, Params as FeedbackParams};
use crate::endpoints::inference::InferenceResponse;
use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::AppStateData;

/// Input for the action endpoint.
#[derive(Debug, Deserialize, Serialize)]
pub struct ActionInputInfo {
    /// The snapshot hash identifying which config version to use.
    pub snapshot_hash: SnapshotHash,
    /// The action to perform (inference, feedback, or run_evaluation).
    #[serde(flatten)]
    pub input: ActionInput,
}

fn default_concurrency() -> usize {
    10
}

/// Parameters for running an evaluation via the action endpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct RunEvaluationActionParams {
    /// Name of the evaluation to run (must be defined in config).
    pub evaluation_name: String,
    /// Name of the dataset to run on.
    /// Either dataset_name or datapoint_ids must be provided, but not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub dataset_name: Option<String>,
    /// Specific datapoint IDs to evaluate.
    /// Either dataset_name or datapoint_ids must be provided, but not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub datapoint_ids: Option<Vec<Uuid>>,
    /// Name of the variant to evaluate.
    pub variant_name: String,
    /// Number of concurrent inference requests.
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    /// Cache configuration for inference requests.
    #[serde(default)]
    pub inference_cache: CacheEnabledMode,
    /// Maximum number of datapoints to evaluate.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub max_datapoints: Option<u32>,
    /// Precision targets for adaptive stopping.
    /// Maps evaluator names to target confidence interval half-widths.
    #[serde(default)]
    pub precision_targets: HashMap<String, f32>,
    /// Include per-datapoint results in the response.
    #[serde(default)]
    pub include_datapoint_results: bool,
}

/// Statistics for a single evaluator.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct EvaluatorStatsResponse {
    /// Mean value of the evaluator.
    pub mean: f32,
    /// Standard error of the evaluator.
    pub stderr: f32,
    /// Number of samples.
    pub count: usize,
}

/// Result for a single datapoint evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct DatapointResult {
    /// ID of the datapoint that was evaluated.
    pub datapoint_id: Uuid,
    /// Whether the evaluation succeeded (inference + at least one evaluator ran).
    pub success: bool,
    /// Per-evaluator scores for this datapoint.
    /// Only populated for successful evaluations.
    #[serde(default)]
    pub evaluations: HashMap<String, Option<f64>>,
    /// Per-evaluator error messages for evaluators that failed on this datapoint.
    /// A datapoint can have both successful evaluations and evaluator errors
    /// if some evaluators succeeded while others failed.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub evaluator_errors: HashMap<String, String>,
    /// Error message if the entire datapoint evaluation failed (e.g., inference error).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub error: Option<String>,
}

/// Response from running an evaluation via the action endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct RunEvaluationActionResponse {
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
    /// Per-datapoint results (only populated if include_datapoint_results was true).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub datapoint_results: Option<Vec<DatapointResult>>,
}

/// The specific action type to execute.
#[derive(Clone, Debug, Serialize, TensorZeroDeserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ActionInput {
    Inference(Box<ClientInferenceParams>),
    Feedback(Box<FeedbackParams>),
    RunEvaluation(Box<RunEvaluationActionParams>),
}

/// Response from the action endpoint.
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ActionResponse {
    Inference(InferenceResponse),
    Feedback(FeedbackResponse),
    RunEvaluation(RunEvaluationActionResponse),
}

/// Get config from cache or load from snapshot.
///
/// This helper is used by the gateway's action handler to load historical
/// config snapshots for reproducible inference and feedback execution.
pub async fn get_or_load_config(
    app_state: &AppStateData,
    snapshot_hash: &SnapshotHash,
) -> Result<Arc<Config>, Error> {
    let cache = app_state.config_snapshot_cache.as_ref().ok_or_else(|| {
        Error::new(ErrorDetails::InvalidRequest {
            message: "Config snapshot cache is not enabled".to_string(),
        })
    })?;

    // Cache hit
    if let Some(config) = cache.get(snapshot_hash) {
        return Ok(config);
    }

    // Cache miss: load from ClickHouse
    let snapshot = app_state
        .clickhouse_connection_info
        .get_config_snapshot(snapshot_hash.clone())
        .await?;

    let runtime_overlay = RuntimeOverlay::from_config(&app_state.config);

    let unwritten_config = Config::load_from_snapshot(
        snapshot,
        runtime_overlay,
        false, // Don't validate credentials for historical configs
    )
    .await?;

    let config = Arc::new(unwritten_config.dangerous_into_config_without_writing());

    cache.insert(snapshot_hash.clone(), config.clone());

    Ok(config)
}
