//! Action endpoint types and dispatch logic for the TensorZero action endpoint.
//!
//! This module provides the type definitions and core action dispatch logic used by both
//! the gateway HTTP handler and embedded clients.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tensorzero_derive::TensorZeroDeserialize;

use tensorzero_core::client::client_inference_params::ClientInferenceParams;
use tensorzero_core::config::snapshot::SnapshotHash;
use tensorzero_core::config::{Config, RuntimeOverlay};
use tensorzero_core::db::ConfigQueries;
use tensorzero_core::endpoints::feedback::feedback;
use tensorzero_core::endpoints::feedback::{FeedbackResponse, Params as FeedbackParams};
use tensorzero_core::endpoints::inference::{InferenceOutput, InferenceResponse, inference};
use tensorzero_core::error::{Error, ErrorDetails};
use tensorzero_core::utils::gateway::AppStateData;

use crate::run_evaluation::{RunEvaluationError, run_evaluation};

// Re-export evaluation types from run_evaluation (single source of truth)
pub use crate::run_evaluation::{
    DatapointResult, EvaluatorStats, RunEvaluationParams, RunEvaluationResponse,
};

// ============================================================================
// Types
// ============================================================================

/// Input for the action endpoint.
#[derive(Debug, Deserialize, Serialize)]
pub struct ActionInputInfo {
    /// The snapshot hash identifying which config version to use.
    pub snapshot_hash: SnapshotHash,
    /// The action to perform (inference, feedback, or run_evaluation).
    #[serde(flatten)]
    pub input: ActionInput,
}

/// The specific action type to execute.
#[derive(Clone, Debug, Serialize, TensorZeroDeserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ActionInput {
    Inference(Box<ClientInferenceParams>),
    Feedback(Box<FeedbackParams>),
    RunEvaluation(Box<RunEvaluationParams>),
}

/// Response from the action endpoint.
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ActionResponse {
    Inference(InferenceResponse),
    Feedback(FeedbackResponse),
    RunEvaluation(RunEvaluationResponse),
}

// ============================================================================
// Config Loading
// ============================================================================

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

// ============================================================================
// Dispatch Logic
// ============================================================================

/// Executes an inference or feedback action using a historical config snapshot.
pub async fn action(
    app_state: &AppStateData,
    params: ActionInputInfo,
) -> Result<ActionResponse, Error> {
    let config = get_or_load_config(app_state, &params.snapshot_hash).await?;

    match params.input {
        ActionInput::Inference(inference_params) => {
            // Reject streaming requests
            if inference_params.stream.unwrap_or(false) {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: "Streaming is not supported for the action endpoint".to_string(),
                }));
            }

            let data = Box::pin(inference(
                config,
                &app_state.http_client,
                app_state.clickhouse_connection_info.clone(),
                app_state.postgres_connection_info.clone(),
                app_state.deferred_tasks.clone(),
                app_state.rate_limiting_manager.clone(),
                (*inference_params).try_into()?,
                None, // No API key for internal endpoint
            ))
            .await?;

            match data.output {
                InferenceOutput::NonStreaming(response) => Ok(ActionResponse::Inference(response)),
                InferenceOutput::Streaming(_) => {
                    // Should not happen since we checked stream=false above
                    Err(Error::new(ErrorDetails::InternalError {
                        message: "Unexpected streaming response".to_string(),
                    }))
                }
            }
        }
        ActionInput::Feedback(feedback_params) => {
            // Build AppStateData with snapshot config
            let snapshot_app_state = AppStateData::new_for_snapshot(
                config,
                app_state.http_client.clone(),
                app_state.clickhouse_connection_info.clone(),
                app_state.postgres_connection_info.clone(),
                app_state.valkey_connection_info.clone(),
                app_state.deferred_tasks.clone(),
                app_state.shutdown_token.clone(),
            )?;

            let response = feedback(snapshot_app_state, *feedback_params, None).await?;
            Ok(ActionResponse::Feedback(response))
        }
        ActionInput::RunEvaluation(eval_params) => {
            // Build AppStateData with snapshot config
            let snapshot_app_state = AppStateData::new_for_snapshot(
                config,
                app_state.http_client.clone(),
                app_state.clickhouse_connection_info.clone(),
                app_state.postgres_connection_info.clone(),
                app_state.valkey_connection_info.clone(),
                app_state.deferred_tasks.clone(),
                app_state.shutdown_token.clone(),
            )?;

            // Run the evaluation using the shared helper
            let response = run_evaluation(snapshot_app_state, &eval_params)
                .await
                .map_err(|e| match e {
                    RunEvaluationError::Validation(message) => {
                        Error::new(ErrorDetails::InvalidRequest { message })
                    }
                    RunEvaluationError::Runtime(message) => {
                        Error::new(ErrorDetails::EvaluationRun { message })
                    }
                })?;

            Ok(ActionResponse::RunEvaluation(response))
        }
    }
}
