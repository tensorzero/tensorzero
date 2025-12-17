//! Action endpoint for executing inference or feedback with historical config snapshots.
//!
//! This endpoint allows executing actions using a specific config snapshot from the database,
//! enabling reproducibility and testing with historical configurations.

use std::sync::Arc;

use axum::extract::State;
use axum::{Json, debug_handler};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::client::client_inference_params::ClientInferenceParams;
use crate::config::snapshot::SnapshotHash;
use crate::config::{Config, RuntimeOverlay};
use crate::db::ConfigQueries;
use crate::endpoints::feedback::{FeedbackResponse, Params as FeedbackParams, feedback};
use crate::endpoints::inference::{InferenceOutput, InferenceResponse, inference};
use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

/// Input for the action endpoint.
#[derive(Debug, Deserialize, Serialize)]
pub struct ActionInputInfo {
    /// The snapshot hash identifying which config version to use.
    pub snapshot_hash: SnapshotHash,
    /// The action to perform (inference or feedback).
    #[serde(flatten)]
    pub input: ActionInput,
}

/// The specific action type to execute.
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ActionInput {
    Inference(Box<ClientInferenceParams>),
    Feedback(Box<FeedbackParams>),
}

/// Response from the action endpoint.
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ActionResponse {
    Inference(InferenceResponse),
    Feedback(FeedbackResponse),
}

/// Handler for `POST /internal/action`
///
/// Executes an inference or feedback action using a historical config snapshot.
#[debug_handler(state = AppStateData)]
#[instrument(name = "action", skip_all, fields(snapshot_hash = %params.snapshot_hash))]
pub async fn action_handler(
    State(app_state): AppState,
    StructuredJson(params): StructuredJson<ActionInputInfo>,
) -> Result<Json<ActionResponse>, Error> {
    let response = action(&app_state, params).await?;
    Ok(Json(response))
}

/// Core action execution logic (framework-agnostic).
///
/// Executes an inference or feedback action using a historical config snapshot.
/// This function can be called directly from the Rust client without HTTP overhead.
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

            let output = Box::pin(inference(
                config,
                &app_state.http_client,
                app_state.clickhouse_connection_info.clone(),
                app_state.postgres_connection_info.clone(),
                app_state.deferred_tasks.clone(),
                (*inference_params).try_into()?,
                None, // No API key for internal endpoint
            ))
            .await?;

            match output {
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
                app_state.deferred_tasks.clone(),
            );

            let response = feedback(snapshot_app_state, *feedback_params, None).await?;
            Ok(ActionResponse::Feedback(response))
        }
    }
}

/// Get config from cache or load from snapshot.
async fn get_or_load_config(
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
