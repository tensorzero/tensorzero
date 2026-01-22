//! Action endpoint handler for the TensorZero Gateway.
//!
//! This module handles the action endpoint dispatch logic, including inference
//! and feedback actions with historical config snapshots.

use axum::extract::State;
use axum::{Json, debug_handler};
use tensorzero_core::endpoints::feedback::feedback;
use tensorzero_core::endpoints::inference::{InferenceOutput, inference};
use tensorzero_core::endpoints::internal::action::{
    ActionInput, ActionInputInfo, ActionResponse, get_or_load_config,
};
use tensorzero_core::error::{Error, ErrorDetails};
use tensorzero_core::utils::gateway::{AppState, AppStateData, StructuredJson};
use tracing::instrument;

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

/// Action execution logic.
///
/// Executes an inference or feedback action using a historical config snapshot.
async fn action(
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
                app_state.deferred_tasks.clone(),
            );

            let response = feedback(snapshot_app_state, *feedback_params, None).await?;
            Ok(ActionResponse::Feedback(response))
        }
    }
}
