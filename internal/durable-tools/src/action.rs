//! Action execution logic for the TensorZero action endpoint.
//!
//! This module provides the core action dispatch logic used by both the gateway
//! HTTP handler and embedded clients.

use tensorzero_core::endpoints::feedback::feedback;
use tensorzero_core::endpoints::inference::{InferenceOutput, inference};
use tensorzero_core::endpoints::internal::action::{
    ActionInput, ActionInputInfo, ActionResponse, get_or_load_config,
};
use tensorzero_core::error::{Error, ErrorDetails};
use tensorzero_core::utils::gateway::AppStateData;

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
            )?;

            let response = feedback(snapshot_app_state, *feedback_params, None).await?;
            Ok(ActionResponse::Feedback(response))
        }
    }
}
