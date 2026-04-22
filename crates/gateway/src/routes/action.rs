//! Action endpoint handler for the TensorZero Gateway.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use durable_tools::action::{ActionInputInfo, ActionResponse, action};
use durable_tools::{Heartbeater, NoopHeartbeater};
use tensorzero_core::error::Error;
use tensorzero_core::utils::gateway::{AppState, StructuredJson};
use tracing::instrument;

/// Handler for `POST /internal/action`
///
/// Executes an inference, feedback, or evaluation action using a historical config snapshot.
#[instrument(name = "action", skip_all, fields(snapshot_hash = %params.snapshot_hash))]
pub async fn action_handler(
    State(app_state): AppState,
    StructuredJson(params): StructuredJson<ActionInputInfo>,
) -> Result<Json<ActionResponse>, Error> {
    let heartbeater: Arc<dyn Heartbeater> = Arc::new(NoopHeartbeater);
    let response = action(&app_state, params, heartbeater).await?;
    Ok(Json(response))
}
