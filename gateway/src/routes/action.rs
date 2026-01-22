//! Action endpoint handler for the TensorZero Gateway.

use axum::extract::State;
use axum::{Json, debug_handler};
use durable_tools::action::action;
use tensorzero_core::endpoints::internal::action::{ActionInputInfo, ActionResponse};
use tensorzero_core::error::Error;
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
