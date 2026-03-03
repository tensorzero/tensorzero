use axum::Json;
use axum::debug_handler;
use axum::extract::{Path, State};
use tracing::instrument;
use uuid::Uuid;

use crate::db::resolve_uuid::{ResolveUuidQueries, ResolveUuidResponse};
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

#[cfg_attr(feature = "openapi", utoipa::path(
    get,
    path = "/internal/resolve_uuid/{id}",
    params(
        ("id" = String, Path, description = "The UUID to resolve"),
    ),
    responses(
        (status = 200, description = "Resolved UUID", body = inline(crate::db::resolve_uuid::ResolveUuidResponse)),
        (status = 400, description = "Bad request"),
    ),
    tag = "Internal"
))]
#[debug_handler(state = AppStateData)]
#[instrument(name = "resolve_uuid_handler", skip_all, fields(id = %id))]
pub async fn resolve_uuid_handler(
    State(app_state): AppState,
    Path(id): Path<Uuid>,
) -> Result<Json<ResolveUuidResponse>, Error> {
    let database = app_state.get_delegating_database();
    let object_types = database.resolve_uuid(&id).await?;
    Ok(Json(ResolveUuidResponse { id, object_types }))
}
