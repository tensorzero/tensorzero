use axum::Json;
use axum::debug_handler;
use axum::extract::{Path, State};
use tracing::instrument;
use uuid::Uuid;

use crate::db::delegating_connection::DelegatingDatabaseConnection;
use crate::db::resolve_uuid::{ResolveUuidQueries, ResolveUuidResponse};
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

#[debug_handler(state = AppStateData)]
#[instrument(name = "resolve_uuid_handler", skip_all, fields(id = %id))]
pub async fn resolve_uuid_handler(
    State(app_state): AppState,
    Path(id): Path<Uuid>,
) -> Result<Json<ResolveUuidResponse>, Error> {
    let database = DelegatingDatabaseConnection::new(
        app_state.clickhouse_connection_info.clone(),
        app_state.postgres_connection_info.clone(),
    );
    let object_types = database.resolve_uuid(&id).await?;
    Ok(Json(ResolveUuidResponse { id, object_types }))
}
