use axum::Json;
use axum::extract::{Query, State};
use tracing::instrument;

use crate::db::datasets::{DatasetQueries, GetDatasetMetadataParams};
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

use super::types::{DatasetMetadata, ListDatasetsRequest, ListDatasetsResponse};

/// Handler for the GET `/internal/datasets` endpoint.
/// Returns metadata for all datasets with optional filtering and pagination.
#[cfg_attr(feature = "openapi", utoipa::path(
    get,
    path = "/internal/datasets",
    responses(
        (status = 200, description = "List of datasets", body = ListDatasetsResponse),
        (status = 400, description = "Bad request"),
    ),
    tag = "Internal"
))]
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "datasets.v1.list_datasets", skip(app_state, params))]
pub async fn list_datasets_handler(
    State(app_state): AppState,
    Query(params): Query<ListDatasetsRequest>,
) -> Result<Json<ListDatasetsResponse>, Error> {
    let database = app_state.get_delegating_database();
    let response = list_datasets(&database, params).await?;
    Ok(Json(response))
}

/// List datasets with optional filtering and pagination.
///
/// This is the non-handler function for use by the embedded client.
pub async fn list_datasets(
    database: &(dyn DatasetQueries + Sync),
    params: ListDatasetsRequest,
) -> Result<ListDatasetsResponse, Error> {
    let db_params = GetDatasetMetadataParams {
        function_name: params.function_name,
        limit: params.limit,
        offset: params.offset,
    };
    let db_datasets = database.get_dataset_metadata(&db_params).await?;

    // Convert from DB type to API type
    let datasets = db_datasets
        .into_iter()
        .map(|db_meta| DatasetMetadata {
            dataset_name: db_meta.dataset_name,
            datapoint_count: db_meta.count,
            last_updated: db_meta.last_updated,
        })
        .collect();

    Ok(ListDatasetsResponse { datasets })
}
