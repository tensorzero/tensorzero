use axum::Json;
use axum::extract::{Query, State};
use tracing::instrument;

use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::datasets::{DatasetQueries, GetDatasetMetadataParams};
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

use super::types::{DatasetMetadata, ListDatasetsRequest, ListDatasetsResponse};

/// Handler for the GET `/internal/datasets` endpoint.
/// Returns metadata for all datasets with optional filtering and pagination.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "datasets.v1.list_datasets", skip(app_state, params))]
pub async fn list_datasets_handler(
    State(app_state): AppState,
    Query(params): Query<ListDatasetsRequest>,
) -> Result<Json<ListDatasetsResponse>, Error> {
    let response = list_datasets(&app_state.clickhouse_connection_info, params).await?;
    Ok(Json(response))
}

/// List datasets with optional filtering and pagination.
///
/// This is the non-handler function for use by the embedded client.
pub async fn list_datasets(
    clickhouse: &ClickHouseConnectionInfo,
    params: ListDatasetsRequest,
) -> Result<ListDatasetsResponse, Error> {
    let db_params = GetDatasetMetadataParams {
        function_name: params.function_name,
        limit: params.limit,
        offset: params.offset,
    };
    let db_datasets = clickhouse.get_dataset_metadata(&db_params).await?;

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
