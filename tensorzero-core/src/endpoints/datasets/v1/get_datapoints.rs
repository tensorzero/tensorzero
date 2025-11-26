use axum::extract::{Path, State};
use axum::Json;
use tracing::instrument;

use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::datasets::{DatasetQueries, GetDatapointsParams};
use crate::endpoints::datasets::validate_dataset_name;
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

use super::types::{GetDatapointsRequest, GetDatapointsResponse, ListDatapointsRequest};

const DEFAULT_LIMIT: u32 = 20;
const DEFAULT_OFFSET: u32 = 0;
const DEFAULT_ALLOW_STALE: bool = false;

/// Handler for the POST `/v1/datasets/{dataset_id}/list_datapoints` endpoint.
/// Lists datapoints from a dataset with optional filtering and pagination.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "datasets.v1.list_datapoints", skip(app_state, request))]
pub async fn list_datapoints_handler(
    State(app_state): AppState,
    Path(dataset_name): Path<String>,
    StructuredJson(request): StructuredJson<ListDatapointsRequest>,
) -> Result<Json<GetDatapointsResponse>, Error> {
    let response =
        list_datapoints(&app_state.clickhouse_connection_info, dataset_name, request).await?;

    Ok(Json(response))
}

/// Handler for the POST `/v1/datasets/get_datapoints` endpoint.
/// Retrieves specific datapoints by their IDs.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "datasets.v1.get_datapoints", skip(app_state, request))]
pub async fn get_datapoints_handler(
    State(app_state): AppState,
    StructuredJson(request): StructuredJson<GetDatapointsRequest>,
) -> Result<Json<GetDatapointsResponse>, Error> {
    tracing::warn!(
        "`Please use `/v1/datasets/{{dataset_name}}/get_datapoints` instead for better performance."
    );

    let response = get_datapoints(
        &app_state.clickhouse_connection_info,
        /*dataset_name=*/ None,
        request,
    )
    .await?;
    Ok(Json(response))
}

/// Handler for the POST `/v1/datasets/{dataset_name}/get_datapoints` endpoint.
/// Retrieves datapoints scoped to a dataset by their IDs.
#[axum::debug_handler(state = AppStateData)]
#[instrument(
    name = "datasets.v1.get_datapoints_by_dataset",
    skip(app_state, request)
)]
pub async fn get_datapoints_by_dataset_handler(
    State(app_state): AppState,
    Path(dataset_name): Path<String>,
    StructuredJson(request): StructuredJson<GetDatapointsRequest>,
) -> Result<Json<GetDatapointsResponse>, Error> {
    validate_dataset_name(&dataset_name)?;

    let response = get_datapoints(
        &app_state.clickhouse_connection_info,
        Some(dataset_name),
        request,
    )
    .await?;
    Ok(Json(response))
}

pub async fn list_datapoints(
    clickhouse: &ClickHouseConnectionInfo,
    dataset_name: String,
    request: ListDatapointsRequest,
) -> Result<GetDatapointsResponse, Error> {
    validate_dataset_name(&dataset_name)?;

    #[expect(deprecated)]
    let limit = if let Some(limit) = request.limit {
        limit
    } else if let Some(page_size) = request.page_size {
        tracing::warn!("`page_size` is deprecated. Please use `limit` instead.");
        page_size
    } else {
        DEFAULT_LIMIT
    };

    let params = GetDatapointsParams {
        dataset_name: Some(dataset_name),
        function_name: request.function_name,
        ids: None, // List all datapoints, not filtering by ID
        limit,
        offset: request.offset.unwrap_or(DEFAULT_OFFSET),
        allow_stale: DEFAULT_ALLOW_STALE,
        filter: request.filter,
        order_by: request.order_by,
        search_query_experimental: request.search_query_experimental,
    };

    let datapoints = clickhouse.get_datapoints(&params).await?;
    let datapoints = datapoints
        .into_iter()
        .map(|dp| dp.into_datapoint())
        .collect::<Result<Vec<_>, _>>()?;

    Ok(GetDatapointsResponse { datapoints })
}

pub async fn get_datapoints(
    clickhouse: &ClickHouseConnectionInfo,
    dataset_name: Option<String>,
    request: GetDatapointsRequest,
) -> Result<GetDatapointsResponse, Error> {
    // If no IDs are provided, return an empty response.
    if request.ids.is_empty() {
        return Ok(GetDatapointsResponse { datapoints: vec![] });
    }

    let params = GetDatapointsParams {
        dataset_name,
        function_name: None,
        ids: Some(request.ids),
        // Return all datapoints matching the IDs.
        limit: u32::MAX,
        offset: 0,
        // Get Datapoints by ID should return stale datapoints.
        allow_stale: true,
        filter: None,
        order_by: None,
        search_query_experimental: None,
    };

    let datapoints = clickhouse.get_datapoints(&params).await?;
    let datapoints = datapoints
        .into_iter()
        .map(|dp| dp.into_datapoint())
        .collect::<Result<Vec<_>, _>>()?;

    Ok(GetDatapointsResponse { datapoints })
}
