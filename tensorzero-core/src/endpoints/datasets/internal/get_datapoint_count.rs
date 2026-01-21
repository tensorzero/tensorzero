//! Datapoint statistics endpoint for getting datapoint counts.

use axum::extract::{Path, Query, State};
use axum::{Json, debug_handler};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::db::datasets::DatasetQueries;
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

/// Query parameters for the datapoint count endpoint
#[derive(Debug, Deserialize)]
pub struct GetDatapointCountQueryParams {
    /// Optional function name to filter by
    pub function_name: Option<String>,
}

/// Response containing datapoint counts
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GetDatapointCountResponse {
    /// The count of datapoints for the dataset
    pub datapoint_count: u64,
}

/// Gets datapoint counts for a dataset
pub async fn get_datapoint_count(
    clickhouse: &impl DatasetQueries,
    dataset_name: &str,
    function_name: Option<&str>,
) -> Result<GetDatapointCountResponse, Error> {
    let datapoint_count = clickhouse
        .count_datapoints_for_dataset(dataset_name, function_name)
        .await?;

    Ok(GetDatapointCountResponse { datapoint_count })
}

/// HTTP handler for the datapoint count endpoint
#[debug_handler(state = AppStateData)]
#[instrument(
    name = "get_datapoint_count_handler",
    skip_all,
    fields(
        dataset_name = %dataset_name,
    )
)]
pub async fn get_datapoint_count_handler(
    State(app_state): AppState,
    Path(dataset_name): Path<String>,
    Query(params): Query<GetDatapointCountQueryParams>,
) -> Result<Json<GetDatapointCountResponse>, Error> {
    let response = get_datapoint_count(
        &app_state.clickhouse_connection_info,
        &dataset_name,
        params.function_name.as_deref(),
    )
    .await?;

    Ok(Json(response))
}
