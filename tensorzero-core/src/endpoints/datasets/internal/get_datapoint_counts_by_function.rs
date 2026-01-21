//! Endpoint for getting datapoint counts grouped by function for a dataset.

use axum::extract::{Path, State};
use axum::{Json, debug_handler};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::db::datasets::{DatasetQueries, FunctionDatapointCount};
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

/// Response containing datapoint counts grouped by function
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct GetDatapointCountsByFunctionResponse {
    /// The counts of datapoints per function, ordered by count DESC
    pub counts: Vec<FunctionDatapointCount>,
}

/// Gets datapoint counts grouped by function for a dataset
pub async fn get_datapoint_counts_by_function(
    clickhouse: &impl DatasetQueries,
    dataset_name: &str,
) -> Result<GetDatapointCountsByFunctionResponse, Error> {
    let counts = clickhouse
        .count_datapoints_by_function(dataset_name)
        .await?;

    Ok(GetDatapointCountsByFunctionResponse { counts })
}

/// HTTP handler for the datapoint counts by function endpoint
#[debug_handler(state = AppStateData)]
#[instrument(
    name = "get_datapoint_counts_by_function_handler",
    skip_all,
    fields(
        dataset_name = %dataset_name,
    )
)]
pub async fn get_datapoint_counts_by_function_handler(
    State(app_state): AppState,
    Path(dataset_name): Path<String>,
) -> Result<Json<GetDatapointCountsByFunctionResponse>, Error> {
    let response =
        get_datapoint_counts_by_function(&app_state.clickhouse_connection_info, &dataset_name)
            .await?;

    Ok(Json(response))
}
