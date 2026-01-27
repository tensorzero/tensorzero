use axum::Json;
use axum::extract::{Path, State};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::db::datasets::DatasetQueries;
use crate::error::Error;
use crate::utils::gateway::{AppState, StructuredJson};

use super::super::legacy::validate_dataset_name;

#[derive(Debug, Deserialize)]
pub struct CloneDatapointsPathParams {
    pub dataset_name: String,
}

#[derive(Debug, Deserialize)]
pub struct CloneDatapointsRequest {
    pub datapoint_ids: Vec<Uuid>,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct CloneDatapointsResponse {
    pub datapoint_ids: Vec<Option<Uuid>>, // None for missing source datapoints
}

/// The handler for the POST `/internal/datasets/{dataset_name}/datapoints/clone` endpoint.
/// This endpoint clones datapoints to a target dataset, preserving all fields except id and dataset_name.
#[tracing::instrument(name = "clone_datapoints_handler", skip(app_state))]
pub async fn clone_datapoints_handler(
    State(app_state): AppState,
    Path(path_params): Path<CloneDatapointsPathParams>,
    StructuredJson(request): StructuredJson<CloneDatapointsRequest>,
) -> Result<Json<CloneDatapointsResponse>, Error> {
    validate_dataset_name(&path_params.dataset_name)?;

    let new_ids = app_state
        .clickhouse_connection_info
        .clone_datapoints(&path_params.dataset_name, &request.datapoint_ids)
        .await?;

    Ok(Json(CloneDatapointsResponse {
        datapoint_ids: new_ids,
    }))
}
