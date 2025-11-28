use axum::extract::{Path, State};
use axum::Json;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::db::clickhouse::ClickHouseConnectionInfo;
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

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
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

    let new_ids = clone_datapoints(
        &path_params.dataset_name,
        &request.datapoint_ids,
        &app_state.clickhouse_connection_info,
    )
    .await?;

    Ok(Json(CloneDatapointsResponse {
        datapoint_ids: new_ids,
    }))
}

pub async fn clone_datapoints(
    target_dataset_name: &str,
    datapoint_ids: &[Uuid],
    clickhouse: &ClickHouseConnectionInfo,
) -> Result<Vec<Option<Uuid>>, Error> {
    if datapoint_ids.is_empty() {
        return Ok(vec![]);
    }

    // Generate all mappings from source to target IDs
    let mappings: Vec<(Uuid, Uuid)> = datapoint_ids
        .iter()
        .map(|id| (*id, Uuid::now_v7()))
        .collect();

    // Round trip 1: Parallel INSERTs with CTE + EXCEPT pattern
    let chat_clone_query = r"
        INSERT INTO ChatInferenceDatapoint
        WITH source AS (
            SELECT ChatInferenceDatapoint.*, mapping.new_id
            FROM ChatInferenceDatapoint FINAL
            INNER JOIN (
                SELECT
                    tupleElement(pair, 1) as old_id,
                    tupleElement(pair, 2) as new_id
                FROM (
                    SELECT arrayJoin({mappings: Array(Tuple(UUID, UUID))}) as pair
                )
            ) AS mapping ON ChatInferenceDatapoint.id = mapping.old_id
            WHERE ChatInferenceDatapoint.staled_at IS NULL
        )
        SELECT * EXCEPT(new_id) REPLACE(
            new_id AS id,
            {target_dataset_name: String} AS dataset_name,
            now64() AS updated_at
        )
        FROM source
    ";

    let json_clone_query = r"
        INSERT INTO JsonInferenceDatapoint
        WITH source AS (
            SELECT JsonInferenceDatapoint.*, mapping.new_id
            FROM JsonInferenceDatapoint FINAL
            INNER JOIN (
                SELECT
                    tupleElement(pair, 1) as old_id,
                    tupleElement(pair, 2) as new_id
                FROM (
                    SELECT arrayJoin({mappings: Array(Tuple(UUID, UUID))}) as pair
                )
            ) AS mapping ON JsonInferenceDatapoint.id = mapping.old_id
            WHERE JsonInferenceDatapoint.staled_at IS NULL
        )
        SELECT * EXCEPT(new_id) REPLACE(
            new_id AS id,
            {target_dataset_name: String} AS dataset_name,
            now64() AS updated_at
        )
        FROM source
    ";

    let mappings_str = format!(
        "[{}]",
        mappings
            .iter()
            .map(|(old, new)| format!("('{old}', '{new}')"))
            .collect::<Vec<_>>()
            .join(",")
    );
    let insert_params = HashMap::from([
        ("target_dataset_name", target_dataset_name),
        ("mappings", mappings_str.as_str()),
    ]);

    let chat_future =
        clickhouse.run_query_synchronous(chat_clone_query.to_string(), &insert_params);
    let json_future =
        clickhouse.run_query_synchronous(json_clone_query.to_string(), &insert_params);

    let (chat_result, json_result) = tokio::join!(chat_future, json_future);
    chat_result?;
    json_result?;

    // Round trip 2: Verify which `new_ids` were actually created (in case some source datapoints don't exist)
    let new_ids_str = format!(
        "[{}]",
        mappings
            .iter()
            .map(|(_, new)| format!("'{new}'"))
            .collect::<Vec<_>>()
            .join(",")
    );

    let verify_query = r"
        SELECT id FROM (
            SELECT id FROM ChatInferenceDatapoint FINAL
            WHERE id IN ({new_ids: Array(UUID)}) AND staled_at IS NULL
            UNION ALL
            SELECT id FROM JsonInferenceDatapoint FINAL
            WHERE id IN ({new_ids: Array(UUID)}) AND staled_at IS NULL
        )
    ";
    let verify_params = HashMap::from([("new_ids", new_ids_str.as_str())]);
    let verify_result = clickhouse
        .run_query_synchronous(verify_query.to_string(), &verify_params)
        .await?;

    let created_ids: std::collections::HashSet<Uuid> = verify_result
        .response
        .lines()
        .filter_map(|line| Uuid::parse_str(line.trim()).ok())
        .collect();

    // Map results based on which `new_ids` were created
    let results: Vec<Option<Uuid>> = mappings
        .iter()
        .map(|(source_id, new_id)| {
            if created_ids.contains(new_id) {
                Some(*new_id)
            } else {
                tracing::warn!("Failed to clone datapoint (likely does not exist): {source_id}");
                None
            }
        })
        .collect();

    Ok(results)
}
