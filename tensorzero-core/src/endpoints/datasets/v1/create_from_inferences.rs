use axum::extract::{Path, State};
use axum::Json;
use std::collections::HashMap;
use tracing::instrument;
use uuid::Uuid;

use crate::config::Config;
use crate::db::clickhouse::query_builder::{OrderBy, OrderByTerm, OrderDirection};
use crate::db::clickhouse::{ClickHouseConnectionInfo, ClickhouseFormat};
use crate::db::datasets::{
    ChatInferenceDatapointInsert, DatapointInsert, DatasetQueries, JsonInferenceDatapointInsert,
};
use crate::db::inferences::{InferenceOutputSource, InferenceQueries, ListInferencesParams};
use crate::endpoints::datasets::v1::types::CreateDatapointsFromInferenceOutputSource;
use crate::endpoints::datasets::validate_dataset_name;
use crate::error::Error;
use crate::stored_inference::StoredInference;
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

use super::types::{
    CreateDatapointResult, CreateDatapointsFromInferenceRequest,
    CreateDatapointsFromInferenceRequestParams, CreateDatapointsFromInferenceResponse,
};

/// Handler for the POST `/v1/datasets/{dataset_id}/from_inferences` endpoint.
/// Creates datapoints from inferences based on either specific inference IDs or an inference query.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "datasets.v1.create_from_inferences", skip(app_state, request))]
pub async fn create_from_inferences_handler(
    State(app_state): AppState,
    Path(dataset_name): Path<String>,
    StructuredJson(request): StructuredJson<CreateDatapointsFromInferenceRequest>,
) -> Result<Json<CreateDatapointsFromInferenceResponse>, Error> {
    let response = create_from_inferences(
        &app_state.config,
        &app_state.clickhouse_connection_info,
        dataset_name,
        request,
    )
    .await?;

    Ok(Json(response))
}

async fn create_from_inferences(
    config: &Config,
    clickhouse: &ClickHouseConnectionInfo,
    dataset_name: String,
    request: CreateDatapointsFromInferenceRequest,
) -> Result<CreateDatapointsFromInferenceResponse, Error> {
    validate_dataset_name(&dataset_name)?;

    let output_source = match request.output_source {
        Some(CreateDatapointsFromInferenceOutputSource::None) => {
            // If we are not including any output in the datapoints, we use Inference for the query to
            // avoid doing a join with the DemonstrationFeedback table. Then, we will drop it when constructing the datapoints.
            InferenceOutputSource::Inference
        }
        Some(CreateDatapointsFromInferenceOutputSource::Inference) => {
            InferenceOutputSource::Inference
        }
        Some(CreateDatapointsFromInferenceOutputSource::Demonstration) => {
            InferenceOutputSource::Demonstration
        }
        None => InferenceOutputSource::Inference,
    };

    let inferences = match request.params {
        CreateDatapointsFromInferenceRequestParams::InferenceIds { inference_ids } => {
            get_inferences_by_ids(config, clickhouse, &inference_ids, output_source).await?
        }
        CreateDatapointsFromInferenceRequestParams::InferenceQuery {
            function_name,
            variant_name,
            filters,
        } => {
            query_inferences(
                config,
                clickhouse,
                &function_name,
                variant_name.as_deref(),
                filters.as_ref(),
                output_source,
            )
            .await?
        }
    };

    // Check for existing datapoints with the same source_inference_id
    let inference_ids: Vec<Uuid> = inferences
        .iter()
        .map(|inf| match inf {
            StoredInference::Chat(chat) => chat.inference_id,
            StoredInference::Json(json) => json.inference_id,
        })
        .collect();

    let existing_datapoints =
        get_existing_datapoints_by_inference_ids(clickhouse, &inference_ids).await?;

    // Convert inferences to datapoints
    let mut results = Vec::new();
    let mut datapoints_to_insert = Vec::new();

    for inference in inferences {
        let inference_id = match &inference {
            StoredInference::Chat(chat) => chat.inference_id,
            StoredInference::Json(json) => json.inference_id,
        };

        // Check if datapoint already exists
        if existing_datapoints.contains(&inference_id) {
            results.push(CreateDatapointResult::Error {
                inference_id,
                error: "Datapoint with this source_inference_id already exists".to_string(),
            });
            continue;
        }

        // Try to convert the inference to a datapoint
        match convert_inference_to_datapoint(&dataset_name, inference, &request.output_source).await
        {
            Ok((datapoint_id, datapoint_insert)) => {
                datapoints_to_insert.push(datapoint_insert);
                results.push(CreateDatapointResult::Success {
                    id: datapoint_id,
                    inference_id,
                });
            }
            Err(e) => {
                results.push(CreateDatapointResult::Error {
                    inference_id,
                    error: e.to_string(),
                });
            }
        }
    }

    // Batch insert all successful datapoints
    if !datapoints_to_insert.is_empty() {
        clickhouse.insert_datapoints(&datapoints_to_insert).await?;
    }

    Ok(CreateDatapointsFromInferenceResponse {
        datapoints: results,
    })
}

/// Get inferences by their IDs.
/// Returns results for all found IDs, silently skips missing ones.
async fn get_inferences_by_ids(
    config: &Config,
    clickhouse: &ClickHouseConnectionInfo,
    inference_ids: &[Uuid],
    output_source: InferenceOutputSource,
) -> Result<Vec<StoredInference>, Error> {
    if inference_ids.is_empty() {
        return Ok(Vec::new());
    }

    // Use list_inferences with the IDs filter to query both tables efficiently
    let params = ListInferencesParams {
        function_name: None, // Will query both ChatInference and JsonInference with UNION ALL
        variant_name: None,
        ids: Some(inference_ids),
        episode_id: None,
        filters: None,
        output_source,
        limit: None,
        offset: None,
        order_by: None,
        format: ClickhouseFormat::JsonEachRow,
    };

    clickhouse.list_inferences(config, &params).await
}

/// Query inferences using the list_inferences functionality.
async fn query_inferences(
    config: &Config,
    clickhouse: &ClickHouseConnectionInfo,
    function_name: &str,
    variant_name: Option<&str>,
    filters: Option<&crate::db::clickhouse::query_builder::InferenceFilter>,
    output_source: InferenceOutputSource,
) -> Result<Vec<StoredInference>, Error> {
    let params = ListInferencesParams {
        function_name: Some(function_name),
        variant_name,
        ids: None,
        episode_id: None,
        filters,
        output_source,
        limit: None, // No limit - get all matching inferences
        offset: None,
        order_by: Some(&[OrderBy {
            term: OrderByTerm::Timestamp,
            direction: OrderDirection::Desc,
        }]),
        format: ClickhouseFormat::JsonEachRow,
    };

    clickhouse.list_inferences(config, &params).await
}

/// Get existing datapoints that have the given inference IDs as their source_inference_id.
/// Returns a set of inference IDs that already have datapoints.
async fn get_existing_datapoints_by_inference_ids(
    clickhouse: &ClickHouseConnectionInfo,
    inference_ids: &[Uuid],
) -> Result<std::collections::HashSet<Uuid>, Error> {
    if inference_ids.is_empty() {
        return Ok(std::collections::HashSet::new());
    }

    // Query both Chat and Json datapoint tables for existing source_inference_ids
    let ids_array: Vec<String> = inference_ids.iter().map(|id| format!("'{id}'")).collect();
    let ids_list = ids_array.join(", ");

    let chat_query = format!(
        r"
        SELECT DISTINCT source_inference_id
        FROM ChatInferenceDatapoint
        WHERE source_inference_id IN ({ids_list})
        AND is_deleted = 0
        FORMAT JSONEachRow;"
    );

    let json_query = format!(
        r"
        SELECT DISTINCT source_inference_id
        FROM JsonInferenceDatapoint
        WHERE source_inference_id IN ({ids_list})
        AND is_deleted = 0
        FORMAT JSONEachRow;"
    );

    let mut existing_ids = std::collections::HashSet::new();

    // Query chat datapoints
    if let Ok(response) = clickhouse
        .run_query_synchronous(chat_query, &HashMap::new())
        .await
    {
        for line in response.response.lines() {
            if let Ok(result) = serde_json::from_str::<serde_json::Value>(line) {
                if let Some(id_str) = result.get("source_inference_id").and_then(|v| v.as_str()) {
                    if let Ok(id) = Uuid::parse_str(id_str) {
                        existing_ids.insert(id);
                    }
                }
            }
        }
    }

    // Query json datapoints
    if let Ok(response) = clickhouse
        .run_query_synchronous(json_query, &HashMap::new())
        .await
    {
        for line in response.response.lines() {
            if let Ok(result) = serde_json::from_str::<serde_json::Value>(line) {
                if let Some(id_str) = result.get("source_inference_id").and_then(|v| v.as_str()) {
                    if let Ok(id) = Uuid::parse_str(id_str) {
                        existing_ids.insert(id);
                    }
                }
            }
        }
    }

    Ok(existing_ids)
}

/// Convert a StoredInference to a DatapointInsert.
/// Returns the datapoint ID and the insert struct.
/// The output_source_override parameter allows overriding to None even if the inference has an output.
async fn convert_inference_to_datapoint(
    dataset_name: &str,
    inference: StoredInference,
    output_source_override: &Option<CreateDatapointsFromInferenceOutputSource>,
) -> Result<(Uuid, DatapointInsert), Error> {
    let datapoint_id = Uuid::now_v7();

    match inference {
        StoredInference::Json(json_inference) => {
            // If output_source is explicitly None, set output to None regardless of what's in the inference
            let output = match output_source_override {
                Some(CreateDatapointsFromInferenceOutputSource::None) => None,
                _ => Some(json_inference.output.clone()),
            };

            let datapoint = JsonInferenceDatapointInsert {
                dataset_name: dataset_name.to_string(),
                function_name: json_inference.function_name.clone(),
                name: None,
                id: datapoint_id,
                episode_id: Some(json_inference.episode_id),
                input: json_inference.input.clone(),
                output,
                output_schema: json_inference.output_schema.clone(),
                tags: Some(json_inference.tags.clone()),
                auxiliary: String::new(),
                staled_at: None,
                source_inference_id: Some(json_inference.inference_id),
                is_custom: false,
            };

            Ok((datapoint_id, DatapointInsert::Json(datapoint)))
        }
        StoredInference::Chat(chat_inference) => {
            // If output_source is explicitly None, set output to None regardless of what's in the inference
            let output = match output_source_override {
                Some(CreateDatapointsFromInferenceOutputSource::None) => None,
                _ => Some(chat_inference.output.clone()),
            };

            let datapoint = ChatInferenceDatapointInsert {
                dataset_name: dataset_name.to_string(),
                function_name: chat_inference.function_name.clone(),
                name: None,
                id: datapoint_id,
                episode_id: Some(chat_inference.episode_id),
                input: chat_inference.input.clone(),
                output,
                tool_params: Some(chat_inference.tool_params.clone()),
                tags: Some(chat_inference.tags.clone()),
                auxiliary: String::new(),
                staled_at: None,
                source_inference_id: Some(chat_inference.inference_id),
                is_custom: false,
            };

            Ok((datapoint_id, DatapointInsert::Chat(datapoint)))
        }
    }
}
