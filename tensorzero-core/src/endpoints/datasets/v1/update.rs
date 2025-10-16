use std::collections::{HashMap, HashSet};

use axum::extract::{Path, State};
use axum::Json;
use chrono::Utc;
use serde::Deserialize;
use serde_json;
use tracing::instrument;
use uuid::Uuid;

use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::datasets::DatasetQueries;
use crate::endpoints::datasets::{
    validate_dataset_name, ChatInferenceDatapoint, Datapoint, JsonInferenceDatapoint,
};
use crate::endpoints::feedback::{
    validate_parse_demonstration, DemonstrationOutput, DynamicDemonstrationInfo,
};
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfig;
use crate::inference::types::stored_input::StoredInput;
use crate::inference::types::{FetchContext, Input, JsonInferenceOutput};
use crate::tool::{ToolCallConfig, ToolCallConfigDatabaseInsert};
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

use super::types::{
    UpdateChatDatapointRequest, UpdateDatapointRequest, UpdateDatapointsRequest,
    UpdateDatapointsResponse, UpdateJsonDatapointRequest,
};

#[derive(Debug, Deserialize)]
pub struct UpdateDatapointsPathParams {
    pub dataset_name: String,
}

#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "datasets.v1.update_datapoints", skip(app_state, request))]
pub async fn update_datapoints_handler(
    State(app_state): AppState,
    Path(path_params): Path<UpdateDatapointsPathParams>,
    StructuredJson(request): StructuredJson<UpdateDatapointsRequest>,
) -> Result<Json<UpdateDatapointsResponse>, Error> {
    validate_dataset_name(&path_params.dataset_name)?;

    if request.datapoints.is_empty() {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "At least one datapoint must be provided".to_string(),
        }));
    }

    let mut seen_ids = HashSet::new();
    for datapoint in &request.datapoints {
        if !seen_ids.insert(datapoint.id()) {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: format!("Duplicate datapoint id provided: {}", datapoint.id()),
            }));
        }
    }

    let fetch_context = FetchContext {
        client: &app_state.http_client,
        object_store_info: &app_state.config.object_store_info,
    };

    // Fetch all datapoints in a single batch query
    let datapoint_ids: Vec<Uuid> = request
        .datapoints
        .iter()
        .map(UpdateDatapointRequest::id)
        .collect();
    let datapoints_vec = app_state
        .clickhouse_connection_info
        .get_datapoints(&path_params.dataset_name, &datapoint_ids, false)
        .await?;

    // Build a HashMap for easy lookup
    let mut datapoints_map: HashMap<Uuid, Datapoint> =
        datapoints_vec.into_iter().map(|dp| (dp.id(), dp)).collect();

    let mut chat_rows: Vec<ChatInferenceDatapoint> = Vec::new();
    let mut json_rows: Vec<JsonInferenceDatapoint> = Vec::new();
    let mut new_ids: Vec<Uuid> = Vec::with_capacity(request.datapoints.len());

    for update in request.datapoints {
        let datapoint_id = update.id();
        let existing = datapoints_map.remove(&datapoint_id).ok_or_else(|| {
            Error::new(ErrorDetails::DatapointNotFound {
                dataset_name: path_params.dataset_name.clone(),
                datapoint_id,
            })
        })?;

        match (update, existing) {
            (UpdateDatapointRequest::Chat(update), Datapoint::Chat(existing)) => {
                let prepared = prepare_chat_update(
                    &app_state,
                    &fetch_context,
                    &path_params.dataset_name,
                    update,
                    existing,
                )
                .await?;
                chat_rows.extend([prepared.stale, prepared.updated]);
                new_ids.push(prepared.new_id);
            }
            (UpdateDatapointRequest::Json(update), Datapoint::Json(existing)) => {
                let prepared = prepare_json_update(
                    &app_state,
                    &fetch_context,
                    &path_params.dataset_name,
                    update,
                    existing,
                )
                .await?;
                json_rows.extend([prepared.stale, prepared.updated]);
                new_ids.push(prepared.new_id);
            }
            (UpdateDatapointRequest::Chat(_), Datapoint::Json(_)) => {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: format!(
                        "Datapoint {datapoint_id} is a JSON datapoint but a chat update was provided"
                    ),
                }));
            }
            (UpdateDatapointRequest::Json(_), Datapoint::Chat(_)) => {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: format!(
                        "Datapoint {datapoint_id} is a chat datapoint but a JSON update was provided"
                    ),
                }));
            }
        }
    }

    if !chat_rows.is_empty() {
        write_chat_rows(&app_state.clickhouse_connection_info, &chat_rows).await?;
    }
    if !json_rows.is_empty() {
        write_json_rows(&app_state.clickhouse_connection_info, &json_rows).await?;
    }

    Ok(Json(UpdateDatapointsResponse { ids: new_ids }))
}

struct PreparedChatUpdate {
    stale: ChatInferenceDatapoint,
    updated: ChatInferenceDatapoint,
    new_id: Uuid,
}

async fn prepare_chat_update(
    app_state: &AppStateData,
    fetch_context: &FetchContext<'_>,
    dataset_name: &str,
    update: UpdateChatDatapointRequest,
    existing: ChatInferenceDatapoint,
) -> Result<PreparedChatUpdate, Error> {
    if existing.dataset_name != dataset_name {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!(
                "Datapoint {} belongs to dataset '{}' instead of '{}'",
                existing.id, existing.dataset_name, dataset_name
            ),
        }));
    }

    let function_config = app_state.config.get_function(&existing.function_name)?;

    let FunctionConfig::Chat(_) = &**function_config else {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!(
                "Function '{}' is not configured as a chat function",
                existing.function_name
            ),
        }));
    };

    let UpdateChatDatapointRequest {
        id: _,
        input,
        output,
        tool_params,
        tags,
        is_deleted,
        name,
    } = update;

    let maybe_new_input =
        resolve_input_if_provided(input, fetch_context, function_config.as_ref()).await?;
    let final_input = maybe_new_input.unwrap_or_else(|| existing.input.clone());

    let final_tool_params = match tool_params {
        None => existing.tool_params.clone(),
        Some(None) => None,
        Some(Some(params)) => Some(params),
    };

    let final_tags = match tags {
        None => existing.tags.clone(),
        Some(None) => None,
        Some(Some(tags)) => Some(tags),
    };

    let final_is_deleted = is_deleted.unwrap_or(existing.is_deleted);
    let final_name = match name {
        None => existing.name.clone(),
        Some(value) => value,
    };

    let tool_call_config: ToolCallConfig = final_tool_params
        .clone()
        .map(ToolCallConfigDatabaseInsert::into)
        .unwrap_or_default();

    let final_output = match output {
        None => existing.output.clone(),
        Some(None) => None,
        Some(Some(value)) => {
            let validated_output = validate_parse_demonstration(
                function_config.as_ref(),
                &value,
                DynamicDemonstrationInfo::Chat(tool_call_config.clone()),
            )
            .await?;
            let DemonstrationOutput::Chat(content) = validated_output else {
                return Err(Error::new(ErrorDetails::InternalError {
                    message: "Expected chat output after validation".to_string(),
                }));
            };
            Some(content)
        }
    };

    let timestamp = Utc::now().to_string();
    let new_id = Uuid::now_v7();

    let mut stale_row = existing.clone();
    stale_row.staled_at = Some(timestamp.clone());
    stale_row.updated_at = timestamp.clone();

    let updated_row = ChatInferenceDatapoint {
        dataset_name: dataset_name.to_string(),
        function_name: existing.function_name.clone(),
        id: new_id,
        episode_id: None,
        input: final_input,
        output: final_output,
        tool_params: final_tool_params,
        tags: final_tags,
        auxiliary: String::new(),
        is_deleted: final_is_deleted,
        is_custom: true,
        source_inference_id: None,
        staled_at: None,
        updated_at: timestamp,
        name: final_name,
    };

    Ok(PreparedChatUpdate {
        stale: stale_row,
        updated: updated_row,
        new_id,
    })
}

struct PreparedJsonUpdate {
    stale: JsonInferenceDatapoint,
    updated: JsonInferenceDatapoint,
    new_id: Uuid,
}

async fn prepare_json_update(
    app_state: &AppStateData,
    fetch_context: &FetchContext<'_>,
    dataset_name: &str,
    update: UpdateJsonDatapointRequest,
    existing: JsonInferenceDatapoint,
) -> Result<PreparedJsonUpdate, Error> {
    if existing.dataset_name != dataset_name {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!(
                "Datapoint {} belongs to dataset '{}' instead of '{}'",
                existing.id, existing.dataset_name, dataset_name
            ),
        }));
    }

    let function_config = app_state.config.get_function(&existing.function_name)?;

    let FunctionConfig::Json(_) = &**function_config else {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!(
                "Function '{}' is not configured as a JSON function",
                existing.function_name
            ),
        }));
    };

    let UpdateJsonDatapointRequest {
        id: _,
        input,
        output,
        output_schema,
        tags,
        is_deleted,
        name,
    } = update;

    let maybe_new_input =
        resolve_input_if_provided(input, fetch_context, function_config.as_ref()).await?;
    let final_input = maybe_new_input.unwrap_or_else(|| existing.input.clone());

    let final_tags = match tags {
        None => existing.tags.clone(),
        Some(None) => None,
        Some(Some(tags)) => Some(tags),
    };

    let final_is_deleted = is_deleted.unwrap_or(existing.is_deleted);
    let final_name = match name {
        None => existing.name.clone(),
        Some(value) => value,
    };

    let final_output_schema = match output_schema {
        None => existing.output_schema.clone(),
        Some(schema) => {
            if schema.is_null() {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: "output_schema cannot be null".to_string(),
                }));
            }
            schema
        }
    };

    let final_output = match output {
        None => existing.output.clone(),
        Some(None) => None,
        Some(Some(value)) => {
            let validated_output = validate_parse_demonstration(
                function_config.as_ref(),
                &value,
                DynamicDemonstrationInfo::Json(final_output_schema.clone()),
            )
            .await?;
            let DemonstrationOutput::Json(json_value) = validated_output else {
                return Err(Error::new(ErrorDetails::InternalError {
                    message: "Expected JSON output after validation".to_string(),
                }));
            };
            let json_output: JsonInferenceOutput =
                serde_json::from_value(json_value).map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!("Failed to deserialize JSON output: {e}"),
                    })
                })?;
            Some(json_output)
        }
    };

    let timestamp = Utc::now().to_string();
    let new_id = Uuid::now_v7();

    let mut stale_row = existing.clone();
    stale_row.staled_at = Some(timestamp.clone());
    stale_row.updated_at = timestamp.clone();

    let updated_row = JsonInferenceDatapoint {
        dataset_name: dataset_name.to_string(),
        function_name: existing.function_name.clone(),
        id: new_id,
        episode_id: None,
        input: final_input,
        output: final_output,
        output_schema: final_output_schema,
        tags: final_tags,
        auxiliary: String::new(),
        is_deleted: final_is_deleted,
        is_custom: true,
        source_inference_id: None,
        staled_at: None,
        updated_at: timestamp,
        name: final_name,
    };

    Ok(PreparedJsonUpdate {
        stale: stale_row,
        updated: updated_row,
        new_id,
    })
}

async fn resolve_input_if_provided(
    input: Option<Input>,
    fetch_context: &FetchContext<'_>,
    function_config: &FunctionConfig,
) -> Result<Option<StoredInput>, Error> {
    match input {
        None => Ok(None),
        Some(input) => {
            function_config.validate_input(&input)?;
            let resolved_input = input
                .into_lazy_resolved_input(FetchContext {
                    client: fetch_context.client,
                    object_store_info: fetch_context.object_store_info,
                })?
                .resolve()
                .await?;
            Ok(Some(resolved_input.into_stored_input()))
        }
    }
}

async fn write_chat_rows(
    clickhouse: &ClickHouseConnectionInfo,
    rows: &[ChatInferenceDatapoint],
) -> Result<(), Error> {
    let written = super::super::legacy::put_chat_datapoints(clickhouse, rows).await?;
    if written != rows.len() as u64 {
        return Err(Error::new(ErrorDetails::InternalError {
            message: format!(
                "Expected to write {} chat datapoint rows but wrote {}",
                rows.len(),
                written
            ),
        }));
    }
    Ok(())
}

async fn write_json_rows(
    clickhouse: &ClickHouseConnectionInfo,
    rows: &[JsonInferenceDatapoint],
) -> Result<(), Error> {
    let written = super::super::legacy::put_json_datapoints(clickhouse, rows).await?;
    if written != rows.len() as u64 {
        return Err(Error::new(ErrorDetails::InternalError {
            message: format!(
                "Expected to write {} JSON datapoint rows but wrote {}",
                rows.len(),
                written
            ),
        }));
    }
    Ok(())
}
