use std::collections::{HashMap, HashSet};

use axum::extract::{Path, State};
use axum::Json;
use chrono::Utc;
use serde::Deserialize;
use serde_json;
use tracing::instrument;
use uuid::Uuid;

use crate::db::datasets::{
    ChatInferenceDatapointInsert, DatapointInsert, DatasetQueries, JsonInferenceDatapointInsert,
};
use crate::endpoints::datasets::{
    validate_dataset_name, ChatInferenceDatapoint, Datapoint, JsonInferenceDatapoint,
    CLICKHOUSE_DATETIME_FORMAT,
};
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfig;
use crate::inference::types::stored_input::StoredInput;
use crate::inference::types::{FetchContext, Input, JsonInferenceOutput};
use crate::jsonschema_util::StaticJSONSchema;
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

use super::types::{
    UpdateChatDatapointRequest, UpdateDatapointRequest, UpdateDatapointsRequest,
    UpdateDatapointsResponse, UpdateJsonDatapointRequest,
};

impl UpdateDatapointRequest {
    pub fn id(&self) -> Uuid {
        match self {
            UpdateDatapointRequest::Chat(chat) => chat.id,
            UpdateDatapointRequest::Json(json) => json.id,
        }
    }
}

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
    let response = update_datapoints(&app_state, &path_params.dataset_name, request).await?;
    Ok(Json(response))
}

/// Business logic for updating datapoints in a dataset.
/// This function validates the request, fetches existing datapoints, prepares updates,
/// and inserts the updated datapoints into ClickHouse.
async fn update_datapoints(
    app_state: &AppStateData,
    dataset_name: &str,
    request: UpdateDatapointsRequest,
) -> Result<UpdateDatapointsResponse, Error> {
    validate_dataset_name(dataset_name)?;

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
        .get_datapoints(dataset_name, &datapoint_ids, false)
        .await?;

    // Build a HashMap to construct new DatapointInserts
    let mut datapoints_map: HashMap<Uuid, Datapoint> =
        datapoints_vec.into_iter().map(|dp| (dp.id(), dp)).collect();

    // Each update will produce two DatapointInserts: one stale and one updated
    let mut datapoints: Vec<DatapointInsert> = Vec::with_capacity(request.datapoints.len() * 2);
    let mut new_ids: Vec<Uuid> = Vec::with_capacity(request.datapoints.len());

    // Create a timestamp for all the staled_at fields in the query.
    let now_timestamp = Utc::now().format(CLICKHOUSE_DATETIME_FORMAT).to_string();

    for update in request.datapoints {
        let datapoint_id = update.id();
        let existing = datapoints_map.remove(&datapoint_id).ok_or_else(|| {
            Error::new(ErrorDetails::DatapointNotFound {
                dataset_name: dataset_name.to_string(),
                datapoint_id,
            })
        })?;

        match (update, existing) {
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
            (UpdateDatapointRequest::Chat(update), Datapoint::Chat(existing)) => {
                let prepared = prepare_chat_update(
                    app_state,
                    &fetch_context,
                    dataset_name,
                    update,
                    existing,
                    &now_timestamp,
                )
                .await?;
                datapoints.extend([prepared.stale, prepared.updated]);
                new_ids.push(prepared.new_id);
            }
            (UpdateDatapointRequest::Json(update), Datapoint::Json(existing)) => {
                let prepared = prepare_json_update(
                    app_state,
                    &fetch_context,
                    dataset_name,
                    update,
                    existing,
                    &now_timestamp,
                )
                .await?;
                datapoints.extend([prepared.stale, prepared.updated]);
                new_ids.push(prepared.new_id);
            }
        }
    }

    app_state
        .clickhouse_connection_info
        .insert_datapoints(&datapoints)
        .await?;
    Ok(UpdateDatapointsResponse { ids: new_ids })
}

struct PreparedUpdate {
    stale: DatapointInsert,
    updated: DatapointInsert,
    new_id: Uuid,
}

async fn prepare_chat_update(
    app_state: &AppStateData,
    fetch_context: &FetchContext<'_>,
    dataset_name: &str,
    update: UpdateChatDatapointRequest,
    existing_datapoint: ChatInferenceDatapoint,
    now_timestamp: &str,
) -> Result<PreparedUpdate, Error> {
    if existing_datapoint.dataset_name != dataset_name {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!(
                "Datapoint {} belongs to dataset '{}' instead of '{}'",
                existing_datapoint.id, existing_datapoint.dataset_name, dataset_name
            ),
        }));
    }

    // If provided, convert the provided input into a StoredInput.
    let function_config = app_state
        .config
        .get_function(&existing_datapoint.function_name)?;
    let FunctionConfig::Chat(_) = &**function_config else {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!(
                "Function '{}' is not configured as a chat function",
                existing_datapoint.function_name
            ),
        }));
    };

    let updated_datapoint_id = Uuid::now_v7();

    // Update old datapoint as staled, and create new datapoint.
    let mut staled_existing_datapoint: ChatInferenceDatapointInsert =
        existing_datapoint.clone().into();
    staled_existing_datapoint.staled_at = Some(now_timestamp.to_owned());

    // Update the datapoint with new data
    let mut updated_datapoint: ChatInferenceDatapointInsert = existing_datapoint.into();
    updated_datapoint.id = updated_datapoint_id;

    let maybe_new_input =
        convert_input_to_stored_input(update.input, fetch_context, function_config.as_ref())
            .await?;
    if let Some(new_input) = maybe_new_input {
        updated_datapoint.input = new_input;
    }

    if let Some(new_output) = update.output {
        updated_datapoint.output = Some(new_output);
    }
    if let Some(new_tool_params) = update.tool_params {
        updated_datapoint.tool_params = new_tool_params;
    }
    if let Some(new_tags) = update.tags {
        updated_datapoint.tags = Some(new_tags);
    }
    if let Some(new_metadata) = update.metadata {
        if let Some(new_name) = new_metadata.name {
            updated_datapoint.name = new_name;
        }
    }

    Ok(PreparedUpdate {
        new_id: updated_datapoint_id,
        stale: DatapointInsert::Chat(staled_existing_datapoint),
        updated: DatapointInsert::Chat(updated_datapoint),
    })
}

async fn prepare_json_update(
    app_state: &AppStateData,
    fetch_context: &FetchContext<'_>,
    dataset_name: &str,
    update: UpdateJsonDatapointRequest,
    existing_datapoint: JsonInferenceDatapoint,
    now_timestamp: &str,
) -> Result<PreparedUpdate, Error> {
    if existing_datapoint.dataset_name != dataset_name {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!(
                "Datapoint {} belongs to dataset '{}' instead of '{}'",
                existing_datapoint.id, existing_datapoint.dataset_name, dataset_name
            ),
        }));
    }

    // If provided, convert the provided input into a StoredInput.
    let function_config = app_state
        .config
        .get_function(&existing_datapoint.function_name)?;
    let FunctionConfig::Json(_) = &**function_config else {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!(
                "Function '{}' is not configured as a JSON function",
                existing_datapoint.function_name
            ),
        }));
    };

    // Grab a copy of IDs for logging.
    let existing_datapoint_id = existing_datapoint.id;
    let updated_datapoint_id = Uuid::now_v7();

    // Update old datapoint as staled, and create new datapoint.
    let mut staled_existing_datapoint: JsonInferenceDatapointInsert =
        existing_datapoint.clone().into();
    staled_existing_datapoint.staled_at = Some(now_timestamp.to_owned());

    // Update the datapoint with new data
    let mut updated_datapoint: JsonInferenceDatapointInsert = existing_datapoint.into();
    updated_datapoint.id = updated_datapoint_id;

    let maybe_new_input =
        convert_input_to_stored_input(update.input, fetch_context, function_config.as_ref())
            .await?;
    if let Some(new_input) = maybe_new_input {
        updated_datapoint.input = new_input;
    }

    if let Some(new_output_schema) = update.output_schema {
        updated_datapoint.output_schema = new_output_schema;
    }
    if let Some(new_output) = update.output {
        updated_datapoint.output = match new_output {
            None => None,
            Some(value) => {
                // Validate the output with schema before saving.
                StaticJSONSchema::from_value(updated_datapoint.output_schema.clone())?
            .validate(&value)
            .map_err(|e| {
                Error::new(ErrorDetails::InvalidRequest {
                    message: format!(
                        "Provided output for datapoint {existing_datapoint_id} does not match function output schema: {e}",
                    ),
                })
            })?;

                Some(JsonInferenceOutput {
                    raw: Some(serde_json::to_string(&value).map_err(|e| {
                        Error::new(ErrorDetails::Serialization {
                            message: format!(
                                "Failed to serialize provided output for datapoint {existing_datapoint_id}: {e}",
                            )
                        })
                    })?),
                    parsed: Some(value),
                })
            }
        }
    }

    if let Some(new_tags) = update.tags {
        updated_datapoint.tags = Some(new_tags);
    }

    if let Some(new_metadata) = update.metadata {
        if let Some(new_name) = new_metadata.name {
            updated_datapoint.name = new_name;
        }
    }

    Ok(PreparedUpdate {
        new_id: updated_datapoint_id,
        stale: DatapointInsert::Json(staled_existing_datapoint),
        updated: DatapointInsert::Json(updated_datapoint),
    })
}

async fn convert_input_to_stored_input(
    input: Option<Input>,
    fetch_context: &FetchContext<'_>,
    function_config: &FunctionConfig,
) -> Result<Option<StoredInput>, Error> {
    match input {
        None => Ok(None),
        Some(input) => {
            function_config.validate_input(&input)?;

            // If the input file is already in ObjectStorage format, do not resolve; and directly skip into StoredInput.
            // Otherwise, if the file needs to be fetched (because it's new), fetch it and convert to StoredInput.
            // This all happens behind the scene in `into_stored_input()`.
            let stored_input = input
                .into_lazy_resolved_input(FetchContext {
                    client: fetch_context.client,
                    object_store_info: fetch_context.object_store_info,
                })?
                // This call may trigger requests to write newly-provided files to object storage.
                //
                // TODO(shuyangli): consider refactoring file writing logic so it's hard to forget making these calls.
                // Should we put it into `into_stored_file`?
                .into_stored_input(fetch_context.object_store_info)
                .await?;
            Ok(Some(stored_input))
        }
    }
}

#[cfg(test)]
mod tests {}
