use axum::extract::{Path, State};
use axum::Json;
use tracing::instrument;
use uuid::Uuid;

use crate::config::Config;
use crate::db::datasets::{
    ChatInferenceDatapointInsert, DatapointInsert, DatasetQueries, JsonInferenceDatapointInsert,
};
use crate::endpoints::datasets::validate_dataset_name;
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfig;
use crate::inference::types::{FetchContext, JsonInferenceOutput};
use crate::jsonschema_util::StaticJSONSchema;
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

use super::types::{
    CreateChatDatapointRequest, CreateDatapointRequest, CreateDatapointsRequest,
    CreateDatapointsResponse, CreateJsonDatapointRequest,
};

/// Handler for the POST `/v1/datasets/{dataset_id}/datapoints` endpoint.
/// Creates manual datapoints in a dataset.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "datasets.v1.create_datapoints", skip(app_state, request))]
pub async fn create_datapoints_handler(
    State(app_state): AppState,
    Path(dataset_name): Path<String>,
    StructuredJson(request): StructuredJson<CreateDatapointsRequest>,
) -> Result<Json<CreateDatapointsResponse>, Error> {
    let response = create_datapoints(&app_state, &dataset_name, request).await?;
    Ok(Json(response))
}

/// Business logic for creating datapoints manually in a dataset.
/// This function validates the request, converts inputs, validates schemas,
/// and inserts the new datapoints into ClickHouse.
///
/// Returns an error if there are no datapoints, or if validation fails.
async fn create_datapoints(
    app_state: &AppStateData,
    dataset_name: &str,
    request: CreateDatapointsRequest,
) -> Result<CreateDatapointsResponse, Error> {
    validate_dataset_name(dataset_name)?;

    if request.datapoints.is_empty() {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "At least one datapoint must be provided".to_string(),
        }));
    }

    let fetch_context = FetchContext {
        client: &app_state.http_client,
        object_store_info: &app_state.config.object_store_info,
    };

    // Build datapoint inserts
    let mut datapoints_to_insert: Vec<DatapointInsert> =
        Vec::with_capacity(request.datapoints.len());
    let mut ids: Vec<Uuid> = Vec::with_capacity(request.datapoints.len());

    for datapoint_request in request.datapoints {
        match datapoint_request {
            CreateDatapointRequest::Chat(chat_request) => {
                let (insert, id) =
                    prepare_chat_create(&app_state.config, &fetch_context, dataset_name, chat_request)
                        .await?;
                datapoints_to_insert.push(DatapointInsert::Chat(insert));
                ids.push(id);
            }
            CreateDatapointRequest::Json(json_request) => {
                let (insert, id) =
                    prepare_json_create(&app_state.config, &fetch_context, dataset_name, json_request)
                        .await?;
                datapoints_to_insert.push(DatapointInsert::Json(insert));
                ids.push(id);
            }
        }
    }

    // Insert all datapoints
    app_state
        .clickhouse_connection_info
        .insert_datapoints(&datapoints_to_insert)
        .await?;

    Ok(CreateDatapointsResponse { ids })
}

async fn prepare_chat_create(
    config: &Config,
    fetch_context: &FetchContext<'_>,
    dataset_name: &str,
    request: CreateChatDatapointRequest,
) -> Result<(ChatInferenceDatapointInsert, Uuid), Error> {
    // Validate function exists and is a chat function
    let function_config = config.get_function(&request.function_name)?;
    let FunctionConfig::Chat(_) = &**function_config else {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!(
                "Function '{}' is not configured as a chat function",
                request.function_name
            ),
        }));
    };

    // Validate and convert input
    function_config.validate_input(&request.input)?;
    let stored_input = request
        .input
        .into_lazy_resolved_input(FetchContext {
            client: fetch_context.client,
            object_store_info: fetch_context.object_store_info,
        })?
        .into_stored_input(fetch_context.object_store_info)
        .await?;

    let id = Uuid::now_v7();
    let name = request.metadata.as_ref().and_then(|m| m.name.clone()).flatten();

    let insert = ChatInferenceDatapointInsert {
        dataset_name: dataset_name.to_string(),
        function_name: request.function_name,
        name,
        id,
        episode_id: request.episode_id,
        input: stored_input,
        output: request.output,
        tool_params: request.tool_params,
        tags: request.tags,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    };

    Ok((insert, id))
}

async fn prepare_json_create(
    config: &Config,
    fetch_context: &FetchContext<'_>,
    dataset_name: &str,
    request: CreateJsonDatapointRequest,
) -> Result<(JsonInferenceDatapointInsert, Uuid), Error> {
    // Validate function exists and is a JSON function
    let function_config = config.get_function(&request.function_name)?;
    let FunctionConfig::Json(_) = &**function_config else {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!(
                "Function '{}' is not configured as a JSON function",
                request.function_name
            ),
        }));
    };

    // Validate and convert input
    function_config.validate_input(&request.input)?;
    let stored_input = request
        .input
        .into_lazy_resolved_input(FetchContext {
            client: fetch_context.client,
            object_store_info: fetch_context.object_store_info,
        })?
        .into_stored_input(fetch_context.object_store_info)
        .await?;

    // Validate output against schema if provided and convert to JsonInferenceOutput
    let output = if let Some(output_value) = request.output {
        let schema = StaticJSONSchema::from_value(request.output_schema.clone()).map_err(|e| {
            Error::new(ErrorDetails::InvalidRequest {
                message: format!("Invalid output schema: {}", e),
            })
        })?;

        schema.validate(&output_value).map_err(|e| {
            Error::new(ErrorDetails::InvalidRequest {
                message: format!("Output does not match schema: {}", e),
            })
        })?;

        Some(JsonInferenceOutput {
            raw: None,
            parsed: Some(output_value),
        })
    } else {
        None
    };

    let id = Uuid::now_v7();
    let name = request.metadata.as_ref().and_then(|m| m.name.clone()).flatten();

    let insert = JsonInferenceDatapointInsert {
        dataset_name: dataset_name.to_string(),
        function_name: request.function_name,
        name,
        id,
        episode_id: request.episode_id,
        input: stored_input,
        output,
        output_schema: request.output_schema,
        tags: request.tags,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    };

    Ok((insert, id))
}
