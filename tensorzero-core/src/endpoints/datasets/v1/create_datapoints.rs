use axum::extract::{Path, State};
use axum::Json;
use tracing::instrument;
use uuid::Uuid;

use crate::config::Config;
use crate::db::datasets::{
    ChatInferenceDatapointInsert, DatapointInsert, DatasetQueries, JsonInferenceDatapointInsert,
};
use crate::endpoints::datasets::validate_dataset_name;
use crate::endpoints::feedback::{
    validate_parse_demonstration, DemonstrationOutput, DynamicDemonstrationInfo,
};
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfig;
use crate::inference::types::{FetchContext, JsonInferenceOutput};
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
pub async fn create_datapoints(
    app_state: &AppStateData,
    dataset_name: &str,
    request: CreateDatapointsRequest,
) -> Result<CreateDatapointsResponse, Error> {
    create_datapoints_impl(
        &app_state.config,
        &app_state.http_client,
        &app_state.clickhouse_connection_info,
        dataset_name,
        request,
    )
    .await
}

/// Internal implementation that can be called from legacy code.
pub async fn create_datapoints_impl(
    config: &Config,
    http_client: &crate::http::TensorzeroHttpClient,
    clickhouse: &crate::db::clickhouse::ClickHouseConnectionInfo,
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
        client: http_client,
        object_store_info: &config.object_store_info,
    };

    // Build datapoint inserts
    let mut datapoints_to_insert: Vec<DatapointInsert> =
        Vec::with_capacity(request.datapoints.len());
    let mut ids: Vec<Uuid> = Vec::with_capacity(request.datapoints.len());

    for datapoint_request in request.datapoints {
        match datapoint_request {
            CreateDatapointRequest::Chat(chat_request) => {
                let (insert, id) =
                    prepare_chat_create(config, &fetch_context, dataset_name, chat_request).await?;
                datapoints_to_insert.push(DatapointInsert::Chat(insert));
                ids.push(id);
            }
            CreateDatapointRequest::Json(json_request) => {
                let (insert, id) =
                    prepare_json_create(config, &fetch_context, dataset_name, json_request).await?;
                datapoints_to_insert.push(DatapointInsert::Json(insert));
                ids.push(id);
            }
        }
    }

    // Insert all datapoints
    clickhouse.insert_datapoints(&datapoints_to_insert).await?;

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
        // This call may trigger requests to write newly-provided files to object storage.
        .into_stored_input(fetch_context.object_store_info)
        .await?;

    // Prepare the tool config
    let tool_config =
        function_config.prepare_tool_config(request.dynamic_tool_params, &config.tools)?;
    let dynamic_demonstration_info =
        DynamicDemonstrationInfo::Chat(tool_config.clone().unwrap_or_default());

    // Validate and parse output if provided
    let output = if let Some(output_value) = request.output {
        let validated_output = validate_parse_demonstration(
            &function_config,
            &output_value,
            dynamic_demonstration_info,
        )
        .await?;

        let DemonstrationOutput::Chat(output) = validated_output else {
            return Err(Error::new(ErrorDetails::InternalError {
                message: "Expected chat output from validate_parse_demonstration".to_string(),
            }));
        };

        Some(output)
    } else {
        None
    };

    let id = Uuid::now_v7();

    let insert = ChatInferenceDatapointInsert {
        dataset_name: dataset_name.to_string(),
        function_name: request.function_name,
        name: request.name,
        id,
        episode_id: request.episode_id,
        input: stored_input,
        output,
        tool_params: tool_config.as_ref().map(|x| x.clone().into()),
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
    let FunctionConfig::Json(json_function_config) = &**function_config else {
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
        // This call may trigger requests to write newly-provided files to object storage.
        .into_stored_input(fetch_context.object_store_info)
        .await?;

    // Determine the output schema (use provided or default to function's schema)
    let output_schema = request
        .output_schema
        .unwrap_or_else(|| json_function_config.output_schema.value.clone());
    let dynamic_demonstration_info = DynamicDemonstrationInfo::Json(output_schema.clone());

    // Validate and parse output if provided
    let output = if let Some(output_value) = request.output {
        let validated_output = validate_parse_demonstration(
            &function_config,
            &output_value,
            dynamic_demonstration_info,
        )
        .await?;

        let DemonstrationOutput::Json(output) = validated_output else {
            return Err(Error::new(ErrorDetails::InternalError {
                message: "Expected JSON output from validate_parse_demonstration".to_string(),
            }));
        };

        Some(JsonInferenceOutput {
            raw: output
                .get("raw")
                .and_then(|v| v.as_str().map(str::to_string)),
            parsed: output.get("parsed").cloned(),
        })
    } else {
        None
    };

    let id: Uuid = Uuid::now_v7();

    let insert = JsonInferenceDatapointInsert {
        dataset_name: dataset_name.to_string(),
        function_name: request.function_name,
        name: request.name,
        id,
        episode_id: request.episode_id,
        input: stored_input,
        output,
        output_schema,
        tags: request.tags,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    };

    Ok((insert, id))
}
