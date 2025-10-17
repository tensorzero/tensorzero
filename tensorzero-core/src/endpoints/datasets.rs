use axum::extract::{Path, Query, State};
use axum::Json;
use chrono::Utc;
use futures::future;
use futures::try_join;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::IntoPyObjectExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, future::Future, pin::Pin};
use tracing::instrument;
use uuid::Uuid;

use crate::db::clickhouse::{ClickHouseConnectionInfo, ExternalDataInfo, TableName};
use crate::db::datasets::{DatasetQueries, GetDatapointParams};
use crate::function::{FunctionConfig, FunctionConfigType};
use crate::http::TensorzeroHttpClient;
use crate::inference::types::stored_input::StoredInput;
use crate::inference::types::{
    ContentBlockChatOutput, FetchContext, Input, JsonInferenceOutput,
    TaggedInferenceDatabaseInsert, Text,
};
use crate::stored_inference::{SimpleStoredSampleInfo, StoredOutput, StoredSample};
use crate::{
    config::Config,
    error::{Error, ErrorDetails},
    serde_util::{deserialize_optional_string_or_parsed_json, deserialize_string_or_parsed_json},
    tool::{DynamicToolParams, ToolCallConfigDatabaseInsert},
    utils::gateway::{AppState, StructuredJson},
    utils::uuid::validate_tensorzero_uuid,
};

#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::{
    content_block_chat_output_to_python, serialize_to_dict, uuid_to_python,
};

#[cfg(debug_assertions)]
use crate::utils::gateway::AppStateData;

use super::feedback::{
    validate_parse_demonstration, DemonstrationOutput, DynamicDemonstrationInfo,
};

pub const CLICKHOUSE_DATETIME_FORMAT: &str = "%Y-%m-%d %H:%M:%S%.6f";

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputKind {
    Inherit,
    Demonstration,
    None,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct Demonstration {
    #[expect(dead_code)]
    id: Uuid,
    #[expect(dead_code)]
    inference_id: Uuid,
    value: String,
}

async fn query_demonstration(
    clickhouse: &ClickHouseConnectionInfo,
    inference_id: Uuid,
    page_size: u32,
) -> Result<Demonstration, Error> {
    let result = clickhouse
        .run_query_synchronous(
            r"
        SELECT
          id,
          inference_id,
          value,
          formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
        FROM DemonstrationFeedbackByInferenceId
        WHERE inference_id = {inference_id:String}
        ORDER BY toUInt128(id) DESC
        LIMIT {page_size:UInt32}
        FORMAT JSONEachRow;"
                .to_string(),
            &HashMap::from([
                ("inference_id", inference_id.to_string().as_str()),
                ("page_size", page_size.to_string().as_str()),
            ]),
        )
        .await?;
    if result.response.is_empty() {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!("No demonstration found for inference `{inference_id}`"),
        }));
    }
    let demonstration: Demonstration = serde_json::from_str(&result.response).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to deserialize demonstration ClickHouse response: {e}"),
        })
    })?;
    Ok(demonstration)
}

async fn query_inference_for_datapoint(
    config: &Config,
    clickhouse: &ClickHouseConnectionInfo,
    inference_id: Uuid,
    function_name: &str,
    variant_name: &str,
    episode_id: Uuid,
) -> Result<TaggedInferenceDatabaseInsert, Error> {
    let function_type = config.get_function(function_name)?.config_type();
    let result = match function_type {
        FunctionConfigType::Chat => {
            clickhouse
                .run_query_synchronous(
                    r"
                SELECT * EXCEPT(timestamp),
                'chat' AS function_type,
                formatDateTime(timestamp, '%Y-%m-%dT%H:%i:%SZ') AS timestamp
                FROM ChatInference
                    WHERE function_name = {function_name:String}
                    AND variant_name = {variant_name:String}
                    AND episode_id = {episode_id:String}
                    AND id = {inference_id:String}
                LIMIT 1
                FORMAT JSONEachRow;"
                        .to_string(),
                    &HashMap::from([
                        ("function_name", function_name),
                        ("variant_name", variant_name),
                        ("episode_id", episode_id.to_string().as_str()),
                        ("inference_id", inference_id.to_string().as_str()),
                    ]),
                )
                .await?
        }
        FunctionConfigType::Json => {
            clickhouse
                .run_query_synchronous(
                    r"
                SELECT * EXCEPT(timestamp),
                'json' AS function_type,
                formatDateTime(timestamp, '%Y-%m-%dT%H:%i:%SZ') AS timestamp
                FROM JsonInference
                WHERE function_name = {function_name:String}
                AND variant_name = {variant_name:String}
                AND episode_id = {episode_id:String}
                AND id = {inference_id:String}
                LIMIT 1
                FORMAT JSONEachRow;"
                        .to_string(),
                    &HashMap::from([
                        ("function_name", function_name),
                        ("variant_name", variant_name),
                        ("episode_id", episode_id.to_string().as_str()),
                        ("inference_id", inference_id.to_string().as_str()),
                    ]),
                )
                .await?
        }
    };

    if result.response.is_empty() {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!("Inference `{inference_id}` not found"),
        }));
    }

    let inference_data: TaggedInferenceDatabaseInsert = serde_json::from_str(&result.response)
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to deserialize inference data: {e}"),
            })
        })?;
    Ok(inference_data)
}

/// This function inserts a new datapoint into `ChatInferenceDatapoint`/`JsonInferenceDatapoint`/
/// based on an existing inference (specified by `inference_id`).
///
/// The inference is mostly copied as-is, except for the 'output' field.
/// Based on the 'output' parameter, the output is copied, ignored, or fetched from a demonstration.
/// Datapoints that are created this way are not marked as custom datapoints.
async fn insert_from_existing(
    config: &Config,
    clickhouse: &ClickHouseConnectionInfo,
    path_params: InsertPathParams,
    existing: &ExistingInferenceInfo,
) -> Result<Uuid, Error> {
    let ExistingInferenceInfo {
        function_name,
        variant_name,
        episode_id,
        inference_id,
        output,
    } = existing;
    let inference_data = query_inference_for_datapoint(
        config,
        clickhouse,
        *inference_id,
        function_name,
        variant_name,
        *episode_id,
    )
    .await?;
    let datapoint_id = Uuid::now_v7();

    match inference_data {
        TaggedInferenceDatabaseInsert::Json(inference) => {
            let output = match output {
                OutputKind::Inherit => Some(inference.output),
                OutputKind::Demonstration => {
                    let demonstration = query_demonstration(clickhouse, *inference_id, 1).await?;
                    Some(serde_json::from_str(&demonstration.value).map_err(|e| {
                        Error::new(ErrorDetails::Serialization {
                            message: format!(
                                "Failed to deserialize JSON demonstration output: {e}"
                            ),
                        })
                    })?)
                }
                OutputKind::None => None,
            };
            // TODO(#3957): the `put_json_datapoints` call should really take a `JsonInferenceDatapointInsert`. We'll fix it separately.
            let datapoint = JsonInferenceDatapoint {
                dataset_name: path_params.dataset_name,
                function_name: inference.function_name,
                name: None,
                id: datapoint_id,
                episode_id: Some(inference.episode_id),
                input: inference.input,
                output,
                output_schema: inference.output_schema,
                tags: Some(inference.tags),
                auxiliary: "{}".to_string(),
                is_deleted: false,
                is_custom: false,
                source_inference_id: Some(*inference_id),
                staled_at: None,

                // Ignored during insert.
                updated_at: Utc::now().to_string(),
            };
            let rows_written = put_json_datapoints(clickhouse, &[datapoint]).await?;
            if rows_written == 0 {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: "Datapoint with this source_inference_id already exists".to_string(),
                }));
            }
        }
        TaggedInferenceDatabaseInsert::Chat(inference) => {
            let output = match output {
                OutputKind::Inherit => Some(inference.output),
                OutputKind::Demonstration => {
                    let demonstration = query_demonstration(clickhouse, *inference_id, 1).await?;
                    Some(serde_json::from_str(&demonstration.value).map_err(|e| {
                        Error::new(ErrorDetails::InvalidRequest {
                            message: format!(
                                "Failed to deserialize chat demonstration output: {e}"
                            ),
                        })
                    })?)
                }
                OutputKind::None => None,
            };
            // TODO(#3957): the `put_chat_datapoints` call should really take a `ChatInferenceDatapointInsert`. We'll fix it separately.
            let datapoint = ChatInferenceDatapoint {
                dataset_name: path_params.dataset_name,
                function_name: inference.function_name,
                name: None,
                id: datapoint_id,
                episode_id: Some(inference.episode_id),
                input: inference.input,
                output,
                tool_params: inference.tool_params,
                tags: Some(inference.tags),
                auxiliary: "{}".to_string(),
                is_deleted: false,
                is_custom: false,
                source_inference_id: Some(*inference_id),
                staled_at: None,

                // Ignored during insert.
                updated_at: Utc::now().to_string(),
            };
            let rows_written = put_chat_datapoints(clickhouse, &[datapoint]).await?;
            if rows_written == 0 {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: "Datapoint with this source_inference_id already exists".to_string(),
                }));
            }
        }
    }
    Ok(datapoint_id)
}

#[derive(Deserialize)]
struct WithFunctionName {
    function_name: String,
}

// The handler for the POST `/internal/datasets/:dataset/datapoints` endpoint.
/// This inserts a new datapoint into `ChatInferenceDatapoint`/`JsonInferenceDatapoint`/
/// based on an existing inference (specified by `inference_id`).
///
/// The inference is mostly copied as-is, except for the 'output' field.
/// Based on the 'output' parameter, the output is copied, ignored, or fetched from a demonstration.
#[instrument(name = "insert_datapoint", skip_all)]
pub async fn insert_from_existing_datapoint_handler(
    State(app_state): AppState,
    Path(path_params): Path<InsertPathParams>,
    StructuredJson(existing_inference_info): StructuredJson<ExistingInferenceInfo>,
) -> Result<Json<InsertDatapointResponse>, Error> {
    validate_dataset_name(&path_params.dataset_name)?;
    let datapoint_id = insert_from_existing(
        &app_state.config,
        &app_state.clickhouse_connection_info,
        path_params,
        &existing_inference_info,
    )
    .await?;
    Ok(Json(InsertDatapointResponse { id: datapoint_id }))
}

/// The handler for the PUT `/internal/datasets/:dataset/datapoints/:id"` endpoint.
/// This writes a datapoint with the given id, overwriting any existing datapoint
/// with the same id.
///
/// The input and output are validated against the function schema
/// (retrieved from the `function_name` argument in the body).
#[instrument(name = "update_datapoint", skip_all)]
pub async fn update_datapoint_handler(
    State(app_state): AppState,
    Path(path_params): Path<UpdatePathParams>,
    // This is deserialized as either a `UpdateChatInferenceDatapointRequest` or `UpdateJsonInferenceDatapointRequest`,
    // based on the type of the function looked up from the `function_name` key.
    StructuredJson(params): StructuredJson<serde_json::Value>,
) -> Result<Json<InsertDatapointResponse>, Error> {
    validate_tensorzero_uuid(path_params.datapoint_id, "Datapoint")?;
    validate_dataset_name(&path_params.dataset_name)?;
    let fetch_context = FetchContext {
        client: &app_state.http_client,
        object_store_info: &app_state.config.object_store_info,
    };
    let function_data: WithFunctionName = serde_json::from_value(params.clone()).map_err(|e| {
        Error::new(ErrorDetails::InvalidRequest {
            message: format!("Failed to deserialize `function_name`: {e}"),
        })
    })?;
    let function_config = app_state
        .config
        .get_function(&function_data.function_name)?;

    // NOTE(shuyangli): Prior to 2025-10-02, this endpoint was not used to update datapoints in-place (they used
    // to always get a new ID, marked as custom, and lose their episode association). Now this can be used to update metadata like name.
    match **function_config {
        FunctionConfig::Chat(_) => {
            let chat: UpdateChatInferenceDatapointRequest = serde_json::from_value(params)
                .map_err(|e| {
                    Error::new(ErrorDetails::InvalidRequest {
                        message: format!("Failed to deserialize chat datapoint: {e}"),
                    })
                })?;

            let resolved_input = chat
                .input
                .clone()
                .into_lazy_resolved_input(fetch_context)?
                .resolve()
                .await?;
            function_config.validate_input(&chat.input)?;
            // If there are no tool params in the UpdateChatInferenceDatapointRequest, we use the default tool params (empty tools).
            // This is consistent with how they are serialized at inference time.
            let dynamic_demonstration_info = DynamicDemonstrationInfo::Chat(
                chat.tool_params
                    .clone()
                    .map(ToolCallConfigDatabaseInsert::into)
                    .unwrap_or_default(),
            );

            // Only validate and parse output if it exists
            let output = if let Some(output) = &chat.output {
                let validated_output = validate_parse_demonstration(
                    &function_config,
                    output,
                    dynamic_demonstration_info,
                )
                .await?;

                let DemonstrationOutput::Chat(output) = validated_output else {
                    return Err(Error::new(ErrorDetails::InternalError {
                        message: "Expected chat output from validate_parse_demonstration"
                            .to_string(),
                    }));
                };

                Some(output)
            } else {
                None
            };

            let datapoint = ChatInferenceDatapoint {
                dataset_name: path_params.dataset_name,
                function_name: chat.function_name,
                name: chat.name,
                id: path_params.datapoint_id,
                episode_id: chat.episode_id,
                input: resolved_input.into_stored_input(),
                output,
                tool_params: chat.tool_params,
                tags: chat.tags,
                auxiliary: chat.auxiliary,
                is_deleted: chat.is_deleted,
                is_custom: chat.is_custom,
                source_inference_id: chat.source_inference_id,
                staled_at: chat.staled_at,

                // Ignored during insert.
                updated_at: Utc::now().to_string(),
            };
            let rows_written =
                put_chat_datapoints(&app_state.clickhouse_connection_info, &[datapoint]).await?;
            if rows_written == 0 {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: "Datapoint with this source_inference_id already exists".to_string(),
                }));
            }
        }
        FunctionConfig::Json(_) => {
            let json: UpdateJsonInferenceDatapointRequest = serde_json::from_value(params)
                .map_err(|e| {
                    Error::new(ErrorDetails::InvalidRequest {
                        message: format!("Failed to deserialize JSON datapoint: {e}"),
                    })
                })?;
            let resolved_input = json
                .input
                .clone()
                .into_lazy_resolved_input(fetch_context)?
                .resolve()
                .await?;
            function_config.validate_input(&json.input)?;
            let dynamic_demonstration_info =
                DynamicDemonstrationInfo::Json(json.output_schema.clone());

            // Determine output based on whether json.output is None
            let output = if let Some(output_value) = &json.output {
                let validated_json = validate_parse_demonstration(
                    &function_config,
                    output_value,
                    dynamic_demonstration_info,
                )
                .await?;

                let DemonstrationOutput::Json(json_out) = validated_json else {
                    return Err(Error::new(ErrorDetails::InternalError {
                        message: "Expected JSON output from validate_parse_demonstration"
                            .to_string(),
                    }));
                };

                let json_out: JsonInferenceOutput =
                    serde_json::from_value(json_out).map_err(|e| {
                        Error::new(ErrorDetails::Serialization {
                            message: format!("Failed to deserialize validated JSON output: {e}"),
                        })
                    })?;

                Some(json_out)
            } else {
                None
            };

            let datapoint = JsonInferenceDatapoint {
                dataset_name: path_params.dataset_name,
                function_name: json.function_name,
                name: json.name,
                id: path_params.datapoint_id,
                episode_id: json.episode_id,
                input: resolved_input.into_stored_input(),
                output,
                output_schema: json.output_schema,
                tags: json.tags,
                auxiliary: json.auxiliary,
                is_deleted: json.is_deleted,
                is_custom: json.is_custom,
                source_inference_id: json.source_inference_id,
                staled_at: json.staled_at,

                // Ignored during insert.
                updated_at: Utc::now().to_string(),
            };
            let rows_written =
                put_json_datapoints(&app_state.clickhouse_connection_info, &[datapoint]).await?;
            if rows_written == 0 {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: "Datapoint with this source_inference_id already exists".to_string(),
                }));
            }
        }
    }
    Ok(Json(InsertDatapointResponse {
        id: path_params.datapoint_id,
    }))
}

/// Note: This type should be a Vec<Enum<ChatDatapointInsert, JsonDatapointInsert>>,
/// however, since the required fields don't distinguish these two types serde will fail to disambiguate them
/// as it deserializes.
///
/// We can disambiguate them by checking the config for the `function_name` that
/// the datapoint is for and then deserializing it as the correct type.
///
/// For the OpenAPI spec we will have to manually create the type for this.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InsertDatapointParams {
    pub datapoints: Vec<Value>,
}

#[derive(Debug, Deserialize)]
pub struct InsertDatapointPathParams {
    pub dataset_name: String,
}

// The handler for the POST `/datasets/{dataset_name}/datapoints` endpoint.
/// This inserts a new datapoint into `ChatInferenceDatapoint`/`JsonInferenceDatapoint`/
#[tracing::instrument(name = "create_datapoints_handler", skip(app_state, params))]
pub async fn create_datapoints_handler(
    State(app_state): AppState,
    Path(path_params): Path<InsertDatapointPathParams>,
    StructuredJson(params): StructuredJson<InsertDatapointParams>,
) -> Result<Json<Vec<Uuid>>, Error> {
    let datapoint_ids = insert_datapoint(
        path_params.dataset_name,
        params,
        &app_state.config,
        &app_state.http_client,
        &app_state.clickhouse_connection_info,
    )
    .await?;
    Ok(Json(datapoint_ids))
}

/// DEPRECATED: Use the POST `/datasets/{dataset_name}/datapoints` endpoint instead.
#[tracing::instrument(name = "bulk_insert_datapoints_handler", skip(app_state, params))]
pub async fn bulk_insert_datapoints_handler(
    State(app_state): AppState,
    Path(path_params): Path<InsertDatapointPathParams>,
    StructuredJson(params): StructuredJson<InsertDatapointParams>,
) -> Result<Json<Vec<Uuid>>, Error> {
    tracing::warn!(
        "DEPRECATION WARNING: The `/datasets/{dataset_name}/datapoints/bulk` endpoint is deprecated. Please use `/datasets/{dataset_name}/datapoints` instead.",
        dataset_name = path_params.dataset_name
    );
    create_datapoints_handler(State(app_state), Path(path_params), StructuredJson(params)).await
}

pub async fn insert_datapoint(
    dataset_name: String,
    params: InsertDatapointParams,
    config: &Config,
    http_client: &TensorzeroHttpClient,
    clickhouse: &ClickHouseConnectionInfo,
) -> Result<Vec<Uuid>, Error> {
    validate_dataset_name(&dataset_name)?;
    let mut chat_datapoints = Vec::with_capacity(params.datapoints.len());
    let mut json_datapoints = Vec::with_capacity(params.datapoints.len());
    let fetch_context = FetchContext {
        client: http_client,
        object_store_info: &config.object_store_info,
    };
    let mut datapoint_ids = Vec::with_capacity(params.datapoints.len());
    for (i, datapoint) in params.datapoints.into_iter().enumerate() {
        let function_name = datapoint
            .get("function_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                Error::new(ErrorDetails::InvalidRequest {
                    message: format!("Expected function name for datapoint {i}"),
                })
            })?;
        let function_config = config.get_function(function_name).map_err(|e| {
            Error::new(ErrorDetails::InvalidRequest {
                message: format!("Failed to get function config for datapoint {i}: {e}"),
            })
        })?;
        match &**function_config {
            FunctionConfig::Chat(_) => {
                let chat: ChatDatapointInsert = serde_json::from_value(datapoint).map_err(|e| {
                    Error::new(ErrorDetails::InvalidRequest {
                        message: format!("Failed to deserialize chat datapoint {i}: {e}"),
                    })
                })?;
                // Validate the input
                function_config.validate_input(&chat.input).map_err(|e| {
                    Error::new(ErrorDetails::InvalidRequest {
                        message: format!("Failed to validate chat input for datapoint {i}: {e}"),
                    })
                })?;
                let resolved_input = chat
                    .input
                    .into_lazy_resolved_input(fetch_context)
                    .map_err(|e| {
                        Error::new(ErrorDetails::InternalError {
                            message: format!(
                                "Failed to lazily resolve chat input for datapoint {i}: {e}"
                            ),
                        })
                    })?
                    .resolve()
                    .await
                    .map_err(|e| {
                        Error::new(ErrorDetails::InternalError {
                            message: format!("Failed to resolve chat input for datapoint {i}: {e}"),
                        })
                    })?;
                // Prepare the tool config
                let tool_config =
                    function_config.prepare_tool_config(chat.dynamic_tool_params, &config.tools)?;
                let dynamic_demonstration_info =
                    DynamicDemonstrationInfo::Chat(tool_config.clone().unwrap_or_default());
                // Validate the output
                let output = if let Some(output) = chat.output {
                    let validated_output = validate_parse_demonstration(
                        &function_config,
                        &serde_json::to_value(output).map_err(|e| {
                            Error::new(ErrorDetails::Serialization {
                                message: format!(
                                    "Failed to serialize chat output for datapoint {i}: {e}"
                                ),
                            })
                        })?,
                        dynamic_demonstration_info,
                    )
                    .await
                    .map_err(|e| {
                        Error::new(ErrorDetails::InvalidRequest {
                            message: format!(
                                "Failed to validate chat output for datapoint {i}: {e}"
                            ),
                        })
                    })?;
                    let DemonstrationOutput::Chat(output) = validated_output else {
                        return Err(Error::new(ErrorDetails::InternalError {
                            message: "Expected chat output from validate_parse_demonstration"
                                .to_string(),
                        }));
                    };
                    Some(output)
                } else {
                    None
                };
                let datapoint_id = Uuid::now_v7();
                datapoint_ids.push(datapoint_id);
                chat_datapoints.push(ChatInferenceDatapoint {
                    dataset_name: dataset_name.clone(),
                    function_name: chat.function_name,
                    name: chat.name,
                    id: datapoint_id,
                    episode_id: None,
                    input: resolved_input.into_stored_input(),
                    output,
                    tool_params: tool_config.as_ref().map(|x| x.clone().into()),
                    tags: chat.tags,
                    auxiliary: String::new(),
                    is_deleted: false,
                    is_custom: true,
                    source_inference_id: None,
                    staled_at: None,
                    // Ignored during insert.
                    updated_at: Utc::now().to_string(),
                });
            }
            FunctionConfig::Json(json_function_config) => {
                let json: JsonDatapointInsert = serde_json::from_value(datapoint).map_err(|e| {
                    Error::new(ErrorDetails::InvalidRequest {
                        message: format!("Failed to deserialize json datapoint {i}: {e}"),
                    })
                })?;
                // Validate the input
                function_config.validate_input(&json.input).map_err(|e| {
                    Error::new(ErrorDetails::InvalidRequest {
                        message: format!("Failed to validate input for datapoint {i}: {e}"),
                    })
                })?;
                let resolved_input = json
                    .input
                    .into_lazy_resolved_input(fetch_context)?
                    .resolve()
                    .await
                    .map_err(|e| {
                        Error::new(ErrorDetails::InternalError {
                            message: format!("Failed to resolve input for datapoint {i}: {e}"),
                        })
                    })?;
                // Validate the outputs against the output schema
                let output_schema = json
                    .output_schema
                    .unwrap_or_else(|| json_function_config.output_schema.value.clone());
                let dynamic_demonstration_info =
                    DynamicDemonstrationInfo::Json(output_schema.clone());
                let output = if let Some(output) = json.output {
                    let validated_output = validate_parse_demonstration(
                        &function_config,
                        &serde_json::to_value(output).map_err(|e| {
                            Error::new(ErrorDetails::Serialization {
                                message: format!(
                                    "Failed to serialize json output for datapoint {i}: {e}"
                                ),
                            })
                        })?,
                        dynamic_demonstration_info,
                    )
                    .await
                    .map_err(|e| {
                        Error::new(ErrorDetails::InvalidRequest {
                            message: format!(
                                "Failed to validate chat output for datapoint {i}: {e}"
                            ),
                        })
                    })?;
                    let DemonstrationOutput::Json(output) = validated_output else {
                        return Err(Error::new(ErrorDetails::InternalError {
                            message: "Expected valid JSON output from validate_parse_demonstration"
                                .to_string(),
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
                let datapoint_id = Uuid::now_v7();
                datapoint_ids.push(datapoint_id);
                let datapoint = JsonInferenceDatapoint {
                    dataset_name: dataset_name.clone(),
                    function_name: json.function_name,
                    name: json.name,
                    id: datapoint_id,
                    episode_id: None,
                    input: resolved_input.into_stored_input(),
                    output,
                    output_schema,
                    tags: json.tags,
                    auxiliary: String::new(),
                    is_deleted: false,
                    is_custom: true,
                    source_inference_id: None,
                    staled_at: None,
                    // Ignored during insert.
                    updated_at: Utc::now().to_string(),
                };
                json_datapoints.push(datapoint);
            }
        }
    }

    #[expect(clippy::type_complexity)]
    let mut futures_vec: Vec<Pin<Box<dyn Future<Output = Result<u64, Error>> + Send>>> = Vec::new();

    if !chat_datapoints.is_empty() {
        futures_vec.push(Box::pin(put_chat_datapoints(clickhouse, &chat_datapoints)));
    }
    if !json_datapoints.is_empty() {
        futures_vec.push(Box::pin(put_json_datapoints(clickhouse, &json_datapoints)));
    }

    // Run all futures concurrently and propagate any error
    future::try_join_all(futures_vec).await?;
    Ok(datapoint_ids)
}

#[derive(Debug, Deserialize)]
pub struct DeleteDatapointPathParams {
    pub dataset_name: String,
    pub datapoint_id: Uuid,
}

/// The handler for the DELETE `/datasets/:dataset_name/datapoints/:datapoint_id` endpoint.
/// This endpoint will stale the datapoint from the dataset (soft delete).
#[tracing::instrument(name = "delete_datapoint_handler", skip(app_state))]
pub async fn delete_datapoint_handler(
    State(app_state): AppState,
    Path(path_params): Path<DeleteDatapointPathParams>,
) -> Result<(), Error> {
    delete_datapoint(
        path_params.dataset_name,
        path_params.datapoint_id,
        &app_state.clickhouse_connection_info,
    )
    .await
}
pub async fn delete_datapoint(
    dataset_name: String,
    datapoint_id: Uuid,
    clickhouse: &ClickHouseConnectionInfo,
) -> Result<(), Error> {
    // Since we don't know whether the datapoint is a chat or json datapoint, we just stale both of these.
    // The INSERT INTO SELECT FROM will just not write anything if the datapoint doesn't exist.
    let json_delete_query = r"
    INSERT INTO JsonInferenceDatapoint
    (dataset_name, function_name, name, id, episode_id, input, output, output_schema,
     tags, auxiliary, is_deleted, is_custom, source_inference_id, updated_at, staled_at)
    SELECT dataset_name, function_name, name, id, episode_id, input, output, output_schema,
           tags, auxiliary, is_deleted, is_custom, source_inference_id, now64(), now64()
    FROM JsonInferenceDatapoint
    WHERE id = {datapoint_id: UUID} AND dataset_name = {dataset_name: String}
";
    let chat_delete_query = r"
    INSERT INTO ChatInferenceDatapoint
    (dataset_name, function_name, name, id, episode_id, input, output, tool_params,
     tags, auxiliary, is_deleted, is_custom, source_inference_id, updated_at, staled_at)
    SELECT dataset_name, function_name, name, id, episode_id, input, output, tool_params,
           tags, auxiliary, is_deleted, is_custom, source_inference_id, now64(), now64()
    FROM ChatInferenceDatapoint
    WHERE id = {datapoint_id: UUID} AND dataset_name = {dataset_name: String}
";
    let datapoint_id = datapoint_id.to_string();
    let json_params = HashMap::from([
        ("datapoint_id", datapoint_id.as_str()),
        ("dataset_name", dataset_name.as_str()),
    ]);

    let chat_params = HashMap::from([
        ("datapoint_id", datapoint_id.as_str()),
        ("dataset_name", dataset_name.as_str()),
    ]);

    let json_future = clickhouse.run_query_synchronous(json_delete_query.to_string(), &json_params);

    let chat_future = clickhouse.run_query_synchronous(chat_delete_query.to_string(), &chat_params);

    let (json_result, chat_result) = tokio::join!(json_future, chat_future);

    json_result?;
    chat_result?;

    Ok(())
}

#[derive(Debug, Deserialize)]
pub struct ListDatapointsQueryParams {
    function_name: Option<String>,
    limit: Option<u32>,
    offset: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct ListDatapointsPathParams {
    dataset_name: String,
}

#[axum::debug_handler(state = AppStateData)]
pub async fn list_datapoints_handler(
    State(app_state): AppState,
    Path(path_params): Path<ListDatapointsPathParams>,
    Query(query_params): Query<ListDatapointsQueryParams>,
) -> Result<Json<Vec<Datapoint>>, Error> {
    list_datapoints(
        path_params.dataset_name,
        &app_state.clickhouse_connection_info,
        query_params.function_name,
        query_params.limit,
        query_params.offset,
    )
    .await
    .map(Json)
}

#[tracing::instrument(name = "list_datapoints", skip(clickhouse))]
pub async fn list_datapoints(
    dataset_name: String,
    clickhouse: &ClickHouseConnectionInfo,
    function_name: Option<String>,
    limit: Option<u32>,
    offset: Option<u32>,
) -> Result<Vec<Datapoint>, Error> {
    let mut query = r"
    WITH dataset as (
        SELECT
            'chat' as type,
            dataset_name,
            function_name,
            name,
            id,
            episode_id,
            input,
            output,
            tool_params,
            '\N' as output_schema, -- for column alignment in UNION ALL
            tags,
            auxiliary,
            source_inference_id,
            is_deleted,
            is_custom,
            staled_at,
            updated_at
        FROM ChatInferenceDatapoint FINAL
        WHERE dataset_name = {dataset_name: String}
        AND staled_at IS NULL"
        .to_string();

    if function_name.is_some() {
        query.push_str(" AND function_name = {function_name: String}");
    }

    query.push_str(
        r"
        UNION ALL
        SELECT
            'json' as type,
            dataset_name,
            function_name,
            name,
            id,
            episode_id,
            input,
            output,
            '\N' as tool_params, -- for column alignment in UNION ALL
            output_schema,
            tags,
            auxiliary,
            source_inference_id,
            is_deleted,
            is_custom,
            staled_at,
            updated_at
        FROM JsonInferenceDatapoint FINAL
        WHERE dataset_name = {dataset_name: String}
        AND staled_at IS NULL",
    );

    if function_name.is_some() {
        query.push_str(" AND function_name = {function_name: String}");
    }

    query.push_str(
        r"
    )
    SELECT * FROM dataset
    ORDER BY id DESC
    LIMIT {limit: UInt32}
    OFFSET {offset: UInt32}
    FORMAT JSONEachRow
    ",
    );
    let limit = limit.unwrap_or(100);
    let offset = offset.unwrap_or(0);
    let limit_str = limit.to_string();
    let offset_str = offset.to_string();

    let mut params = HashMap::from([
        ("dataset_name", dataset_name.as_str()),
        ("limit", limit_str.as_str()),
        ("offset", offset_str.as_str()),
    ]);

    if let Some(ref fn_name) = function_name {
        params.insert("function_name", fn_name.as_str());
    }

    let result = clickhouse
        .run_query_synchronous(query.to_string(), &params)
        .await?;
    if result.response.is_empty() {
        return Ok(vec![]);
    }
    let result_lines = result.response.trim().split("\n").collect::<Vec<&str>>();

    let datapoints: Result<Vec<Datapoint>, _> = result_lines
        .iter()
        .map(|line| serde_json::from_str(line))
        .collect();
    let datapoints = match datapoints {
        Ok(datapoints) => datapoints,
        Err(e) => {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: format!("Failed to deserialize datapoints: {e}"),
            }));
        }
    };

    Ok(datapoints)
}

pub struct BatchDatapointOutputWithSize {
    output: Option<Vec<Option<Value>>>,
    size: usize,
}

impl TryFrom<BatchDatapointOutputWithSize> for Vec<Option<Value>> {
    type Error = Error;

    fn try_from(value: BatchDatapointOutputWithSize) -> Result<Self, Self::Error> {
        let size = value.size;
        if let Some(output) = value.output {
            let output_len = output.len();
            if output_len == value.size {
                Ok(output)
            } else {
                Err(Error::new(ErrorDetails::InvalidRequest {
                    message: format!(
                        "Output size ({output_len}) does not match number of datapoints ({size})",
                    ),
                }))
            }
        } else {
            let mut output = Vec::with_capacity(size);
            for _ in 0..size {
                output.push(None);
            }
            Ok(output)
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct GetDatapointPathParams {
    dataset_name: String,
    datapoint_id: Uuid,
}

#[axum::debug_handler(state = AppStateData)]
pub async fn get_datapoint_handler(
    State(app_state): AppState,
    Path(path_params): Path<GetDatapointPathParams>,
) -> Result<Json<Datapoint>, Error> {
    app_state
        .clickhouse_connection_info
        .get_datapoint(&GetDatapointParams {
            dataset_name: path_params.dataset_name,
            datapoint_id: path_params.datapoint_id,
            allow_stale: None,
        })
        .await
        .map(Json)
}

#[derive(Debug, Deserialize)]
pub struct InsertPathParams {
    pub dataset_name: String,
}

#[derive(Debug, Deserialize)]
pub struct UpdatePathParams {
    pub dataset_name: String,
    pub datapoint_id: Uuid,
}

#[derive(Debug, Deserialize)]
pub struct DeletePathParams {
    pub dataset: String,
    pub function: String,
    pub kind: DatapointKind,
    pub id: Uuid,
}

#[derive(Debug, Deserialize)]
pub struct ExistingInferenceInfo {
    pub output: OutputKind,
    pub inference_id: Uuid,
    pub function_name: String,
    pub variant_name: String,
    pub episode_id: Uuid,
}

#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum DatapointKind {
    Chat,
    Json,
}

impl DatapointKind {
    pub fn table_name(&self) -> TableName {
        match self {
            DatapointKind::Chat => TableName::ChatInferenceDatapoint,
            DatapointKind::Json => TableName::JsonInferenceDatapoint,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct InsertDatapointResponse {
    id: Uuid,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub enum Datapoint {
    Chat(ChatInferenceDatapoint),
    Json(JsonInferenceDatapoint),
}

impl Datapoint {
    pub fn dataset_name(&self) -> &str {
        match self {
            Datapoint::Chat(datapoint) => &datapoint.dataset_name,
            Datapoint::Json(datapoint) => &datapoint.dataset_name,
        }
    }

    pub fn input(&self) -> &StoredInput {
        match self {
            Datapoint::Chat(datapoint) => &datapoint.input,
            Datapoint::Json(datapoint) => &datapoint.input,
        }
    }

    pub fn tool_call_config(&self) -> Option<&ToolCallConfigDatabaseInsert> {
        match self {
            Datapoint::Chat(datapoint) => datapoint.tool_params.as_ref(),
            Datapoint::Json(_datapoint) => None,
        }
    }

    pub fn output_schema(&self) -> Option<&serde_json::Value> {
        match self {
            Datapoint::Chat(_datapoint) => None,
            Datapoint::Json(datapoint) => Some(&datapoint.output_schema),
        }
    }

    pub fn id(&self) -> Uuid {
        match self {
            Datapoint::Chat(datapoint) => datapoint.id,
            Datapoint::Json(datapoint) => datapoint.id,
        }
    }

    pub fn name(&self) -> Option<&str> {
        match self {
            Datapoint::Chat(datapoint) => datapoint.name.as_deref(),
            Datapoint::Json(datapoint) => datapoint.name.as_deref(),
        }
    }
}

/// These input datapoints are used as input types by the `insert_datapoint` endpoint
#[cfg(feature = "pyo3")]
#[pymethods]
impl Datapoint {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }

    #[getter]
    pub fn get_id<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        uuid_to_python(py, self.id())
    }

    #[getter]
    pub fn get_input<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.input().clone().into_bound_py_any(py)
    }

    #[getter]
    pub fn get_output<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self {
            Datapoint::Chat(datapoint) => match &datapoint.output {
                Some(output) => output
                    .iter()
                    .map(|x| content_block_chat_output_to_python(py, x.clone()))
                    .collect::<PyResult<Vec<_>>>()?
                    .into_bound_py_any(py)?,
                None => py.None().into_bound(py),
            },
            Datapoint::Json(datapoint) => datapoint.output.clone().into_bound_py_any(py)?,
        })
    }

    #[getter]
    pub fn get_dataset_name(&self) -> String {
        self.dataset_name().to_string()
    }

    #[getter]
    pub fn get_function_name(&self) -> String {
        self.function_name().to_string()
    }

    #[getter]
    pub fn get_tool_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.tool_call_config() {
            Some(tool_params) => tool_params.clone().into_bound_py_any(py),
            None => Ok(py.None().into_bound(py)),
        }
    }

    #[getter]
    pub fn get_output_schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self.output_schema() {
            Some(output_schema) => serialize_to_dict(py, output_schema)?.into_bound(py),
            None => py.None().into_bound(py),
        })
    }

    #[getter]
    pub fn get_is_custom(&self) -> bool {
        match self {
            Datapoint::Chat(datapoint) => datapoint.is_custom,
            Datapoint::Json(datapoint) => datapoint.is_custom,
        }
    }

    #[getter]
    pub fn get_name(&self) -> Option<String> {
        self.name().map(std::string::ToString::to_string)
    }
}

impl std::fmt::Display for Datapoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

/// These input datapoints are used as input types by the `insert_datapoint` endpoint
/// The distinction here is that they do not include the `dataset_name` field,
/// which is instead specified as a path parameter.
/// We also use Input rather than ResolvedInput because the input is not resolved
/// when created.
/// We also do not allow users to specify the `id` or `episode_id` fields.
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatDatapointInsert {
    pub function_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub input: Input,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
    #[serde(flatten)]
    pub dynamic_tool_params: DynamicToolParams,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct JsonDatapointInsert {
    pub function_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub input: Input,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
    pub output_schema: Option<Value>, // Default to the function's output schema
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<HashMap<String, String>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct ChatInferenceDatapoint {
    pub dataset_name: String,
    pub function_name: String,
    pub id: Uuid,
    pub episode_id: Option<Uuid>,
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub input: StoredInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    #[serde(deserialize_with = "deserialize_optional_string_or_parsed_json")]
    #[cfg_attr(test, ts(optional))]
    pub output: Option<Vec<ContentBlockChatOutput>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    #[serde(deserialize_with = "deserialize_optional_string_or_parsed_json")]
    #[cfg_attr(test, ts(optional))]
    pub tool_params: Option<ToolCallConfigDatabaseInsert>,

    // By default, ts_rs generates { [key in string]?: string } | undefined, which means values are string | undefined which isn't what we want.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    #[cfg_attr(test, ts(type = "Record<string, string>"), ts(optional))]
    pub tags: Option<HashMap<String, String>>,
    #[serde(skip_serializing, default)] // this will become an object
    pub auxiliary: String,
    pub is_deleted: bool,
    #[serde(default)]
    pub is_custom: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    #[cfg_attr(test, ts(optional))]
    pub source_inference_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    #[cfg_attr(test, ts(optional))]
    pub staled_at: Option<String>,
    pub updated_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    #[cfg_attr(test, ts(optional))]
    pub name: Option<String>,
}

impl std::fmt::Display for ChatInferenceDatapoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct JsonInferenceDatapoint {
    pub dataset_name: String,
    pub function_name: String,
    pub id: Uuid,
    pub episode_id: Option<Uuid>,
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub input: StoredInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    #[serde(deserialize_with = "deserialize_optional_string_or_parsed_json")]
    #[cfg_attr(test, ts(optional))]
    pub output: Option<JsonInferenceOutput>,
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub output_schema: serde_json::Value,

    // By default, ts_rs generates { [key in string]?: string } | undefined, which means values are string | undefined which isn't what we want.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    #[cfg_attr(test, ts(type = "Record<string, string>"), ts(optional))]
    pub tags: Option<HashMap<String, String>>,
    #[serde(skip_serializing, default)] // this will become an object
    pub auxiliary: String,
    pub is_deleted: bool,
    #[serde(default)]
    pub is_custom: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    #[cfg_attr(test, ts(optional))]
    pub source_inference_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    #[cfg_attr(test, ts(optional))]
    pub staled_at: Option<String>,
    pub updated_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    #[cfg_attr(test, ts(optional))]
    pub name: Option<String>,
}

impl std::fmt::Display for JsonInferenceDatapoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl StoredSample for Datapoint {
    fn function_name(&self) -> &str {
        match self {
            Datapoint::Chat(datapoint) => &datapoint.function_name,
            Datapoint::Json(datapoint) => &datapoint.function_name,
        }
    }

    fn input(&self) -> &StoredInput {
        match self {
            Datapoint::Chat(datapoint) => &datapoint.input,
            Datapoint::Json(datapoint) => &datapoint.input,
        }
    }

    fn input_mut(&mut self) -> &mut StoredInput {
        match self {
            Datapoint::Chat(datapoint) => &mut datapoint.input,
            Datapoint::Json(datapoint) => &mut datapoint.input,
        }
    }

    fn into_input(self) -> StoredInput {
        match self {
            Datapoint::Chat(datapoint) => datapoint.input,
            Datapoint::Json(datapoint) => datapoint.input,
        }
    }

    fn owned_simple_info(self) -> SimpleStoredSampleInfo {
        match self {
            Datapoint::Chat(datapoint) => SimpleStoredSampleInfo {
                function_name: datapoint.function_name,
                input: datapoint.input,
                output: datapoint.output.clone(),
                stored_output: datapoint.output.map(StoredOutput::Chat),
                dispreferred_outputs: Vec::default(),
                tool_params: datapoint.tool_params,
                output_schema: None,
                episode_id: None,
                inference_id: None,
                tags: datapoint.tags.unwrap_or_default(),
            },
            Datapoint::Json(datapoint) => {
                let stored_output = datapoint.output.clone().map(StoredOutput::Json);
                let output = datapoint.output.map(|output| match output.raw {
                    Some(raw) => vec![ContentBlockChatOutput::Text(Text { text: raw })],
                    None => vec![],
                });
                SimpleStoredSampleInfo {
                    function_name: datapoint.function_name,
                    input: datapoint.input,
                    output,
                    stored_output,
                    dispreferred_outputs: Vec::default(),
                    tool_params: None,
                    output_schema: Some(datapoint.output_schema),
                    episode_id: None,
                    inference_id: None,
                    tags: datapoint.tags.unwrap_or_default(),
                }
            }
        }
    }
}

/// If the input is None then we should return None
/// If the input is Value::Null we should return None
/// If the input is some other Value::* we should return it.
fn deserialize_optional_json_value<'de, D>(
    deserializer: D,
) -> Result<Option<serde_json::Value>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt = Option::<serde_json::Value>::deserialize(deserializer)?;
    match opt {
        None => Ok(None),
        Some(value) => match value {
            serde_json::Value::Null => Ok(None),
            _ => Ok(Some(value)),
        },
    }
}

// UpdateChatInferenceDatapointRequest is used in `update_datapoint_handler` to update an existing ChatInferenceDatapoint
// in place. It's not a partial update - the entire request is written to the database.
//
// TODO(shuyangli): this is currently being manually kept in sync with `ChatInferenceDatapoint`. Can we fix this?
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateChatInferenceDatapointRequest {
    pub function_name: String,
    #[serde(default)]
    pub episode_id: Option<Uuid>,
    pub input: Input,
    #[serde(default)]
    #[serde(deserialize_with = "deserialize_optional_json_value")]
    pub output: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_params: Option<ToolCallConfigDatabaseInsert>,
    #[serde(default)]
    pub tags: Option<HashMap<String, String>>,
    #[serde(default)]
    pub auxiliary: String,
    #[serde(default)]
    pub is_deleted: bool,
    pub is_custom: bool,
    #[serde(default)]
    pub source_inference_id: Option<Uuid>,
    #[serde(default)]
    pub staled_at: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

// UpdateJsonInferenceDatapointRequest is used in `update_datapoint_handler` to update an existing JsonInferenceDatapoint
// in place. It's not a partial update - the entire request is written to the database.
//
// TODO(shuyangli): this is currently being manually kept in sync with `JsonInferenceDatapoint`. Can we fix this?
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateJsonInferenceDatapointRequest {
    pub function_name: String,
    #[serde(default)]
    pub episode_id: Option<Uuid>,
    pub input: Input,
    #[serde(default)]
    #[serde(deserialize_with = "deserialize_optional_json_value")]
    pub output: Option<serde_json::Value>,
    pub output_schema: serde_json::Value,
    #[serde(default)]
    pub tags: Option<HashMap<String, String>>,
    #[serde(skip_serializing, default)] // this will become an object
    pub auxiliary: String,
    #[serde(default)]
    pub is_deleted: bool,
    pub is_custom: bool,
    #[serde(default)]
    pub source_inference_id: Option<Uuid>,
    #[serde(default)]
    pub staled_at: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

pub(crate) fn validate_dataset_name(dataset_name: &str) -> Result<(), Error> {
    if dataset_name == "builder" || dataset_name.starts_with("tensorzero::") {
        Err(Error::new(ErrorDetails::InvalidDatasetName {
            dataset_name: dataset_name.to_string(),
        }))
    } else {
        Ok(())
    }
}

/// Puts a chat datapoint into ClickHouse
/// Returns the number of rows written to ClickHouse
async fn put_chat_datapoints(
    clickhouse: &ClickHouseConnectionInfo,
    datapoints: &[ChatInferenceDatapoint],
) -> Result<u64, Error> {
    let serialized_datapoints = datapoints
        .iter()
        .map(|datapoint| {
            serde_json::to_string(datapoint).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Failed to serialize datapoint: {e}"),
                })
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let query = r"
    INSERT INTO ChatInferenceDatapoint
        (
            dataset_name,
            function_name,
            name,
            id,
            episode_id,
            input,
            output,
            tool_params,
            tags,
            auxiliary,
            is_deleted,
            is_custom,
            source_inference_id
        )
        SELECT
            new_data.dataset_name,
            new_data.function_name,
            new_data.name,
            new_data.id,
            new_data.episode_id,
            new_data.input,
            new_data.output,
            new_data.tool_params,
            new_data.tags,
            new_data.auxiliary,
            new_data.is_deleted,
            new_data.is_custom,
            new_data.source_inference_id
        FROM new_data
        ";

    let external_data = ExternalDataInfo {
        external_data_name: "new_data".to_string(),
        structure: "dataset_name LowCardinality(String), function_name LowCardinality(String), name Nullable(String), id UUID, episode_id Nullable(UUID), input String, output Nullable(String), tool_params String, tags Map(String, String), auxiliary String, is_deleted Bool, is_custom Bool, source_inference_id Nullable(UUID), updated_at String".to_string(),
        format: "JSONEachRow".to_string(),
        data: serialized_datapoints.join("\n"),
    };
    let result = clickhouse
        .run_query_with_external_data(external_data, query.to_string())
        .await?;
    Ok(result.metadata.written_rows)
}

/// Puts a json datapoint into ClickHouse
/// Returns the number of rows written to ClickHouse
async fn put_json_datapoints(
    clickhouse: &ClickHouseConnectionInfo,
    datapoints: &[JsonInferenceDatapoint],
) -> Result<u64, Error> {
    let serialized_datapoints = datapoints
        .iter()
        .map(|datapoint| {
            serde_json::to_string(datapoint).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Failed to serialize datapoint: {e}"),
                })
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let query = r"
        INSERT INTO JsonInferenceDatapoint
        (
            dataset_name,
            function_name,
            name,
            id,
            episode_id,
            input,
            output,
            output_schema,
            tags,
            auxiliary,
            is_deleted,
            is_custom,
            source_inference_id
        )
        SELECT
            new_data.dataset_name,
            new_data.function_name,
            new_data.name,
            new_data.id,
            new_data.episode_id,
            new_data.input,
            new_data.output,
            new_data.output_schema,
            new_data.tags,
            new_data.auxiliary,
            new_data.is_deleted,
            new_data.is_custom,
            new_data.source_inference_id
        FROM new_data
        ";

    let external_data = ExternalDataInfo {
        external_data_name: "new_data".to_string(),
        structure: "dataset_name LowCardinality(String), function_name LowCardinality(String), name Nullable(String), id UUID, episode_id Nullable(UUID), input String, output Nullable(String), output_schema Nullable(String), tags Map(String, String), auxiliary String, is_deleted Bool, is_custom Bool, source_inference_id Nullable(UUID), updated_at String".to_string(),
        format: "JSONEachRow".to_string(),
        data: serialized_datapoints.join("\n"),
    };
    let result = clickhouse
        .run_query_with_external_data(external_data, query.to_string())
        .await?;
    Ok(result.metadata.written_rows)
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct StaleDatasetResponse {
    pub num_staled_datapoints: u64,
}

#[axum::debug_handler(state = AppStateData)]
pub async fn stale_dataset_handler(
    State(app_state): AppState,
    // These are the same as the path params for `list_datapoints_handler`
    Path(path_params): Path<ListDatapointsPathParams>,
) -> Result<Json<StaleDatasetResponse>, Error> {
    let response = stale_dataset(
        &app_state.clickhouse_connection_info,
        &path_params.dataset_name,
    )
    .await?;
    Ok(Json(response))
}

/// Stales all datapoints in a dataset that have not been staled yet.
/// This is a soft deletion, so evaluation runs will still refer to it.
/// Returns the number of datapoints that were staled.
pub async fn stale_dataset(
    clickhouse: &ClickHouseConnectionInfo,
    dataset_name: &str,
) -> Result<StaleDatasetResponse, Error> {
    // NOTE: in the two queries below, we don't alias to staled_at because then we won't select any rows.
    let chat_query = r"
    INSERT INTO ChatInferenceDatapoint
    SELECT
        *
        REPLACE (
            now64() AS updated_at,
            now64() AS staled_at
        )
    FROM ChatInferenceDatapoint FINAL
    WHERE dataset_name = {dataset_name:String}
    AND staled_at IS NULL
    "
    .to_string();

    let json_query = r"
    INSERT INTO JsonInferenceDatapoint
    SELECT
        *
        REPLACE (
            now64() AS updated_at,
            now64() AS staled_at
        )
    FROM JsonInferenceDatapoint FINAL
    WHERE dataset_name = {dataset_name:String}
    AND staled_at IS NULL
    "
    .to_string();
    let query_params = HashMap::from([("dataset_name", dataset_name)]);

    let (chat_result, json_result) = try_join!(
        clickhouse.run_query_synchronous(chat_query, &query_params),
        clickhouse.run_query_synchronous(json_query, &query_params)
    )?;
    Ok(StaleDatasetResponse {
        num_staled_datapoints: chat_result.metadata.written_rows
            + json_result.metadata.written_rows,
    })
}

#[cfg(test)]
mod test {
    use super::*;

    use serde_json::json;

    #[test]
    fn test_synthetic_chat_datapoint_with_none_output() {
        let json_str = r#"{
            "function_name": "test_function",
            "input": {"system": {"assistant_name": "Test"}, "messages": []},
            "output": null,
            "tool_params": null,
            "tags": null,
            "auxiliary": "",
            "is_custom": false
        }"#;

        let datapoint: UpdateChatInferenceDatapointRequest =
            serde_json::from_str(json_str).unwrap();
        assert_eq!(datapoint.function_name, "test_function");
        assert_eq!(datapoint.output, None);
        assert_eq!(datapoint.tool_params, None);
        assert_eq!(datapoint.tags, None);
        assert_eq!(datapoint.auxiliary, "");
        assert!(!datapoint.is_custom);
    }

    #[test]
    fn test_synthetic_chat_datapoint_with_some_output() {
        let json_str = r#"{
            "function_name": "test_function",
            "input": {"system": {"assistant_name": "Test"}, "messages": []},
            "output": [{"type": "text", "value": "Hello"}],
            "tool_params": {"tools_available": [], "tool_choice": "auto", "parallel_tool_calls": false},
            "tags": {"source": "test"},
            "auxiliary": "extra data",
            "is_custom": true
        }"#;

        let datapoint: UpdateChatInferenceDatapointRequest =
            serde_json::from_str(json_str).unwrap();
        assert_eq!(datapoint.function_name, "test_function");
        assert!(datapoint.output.is_some());
        assert!(datapoint.tool_params.is_some());
        assert_eq!(
            datapoint.tags,
            Some(HashMap::from([("source".to_string(), "test".to_string())]))
        );
        assert_eq!(datapoint.auxiliary, "extra data");
        assert!(datapoint.is_custom);
    }

    #[test]
    fn test_synthetic_json_datapoint_with_none_output() {
        let json_str = r#"{
            "function_name": "test_json_function",
            "input": {"system": {"assistant_name": "Test"}, "messages": []},
            "output": null,
            "output_schema": {},
            "tags": null,
            "auxiliary": "",
            "is_custom": false
        }"#;

        let datapoint: UpdateJsonInferenceDatapointRequest =
            serde_json::from_str(json_str).unwrap();
        assert_eq!(datapoint.function_name, "test_json_function");
        assert_eq!(datapoint.output, None);
        assert_eq!(datapoint.output_schema, json!({}));
        assert_eq!(datapoint.tags, None);
        assert!(!datapoint.is_custom);
        assert_eq!(datapoint.auxiliary, "");
    }

    #[test]
    fn test_synthetic_json_datapoint_with_some_output() {
        let json_str = r#"{
            "function_name": "test_json_function",
            "input": {"system": {"assistant_name": "Test"}, "messages": []},
            "output": {"answer": "Hello"},
            "output_schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
            "tags": {"source": "test"},
            "auxiliary": "extra data",
            "is_custom": true
        }"#;

        let datapoint: UpdateJsonInferenceDatapointRequest =
            serde_json::from_str(json_str).unwrap();
        assert_eq!(datapoint.function_name, "test_json_function");
        assert!(datapoint.output.is_some());
        assert_eq!(datapoint.output.as_ref().unwrap()["answer"], "Hello");
        assert_eq!(
            datapoint.tags,
            Some(HashMap::from([("source".to_string(), "test".to_string())]))
        );
        assert_eq!(datapoint.auxiliary, "extra data");
        assert!(datapoint.is_custom);
    }

    #[test]
    fn test_synthetic_json_datapoint_with_missing_output() {
        let json_str = r#"{
            "function_name": "test_json_function",
            "input": {"system": {"assistant_name": "Test"}, "messages": []},
            "output_schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
            "tags": {"source": "test"},
            "auxiliary": "extra data",
            "is_custom": true
        }"#;

        let datapoint: UpdateJsonInferenceDatapointRequest =
            serde_json::from_str(json_str).unwrap();
        assert_eq!(datapoint.function_name, "test_json_function");
        assert_eq!(datapoint.output, None);
        assert_eq!(
            datapoint.output_schema,
            json!({"type": "object", "properties": {"answer": {"type": "string"}}})
        );
        assert_eq!(
            datapoint.tags,
            Some(HashMap::from([("source".to_string(), "test".to_string())]))
        );
        assert_eq!(datapoint.auxiliary, "extra data");
    }

    #[test]
    fn test_validate_dataset_name_builder() {
        let err = validate_dataset_name("builder").unwrap_err();
        assert_eq!(err.to_string(), "Invalid dataset name: builder. Datasets cannot be named \"builder\" or begin with \"tensorzero::\"");
    }

    #[test]
    fn test_validate_dataset_name_tensorzero_prefix() {
        let err = validate_dataset_name("tensorzero::test").unwrap_err();
        assert_eq!(err.to_string(), "Invalid dataset name: tensorzero::test. Datasets cannot be named \"builder\" or begin with \"tensorzero::\"");
    }

    #[test]
    fn test_validate_dataset_name_valid() {
        validate_dataset_name("test").unwrap();
    }

    #[test]
    fn test_deserialize_synthetic_json_datapoint() {
        let json_str = r#"{"id":"0196368f-1ae8-7551-b5df-9a61593eb307","function_name":"extract_entities","variant_name":"gpt4o_mini_initial_prompt","episode_id":"0196368f-1ae8-7551-b5df-9a7df7e83048","input":"{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"value\":\"Mark Philippoussis ( Australia ) beat Andrei Olhovskiy ( Russia ) 6 - 3 6-4 6-2\"}]}]}","output":"{\"raw\":\"{\\n    \\\"person\\\": [\\\"Mark Philippoussis\\\", \\\"Andrei Olhovskiy\\\"],\\n    \\\"organization\\\": [],\\n    \\\"location\\\": [\\\"Australia\\\", \\\"Russia\\\"],\\n    \\\"miscellaneous\\\": [\\\"6 - 3\\\", \\\"6-4\\\", \\\"6-2\\\"]\\n}\",\"parsed\":{\"person\":[\"Mark Philippoussis\",\"Andrei Olhovskiy\"],\"organization\":[],\"location\":[\"Australia\",\"Russia\"],\"miscellaneous\":[\"6 - 3\",\"6-4\",\"6-2\"]}}","tool_params":"","inference_params":"{\"chat_completion\":{}}","processing_time_ms":12,"output_schema":"{\"$schema\":\"http:\/\/json-schema.org\/draft-07\/schema#\",\"type\":\"object\",\"properties\":{\"person\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}},\"organization\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}},\"location\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}},\"miscellaneous\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}},\"required\":[\"person\",\"organization\",\"location\",\"miscellaneous\"],\"additionalProperties\":false}","auxiliary_content":"","timestamp":"2025-04-14T23:07:50Z","tags":{"tensorzero::dataset_name":"foo","tensorzero::datapoint_id":"0193829b-cd48-7731-9df0-4e325119d96d","tensorzero::evaluation_name":"entity_extraction","tensorzero::evaluation_run_id":"0196368f-19bd-7082-a677-1c0bf346ff24"},"function_type":"json","extra_body":"[]"}"#;
        let _: TaggedInferenceDatabaseInsert = serde_json::from_str(json_str).unwrap();
    }
}
