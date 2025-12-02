use axum::extract::{Path, Query, State};
use axum::Json;
use chrono::Utc;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::{
    types::{PyDict, PyModule},
    IntoPyObjectExt,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, sync::Arc};
use tensorzero_derive::export_schema;
use tracing::instrument;
use uuid::Uuid;

use crate::db::clickhouse::{ClickHouseConnectionInfo, TableName};
use crate::db::datasets::{
    ChatInferenceDatapointInsert, DatapointInsert, DatasetQueries, GetDatapointParams,
    JsonInferenceDatapointInsert,
};
use crate::endpoints::datasets::v1::create_datapoints;
use crate::endpoints::datasets::v1::types::{
    CreateChatDatapointRequest, CreateDatapointRequest, CreateDatapointsRequest,
    CreateJsonDatapointRequest, JsonDatapointOutputUpdate,
};
use crate::endpoints::feedback::{
    validate_parse_demonstration, DemonstrationOutput, DynamicDemonstrationInfo,
};
use crate::function::{FunctionConfig, FunctionConfigType};
use crate::http::TensorzeroHttpClient;
use crate::inference::types::stored_input::StoredInput;
use crate::inference::types::{
    ContentBlockChatOutput, FetchContext, Input, JsonInferenceOutput,
    TaggedInferenceDatabaseInsert, Text,
};
use crate::jsonschema_util::DynamicJSONSchema;
use crate::stored_inference::{SimpleStoredSampleInfo, StoredOutput, StoredSample};
use crate::tool::{LegacyToolCallConfigDatabaseInsert, Tool};
use crate::{
    config::Config,
    error::{Error, ErrorDetails},
    serde_util::{deserialize_optional_string_or_parsed_json, deserialize_string_or_parsed_json},
    tool::{
        deserialize_optional_tool_info, DynamicToolParams, StaticToolConfig,
        ToolCallConfigDatabaseInsert,
    },
    utils::gateway::{AppState, StructuredJson},
    utils::uuid::validate_tensorzero_uuid,
};

#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::{
    content_block_chat_output_to_python, serialize_to_dict, uuid_to_python,
};

#[cfg(debug_assertions)]
use crate::utils::gateway::AppStateData;

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
    limit: u32,
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
        LIMIT {limit:UInt32}
        FORMAT JSONEachRow;"
                .to_string(),
            &HashMap::from([
                ("inference_id", inference_id.to_string().as_str()),
                ("limit", limit.to_string().as_str()),
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
            // TODO(#4737): review the usage of Stored* and *Insert types.
            let datapoint = StoredJsonInferenceDatapoint {
                dataset_name: path_params.dataset_name,
                function_name: inference.function_name,
                name: None,
                id: datapoint_id,
                episode_id: Some(inference.episode_id),
                input: inference.input,
                output,
                output_schema: inference.output_schema,
                tags: Some(inference.tags),
                auxiliary: String::new(),
                is_custom: false,
                source_inference_id: Some(*inference_id),
                staled_at: None,

                // Ignored during insert.
                is_deleted: false,
                updated_at: Utc::now().to_string(),
            };
            let rows_written = clickhouse
                .insert_datapoints(&[DatapointInsert::Json(datapoint.into())])
                .await?;
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
            let datapoint = StoredChatInferenceDatapoint {
                dataset_name: path_params.dataset_name,
                function_name: inference.function_name,
                name: None,
                id: datapoint_id,
                episode_id: Some(inference.episode_id),
                input: inference.input,
                output,
                tool_params: inference.tool_params,
                tags: Some(inference.tags),
                auxiliary: String::new(),
                is_deleted: false,
                is_custom: false,
                source_inference_id: Some(*inference_id),
                staled_at: None,

                // Ignored during insert.
                updated_at: Utc::now().to_string(),
            };
            let rows_written = clickhouse
                .insert_datapoints(&[DatapointInsert::Chat(datapoint.into())])
                .await?;
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
                .into_lazy_resolved_input(&fetch_context)?
                .resolve()
                .await?;
            function_config.validate_input(&chat.input)?;
            // If there are no tool params in the UpdateChatInferenceDatapointRequest, we use the default tool params (empty tools).
            // This is consistent with how they are serialized at inference time.

            // Convert legacy tool params to new format using existing pipeline
            let tool_params_new = if let Some(legacy) = chat.tool_params.clone() {
                // Convert to DynamicToolParams: treat all legacy tools as additional_tools
                // and use FunctionDefault for allowed_tools
                let dynamic_params = DynamicToolParams {
                    allowed_tools: None, // FunctionDefault - use function's default tools
                    additional_tools: Some(
                        legacy
                            .tools_available
                            .iter()
                            .map(|t| Tool::Function(t.clone()))
                            .collect(),
                    ), // All legacy tools as dynamic
                    tool_choice: Some(legacy.tool_choice.clone()),
                    parallel_tool_calls: legacy.parallel_tool_calls,
                    provider_tools: vec![],
                };

                // Use existing pipeline to convert to ToolCallConfigDatabaseInsert
                function_config.dynamic_tool_params_to_database_insert(
                    dynamic_params,
                    &app_state.config.tools,
                )?
            } else {
                Some(ToolCallConfigDatabaseInsert::default())
            };

            // For demonstration validation, convert to ToolCallConfig
            let dynamic_demonstration_info = if let Some(ref tool_params) = tool_params_new {
                DynamicDemonstrationInfo::Chat(
                    tool_params
                        .clone()
                        .into_tool_call_config(&function_config, &app_state.config.tools)?,
                )
            } else {
                DynamicDemonstrationInfo::Chat(None)
            };

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

            let datapoint = StoredChatInferenceDatapoint {
                dataset_name: path_params.dataset_name,
                function_name: chat.function_name,
                name: chat.name,
                id: path_params.datapoint_id,
                episode_id: chat.episode_id,
                input: resolved_input.into_stored_input(),
                output,
                tool_params: tool_params_new,
                tags: chat.tags,
                auxiliary: chat.auxiliary,
                is_deleted: chat.is_deleted,
                is_custom: chat.is_custom,
                source_inference_id: chat.source_inference_id,
                staled_at: chat.staled_at,

                // Ignored during insert.
                updated_at: Utc::now().to_string(),
            };
            let rows_written = app_state
                .clickhouse_connection_info
                .insert_datapoints(&[DatapointInsert::Chat(datapoint.into())])
                .await?;
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
                .into_lazy_resolved_input(&fetch_context)?
                .resolve()
                .await?;
            function_config.validate_input(&json.input)?;

            // Validate the user-provided output_schema
            let schema_str = serde_json::to_string(&json.output_schema).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Failed to serialize output_schema: {e}"),
                })
            })?;
            let parsed_schema = DynamicJSONSchema::parse_from_str(&schema_str).map_err(|e| {
                Error::new(ErrorDetails::InvalidRequest {
                    message: format!("Invalid output_schema: {e}"),
                })
            })?;
            // Ensure the schema is valid by forcing compilation
            parsed_schema.ensure_valid().await?;

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

            let datapoint = StoredJsonInferenceDatapoint {
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
            let rows_written = app_state
                .clickhouse_connection_info
                .insert_datapoints(&[DatapointInsert::Json(datapoint.into())])
                .await?;
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
/// DEPRECATED: Use the POST `/v1/datasets/{dataset_name}/datapoints` endpoint instead.
#[tracing::instrument(name = "create_datapoints_handler", skip(app_state, params))]
#[deprecated(
    since = "2025.11.1",
    note = "Use the POST `/v1/datasets/{dataset_name}/datapoints` endpoint instead."
)]
pub async fn create_datapoints_handler(
    State(app_state): AppState,
    Path(path_params): Path<InsertDatapointPathParams>,
    StructuredJson(params): StructuredJson<InsertDatapointParams>,
) -> Result<Json<Vec<Uuid>>, Error> {
    crate::utils::deprecation_warning(
        &format!("The `/datasets/{}/datapoints` endpoint is deprecated. Please use `/v1/datasets/{}/datapoints` instead.", path_params.dataset_name, path_params.dataset_name)
    );
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

/// DEPRECATED: Use the POST `/v1/datasets/{dataset_name}/datapoints` endpoint instead.
#[tracing::instrument(name = "bulk_insert_datapoints_handler", skip(app_state, params))]
#[deprecated(
    since = "2025.11.1",
    note = "Use the POST `/v1/datasets/{dataset_name}/datapoints` endpoint instead."
)]
pub async fn bulk_insert_datapoints_handler(
    State(app_state): AppState,
    Path(path_params): Path<InsertDatapointPathParams>,
    StructuredJson(params): StructuredJson<InsertDatapointParams>,
) -> Result<Json<Vec<Uuid>>, Error> {
    crate::utils::deprecation_warning(
        &format!("The `/datasets/{}/datapoints/bulk` endpoint is deprecated. Please use `/v1/datasets/{}/datapoints` instead.", path_params.dataset_name, path_params.dataset_name)
    );
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

pub async fn insert_datapoint(
    dataset_name: String,
    params: InsertDatapointParams,
    config: &Config,
    http_client: &TensorzeroHttpClient,
    clickhouse: &ClickHouseConnectionInfo,
) -> Result<Vec<Uuid>, Error> {
    validate_dataset_name(&dataset_name)?;

    // Convert legacy datapoints to v1 request format
    let mut v1_datapoints = Vec::with_capacity(params.datapoints.len());

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

                // Convert the legacy Value output to Vec<ContentBlockChatOutput>
                // Prepare the tool config
                let tool_config = function_config
                    .prepare_tool_config(chat.dynamic_tool_params.clone(), &config.tools)?;
                let dynamic_demonstration_info =
                    DynamicDemonstrationInfo::Chat(tool_config.clone());
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

                v1_datapoints.push(CreateDatapointRequest::Chat(CreateChatDatapointRequest {
                    function_name: chat.function_name,
                    name: chat.name,
                    episode_id: None,
                    input: chat.input,
                    output,
                    dynamic_tool_params: chat.dynamic_tool_params,
                    tags: chat.tags,
                }));
            }
            FunctionConfig::Json(json_function_config) => {
                let json: JsonDatapointInsert = serde_json::from_value(datapoint).map_err(|e| {
                    Error::new(ErrorDetails::InvalidRequest {
                        message: format!("Failed to deserialize json datapoint {i}: {e}"),
                    })
                })?;

                // Legacy insert_datapoint API requires JSON output to be valid, but v1 API doesn't, so we explicitly validate here.
                // We throw away the validation output
                let output_schema = if let Some(user_schema) = &json.output_schema {
                    // Validate the schema by attempting to parse it
                    let schema_str = serde_json::to_string(user_schema).map_err(|e| {
                        Error::new(ErrorDetails::Serialization {
                            message: format!(
                                "Failed to serialize output_schema for datapoint {i}: {e}"
                            ),
                        })
                    })?;
                    let parsed_schema =
                        DynamicJSONSchema::parse_from_str(&schema_str).map_err(|e| {
                            Error::new(ErrorDetails::InvalidRequest {
                                message: format!("Invalid output_schema for datapoint {i}: {e}"),
                            })
                        })?;
                    // Ensure the schema is valid by forcing compilation
                    parsed_schema.ensure_valid().await?;
                    user_schema.clone()
                } else {
                    json_function_config.output_schema.value.clone()
                };
                let dynamic_demonstration_info =
                    DynamicDemonstrationInfo::Json(output_schema.clone());
                if let Some(output) = &json.output {
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
                    let DemonstrationOutput::Json(_) = validated_output else {
                        return Err(Error::new(ErrorDetails::InternalError {
                            message: "Expected valid JSON output from validate_parse_demonstration"
                                .to_string(),
                        }));
                    };
                }

                // Convert legacy Value output to JsonDatapointOutputUpdate
                let output_update = match json.output {
                    Some(output) => {
                        let raw = match output {
                            serde_json::Value::Object(_) => Some(output.to_string()),
                            serde_json::Value::Null => None,
                            _ => {
                                return Err(Error::new(ErrorDetails::InvalidRequest {
                                    message: "The field `output` must be an object or null."
                                        .to_string(),
                                }))
                            }
                        };
                        Some(JsonDatapointOutputUpdate { raw })
                    }
                    None => None,
                };

                v1_datapoints.push(CreateDatapointRequest::Json(CreateJsonDatapointRequest {
                    function_name: json.function_name,
                    name: json.name,
                    episode_id: None,
                    input: json.input,
                    output: output_update,
                    output_schema: json.output_schema,
                    tags: json.tags,
                }));
            }
        }
    }

    // Delegate to the v1 implementation
    let v1_request = CreateDatapointsRequest {
        datapoints: v1_datapoints,
    };

    let response =
        create_datapoints(config, http_client, clickhouse, &dataset_name, v1_request).await?;

    Ok(response.ids)
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
    (dataset_name, function_name, id, episode_id, input, output, output_schema,
     tags, auxiliary, is_deleted, updated_at, staled_at, source_inference_id, is_custom, name)
    SELECT dataset_name, function_name, id, episode_id, input, output, output_schema,
           tags, auxiliary, is_deleted, now64(), now64(), source_inference_id, is_custom, name
    FROM JsonInferenceDatapoint
    WHERE id = {datapoint_id: UUID} AND dataset_name = {dataset_name: String}
";
    let chat_delete_query = r"
    INSERT INTO ChatInferenceDatapoint
    (dataset_name, function_name, name, id, episode_id, input, output, tool_params,
    dynamic_tools, dynamic_provider_tools, tool_choice, parallel_tool_calls, allowed_tools,
     tags, auxiliary, is_deleted, is_custom, source_inference_id, updated_at, staled_at)
    SELECT dataset_name, function_name, name, id, episode_id, input, output, tool_params,
    dynamic_tools, dynamic_provider_tools, tool_choice, parallel_tool_calls, allowed_tools,
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
    let datapoints = list_datapoints(
        path_params.dataset_name,
        &app_state.clickhouse_connection_info,
        &app_state.config,
        query_params.function_name,
        query_params.limit,
        query_params.offset,
    )
    .await?;

    Ok(Json(datapoints))
}

#[tracing::instrument(name = "list_datapoints", skip(clickhouse))]
pub async fn list_datapoints(
    dataset_name: String,
    clickhouse: &ClickHouseConnectionInfo,
    config: &Config,
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
            dynamic_tools,
            dynamic_provider_tools,
            parallel_tool_calls,
            tool_choice,
            allowed_tools,
            '\N' as output_schema, -- for column alignment in UNION ALL
            tags,
            auxiliary,
            source_inference_id,
            is_deleted,
            is_custom,
            staled_at,
            formatDateTime(updated_at, '%Y-%m-%dT%H:%i:%SZ') AS updated_at
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
            [] as dynamic_tools,
            [] as dynamic_provider_tools,
            NULL as parallel_tool_calls,
            NULL as tool_choice,
            NULL as allowed_tools,
            output_schema,
            tags,
            auxiliary,
            source_inference_id,
            is_deleted,
            is_custom,
            staled_at,
            formatDateTime(updated_at, '%Y-%m-%dT%H:%i:%SZ') AS updated_at
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
        .map(|line| serde_json::from_str::<StoredDatapoint>(line)?.into_datapoint())
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
    let datapoint = app_state
        .clickhouse_connection_info
        .get_datapoint(&GetDatapointParams {
            dataset_name: path_params.dataset_name,
            datapoint_id: path_params.datapoint_id,
            allow_stale: None,
        })
        .await?;

    // Convert storage type to wire type
    let wire = datapoint.into_datapoint()?;
    Ok(Json(wire))
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

/// Wire variant of Datapoint enum for API responses with Python/TypeScript bindings
/// This one should be used in all public interfaces.
#[derive(Clone, Debug, Deserialize, Serialize, ts_rs::TS, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "pyo3", pyclass(str, name = "LegacyDatapoint"))]
#[ts(export)]
#[export_schema]
pub enum Datapoint {
    #[schemars(title = "DatapointChat")]
    Chat(ChatInferenceDatapoint),
    #[schemars(title = "DatapointJson")]
    Json(JsonInferenceDatapoint),
}

impl std::fmt::Display for Datapoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl Datapoint {
    pub fn dataset_name(&self) -> &str {
        match self {
            Datapoint::Chat(datapoint) => &datapoint.dataset_name,
            Datapoint::Json(datapoint) => &datapoint.dataset_name,
        }
    }

    pub fn function_name(&self) -> &str {
        match self {
            Datapoint::Chat(datapoint) => &datapoint.function_name,
            Datapoint::Json(datapoint) => &datapoint.function_name,
        }
    }

    pub fn id(&self) -> Uuid {
        match self {
            Datapoint::Chat(datapoint) => datapoint.id,
            Datapoint::Json(datapoint) => datapoint.id,
        }
    }

    pub fn input(&self) -> &Input {
        match self {
            Datapoint::Chat(datapoint) => &datapoint.input,
            Datapoint::Json(datapoint) => &datapoint.input,
        }
    }

    pub fn tool_call_config(&self) -> Option<&DynamicToolParams> {
        match self {
            Datapoint::Chat(datapoint) => Some(&datapoint.tool_params),
            Datapoint::Json(_) => None,
        }
    }

    pub fn output_schema(&self) -> Option<&serde_json::Value> {
        match self {
            Datapoint::Chat(_datapoint) => None,
            Datapoint::Json(datapoint) => Some(&datapoint.output_schema),
        }
    }
}

impl StoredDatapoint {
    /// Convert to wire type, properly handling tool params by subtracting static tools
    /// TODO(shuyangli): Add parameter to optionally fetch files from object storage
    pub fn into_datapoint(self) -> Result<Datapoint, Error> {
        match self {
            StoredDatapoint::Chat(chat) => Ok(Datapoint::Chat(chat.into_datapoint())),
            StoredDatapoint::Json(json) => Ok(Datapoint::Json(json.into_datapoint())),
        }
    }
}

impl ChatInferenceDatapoint {
    /// Convert to storage type, properly handling tool params with function config.
    /// If `fetch_context` is provided, any external URLs or Base64 files will be properly resolved and stored.
    /// If `fetch_context` is not provided, we will return an error if the input contains any external URLs or Base64 files,
    /// because we cannot represent them as the database type.
    pub async fn into_storage(
        self,
        function_config: &FunctionConfig,
        static_tools: &HashMap<String, Arc<StaticToolConfig>>,
        fetch_context: &FetchContext<'_>,
    ) -> Result<StoredChatInferenceDatapoint, Error> {
        let tool_params = function_config
            .dynamic_tool_params_to_database_insert(self.tool_params, static_tools)?;
        let stored_input = self
            .input
            .into_lazy_resolved_input(fetch_context)?
            .into_stored_input(fetch_context.object_store_info)
            .await?;

        Ok(StoredChatInferenceDatapoint {
            dataset_name: self.dataset_name,
            function_name: self.function_name,
            id: self.id,
            episode_id: self.episode_id,
            input: stored_input,
            output: self.output,
            tool_params,
            tags: self.tags,
            auxiliary: self.auxiliary,
            is_deleted: self.is_deleted,
            is_custom: self.is_custom,
            source_inference_id: self.source_inference_id,
            staled_at: self.staled_at,
            updated_at: self.updated_at,
            name: self.name,
        })
    }

    /// Convert to storage type, without resolving network resources for files.
    /// This is used in PyO3 where we do not have a fetch context available.
    /// Returns an error if the input contains any external URLs or Base64 files.
    pub fn into_storage_without_file_handling(
        self,
        function_config: &FunctionConfig,
        static_tools: &HashMap<String, Arc<StaticToolConfig>>,
    ) -> Result<StoredChatInferenceDatapoint, Error> {
        let tool_params = function_config
            .dynamic_tool_params_to_database_insert(self.tool_params, static_tools)?;

        let stored_input = self.input.into_stored_input_without_file_handling()?;

        Ok(StoredChatInferenceDatapoint {
            dataset_name: self.dataset_name,
            function_name: self.function_name,
            id: self.id,
            episode_id: self.episode_id,
            input: stored_input,
            output: self.output,
            tool_params,
            tags: self.tags,
            auxiliary: self.auxiliary,
            is_deleted: self.is_deleted,
            is_custom: self.is_custom,
            source_inference_id: self.source_inference_id,
            staled_at: self.staled_at,
            updated_at: self.updated_at,
            name: self.name,
        })
    }
}

impl JsonInferenceDatapoint {
    /// Convert to storage type, possibly handling input file storage.
    /// If `fetch_context` is provided, any external URLs or Base64 files will be properly resolved and stored.
    /// If `fetch_context` is not provided, we will return an error if the input contains any external URLs or Base64 files,
    /// because we cannot represent them as the database type.
    pub async fn into_storage(
        self,
        fetch_context: Option<&FetchContext<'_>>,
    ) -> Result<StoredJsonInferenceDatapoint, Error> {
        let stored_input = match fetch_context {
            Some(fetch_context) => {
                self.input
                    .into_lazy_resolved_input(fetch_context)?
                    .into_stored_input(fetch_context.object_store_info)
                    .await?
            }
            None => self.input.into_stored_input_without_file_handling()?,
        };

        Ok(StoredJsonInferenceDatapoint {
            dataset_name: self.dataset_name,
            function_name: self.function_name,
            id: self.id,
            episode_id: self.episode_id,
            input: stored_input,
            output: self.output,
            output_schema: self.output_schema,
            tags: self.tags,
            auxiliary: self.auxiliary,
            is_deleted: self.is_deleted,
            is_custom: self.is_custom,
            source_inference_id: self.source_inference_id,
            staled_at: self.staled_at,
            updated_at: self.updated_at,
            name: self.name,
        })
    }

    /// Convert to storage type, without resolving network resources for files.
    /// This is used in PyO3 where we do not have a fetch context available.
    /// Returns an error if the input contains any external URLs or Base64 files.
    pub fn into_storage_without_file_handling(self) -> Result<StoredJsonInferenceDatapoint, Error> {
        let stored_input = self.input.into_stored_input_without_file_handling()?;

        Ok(StoredJsonInferenceDatapoint {
            dataset_name: self.dataset_name,
            function_name: self.function_name,
            id: self.id,
            episode_id: self.episode_id,
            input: stored_input,
            output: self.output,
            output_schema: self.output_schema,
            tags: self.tags,
            auxiliary: self.auxiliary,
            is_deleted: self.is_deleted,
            is_custom: self.is_custom,
            source_inference_id: self.source_inference_id,
            staled_at: self.staled_at,
            updated_at: self.updated_at,
            name: self.name,
        })
    }
}

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
        // This is python_helpers.rs convert_response_to_python_dataclass, but we can't import it across crates.
        // We will remove the whole Datapoint type and replace with generated types soon.

        // Serialize Rust response to JSON dict

        let dict = serialize_to_dict(py, self.input().clone())?;

        // Import the target dataclass
        let module = PyModule::import(py, "tensorzero")?;
        let data_class = module.getattr("Input")?;

        // Use dacite.from_dict to construct the dataclass, so that it can handle nested dataclass construction.
        let dacite = PyModule::import(py, "dacite")?;
        let from_dict = dacite.getattr("from_dict")?;

        // Call dacite.from_dict(data_class=TargetClass, data=dict)
        let kwargs = PyDict::new(py);
        kwargs.set_item("data_class", data_class)?;
        kwargs.set_item("data", dict)?;

        from_dict.call((), Some(&kwargs))
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
    pub fn get_allowed_tools(&self) -> Option<Vec<String>> {
        match self {
            Datapoint::Chat(datapoint) => datapoint.tool_params.allowed_tools.clone(),
            Datapoint::Json(_) => None,
        }
    }

    #[getter]
    pub fn get_additional_tools<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Datapoint::Chat(datapoint) => datapoint
                .tool_params
                .additional_tools
                .clone()
                .into_bound_py_any(py),
            Datapoint::Json(_) => Ok(py.None().into_bound(py)),
        }
    }

    // Note: We're intentionally skipping tool_choice as it's not exposed in the Python API

    #[getter]
    pub fn get_parallel_tool_calls(&self) -> Option<bool> {
        match self {
            Datapoint::Chat(datapoint) => datapoint.tool_params.parallel_tool_calls,
            Datapoint::Json(_) => None,
        }
    }

    #[getter]
    pub fn get_provider_tools<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Datapoint::Chat(datapoint) => datapoint
                .tool_params
                .provider_tools
                .clone()
                .into_bound_py_any(py),
            Datapoint::Json(_) => Ok(py.None().into_bound(py)),
        }
    }

    #[getter]
    pub fn get_output_schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self {
            Datapoint::Chat(_) => py.None().into_bound(py),
            Datapoint::Json(datapoint) => {
                serialize_to_dict(py, &datapoint.output_schema)?.into_bound(py)
            }
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
        match self {
            Datapoint::Chat(datapoint) => datapoint.name.clone(),
            Datapoint::Json(datapoint) => datapoint.name.clone(),
        }
    }
}

/// Storage variant of Datapoint enum for database operations (no Python/TypeScript bindings)
/// Convert to Datapoint with `.into_datapoint()`
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StoredDatapoint {
    Chat(StoredChatInferenceDatapoint),
    Json(StoredJsonInferenceDatapoint),
}

impl StoredDatapoint {
    pub fn dataset_name(&self) -> &str {
        match self {
            StoredDatapoint::Chat(datapoint) => &datapoint.dataset_name,
            StoredDatapoint::Json(datapoint) => &datapoint.dataset_name,
        }
    }

    pub fn input(&self) -> &StoredInput {
        match self {
            StoredDatapoint::Chat(datapoint) => &datapoint.input,
            StoredDatapoint::Json(datapoint) => &datapoint.input,
        }
    }

    pub fn tool_call_config(&self) -> Option<&ToolCallConfigDatabaseInsert> {
        match self {
            StoredDatapoint::Chat(datapoint) => datapoint.tool_params.as_ref(),
            StoredDatapoint::Json(_datapoint) => None,
        }
    }

    pub fn output_schema(&self) -> Option<&serde_json::Value> {
        match self {
            StoredDatapoint::Chat(_datapoint) => None,
            StoredDatapoint::Json(datapoint) => Some(&datapoint.output_schema),
        }
    }

    pub fn id(&self) -> Uuid {
        match self {
            StoredDatapoint::Chat(datapoint) => datapoint.id,
            StoredDatapoint::Json(datapoint) => datapoint.id,
        }
    }

    pub fn name(&self) -> Option<&str> {
        match self {
            StoredDatapoint::Chat(datapoint) => datapoint.name.as_deref(),
            StoredDatapoint::Json(datapoint) => datapoint.name.as_deref(),
        }
    }
}

impl std::fmt::Display for StoredDatapoint {
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

/// Wire variant of ChatInferenceDatapoint for API responses with Python/TypeScript bindings
/// This one should be used in all public interfaces.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS, JsonSchema)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[ts(export, optional_fields)]
#[export_schema]
pub struct ChatInferenceDatapoint {
    pub dataset_name: String,
    pub function_name: String,
    pub id: Uuid,
    pub episode_id: Option<Uuid>,
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub input: Input,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    #[serde(deserialize_with = "deserialize_optional_string_or_parsed_json")]
    pub output: Option<Vec<ContentBlockChatOutput>>,
    // `tool_params` are always flattened to match the convention of LLM APIs
    #[serde(flatten)]
    #[serde(default)]
    pub tool_params: DynamicToolParams,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    #[cfg_attr(test, ts(type = "Record<string, string>"))]
    pub tags: Option<HashMap<String, String>>,
    #[serde(skip_serializing, default)]
    pub auxiliary: String,
    pub is_deleted: bool,
    #[serde(default)]
    pub is_custom: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub source_inference_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub staled_at: Option<String>,
    pub updated_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub name: Option<String>,
}

impl std::fmt::Display for ChatInferenceDatapoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl StoredChatInferenceDatapoint {
    /// Convert to wire type, converting tool params from storage format to wire format using From<> trait
    /// TODO(shuyangli): Add parameter to optionally fetch files from object storage
    pub fn into_datapoint(self) -> ChatInferenceDatapoint {
        let tool_params = self.tool_params.map(|tp| tp.into()).unwrap_or_default();

        ChatInferenceDatapoint {
            dataset_name: self.dataset_name,
            function_name: self.function_name,
            id: self.id,
            episode_id: self.episode_id,
            input: self.input.into_input(),
            output: self.output,
            tool_params,
            tags: self.tags,
            auxiliary: self.auxiliary,
            is_deleted: self.is_deleted,
            is_custom: self.is_custom,
            source_inference_id: self.source_inference_id,
            staled_at: self.staled_at,
            updated_at: self.updated_at,
            name: self.name,
        }
    }
}

/// Storage variant of ChatInferenceDatapoint for database operations (no Python/TypeScript bindings)
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StoredChatInferenceDatapoint {
    pub dataset_name: String,
    pub function_name: String,
    pub id: Uuid,
    pub episode_id: Option<Uuid>,
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub input: StoredInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    #[serde(deserialize_with = "deserialize_optional_string_or_parsed_json")]
    pub output: Option<Vec<ContentBlockChatOutput>>,
    #[serde(flatten, deserialize_with = "deserialize_optional_tool_info")]
    pub tool_params: Option<ToolCallConfigDatabaseInsert>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub tags: Option<HashMap<String, String>>,
    #[serde(skip_serializing, default)] // this will become an object
    pub auxiliary: String,
    pub is_deleted: bool,
    #[serde(default)]
    pub is_custom: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub source_inference_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub staled_at: Option<String>,
    pub updated_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub name: Option<String>,
}

impl std::fmt::Display for StoredChatInferenceDatapoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl From<StoredChatInferenceDatapoint> for ChatInferenceDatapointInsert {
    fn from(datapoint: StoredChatInferenceDatapoint) -> Self {
        ChatInferenceDatapointInsert {
            dataset_name: datapoint.dataset_name,
            function_name: datapoint.function_name,
            name: datapoint.name,
            id: datapoint.id,
            episode_id: datapoint.episode_id,
            input: datapoint.input,
            output: datapoint.output,
            tool_params: datapoint.tool_params,
            tags: datapoint.tags,
            auxiliary: datapoint.auxiliary,
            staled_at: datapoint.staled_at,
            source_inference_id: datapoint.source_inference_id,
            is_custom: datapoint.is_custom,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS, JsonSchema)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[export_schema]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct JsonInferenceDatapoint {
    pub dataset_name: String,
    pub function_name: String,
    pub id: Uuid,
    pub episode_id: Option<Uuid>,
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub input: Input,
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

/// Storage variant of JsonInferenceDatapoint for database operations (no Python/TypeScript bindings).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StoredJsonInferenceDatapoint {
    /// Name of the dataset to write to.
    pub dataset_name: String,

    /// Name of the function that generated this datapoint.
    pub function_name: String,

    /// Unique identifier for the datapoint.
    pub id: Uuid,

    /// Episode ID that the datapoint belongs to.
    pub episode_id: Option<Uuid>,

    /// Input type that we directly store in ClickHouse.
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub input: StoredInput,

    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_string_or_parsed_json"
    )]
    pub output: Option<JsonInferenceOutput>,

    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub output_schema: serde_json::Value,

    // By default, ts_rs generates { [key in string]?: string } | undefined, which means values are string | undefined which isn't what we want.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tags: Option<HashMap<String, String>>,

    /// Deprecated, do not use.
    #[serde(skip_serializing, default)]
    pub auxiliary: String,

    /// If true, this datapoint was deleted.
    pub is_deleted: bool,

    /// If true, this datapoint was manually created or edited by the user.
    #[serde(default)]
    pub is_custom: bool,

    /// Source inference ID that generated this datapoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_inference_id: Option<Uuid>,

    /// Timestamp when the datapoint was marked as stale.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub staled_at: Option<String>,

    /// Timestamp when the datapoint was updated.
    pub updated_at: String,

    /// Human-readable name of the datapoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl From<StoredJsonInferenceDatapoint> for JsonInferenceDatapointInsert {
    fn from(datapoint: StoredJsonInferenceDatapoint) -> Self {
        JsonInferenceDatapointInsert {
            dataset_name: datapoint.dataset_name,
            function_name: datapoint.function_name,
            name: datapoint.name,
            id: datapoint.id,
            episode_id: datapoint.episode_id,
            input: datapoint.input,
            output: datapoint.output,
            output_schema: datapoint.output_schema,
            tags: datapoint.tags,
            auxiliary: datapoint.auxiliary,
            staled_at: datapoint.staled_at,
            source_inference_id: datapoint.source_inference_id,
            is_custom: datapoint.is_custom,
        }
    }
}

impl StoredJsonInferenceDatapoint {
    pub fn into_datapoint(self) -> JsonInferenceDatapoint {
        JsonInferenceDatapoint {
            dataset_name: self.dataset_name,
            function_name: self.function_name,
            id: self.id,
            episode_id: self.episode_id,
            input: self.input.into_input(),
            output: self.output,
            output_schema: self.output_schema,
            tags: self.tags,
            auxiliary: self.auxiliary,
            is_deleted: self.is_deleted,
            is_custom: self.is_custom,
            source_inference_id: self.source_inference_id,
            staled_at: self.staled_at,
            updated_at: self.updated_at,
            name: self.name,
        }
    }
}

impl StoredSample for StoredDatapoint {
    fn function_name(&self) -> &str {
        match self {
            StoredDatapoint::Chat(datapoint) => &datapoint.function_name,
            StoredDatapoint::Json(datapoint) => &datapoint.function_name,
        }
    }

    fn input(&self) -> &StoredInput {
        match self {
            StoredDatapoint::Chat(datapoint) => &datapoint.input,
            StoredDatapoint::Json(datapoint) => &datapoint.input,
        }
    }

    fn input_mut(&mut self) -> &mut StoredInput {
        match self {
            StoredDatapoint::Chat(datapoint) => &mut datapoint.input,
            StoredDatapoint::Json(datapoint) => &mut datapoint.input,
        }
    }

    fn into_input(self) -> StoredInput {
        match self {
            StoredDatapoint::Chat(datapoint) => datapoint.input,
            StoredDatapoint::Json(datapoint) => datapoint.input,
        }
    }

    fn owned_simple_info(self) -> SimpleStoredSampleInfo {
        match self {
            StoredDatapoint::Chat(datapoint) => SimpleStoredSampleInfo {
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
            StoredDatapoint::Json(datapoint) => {
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
pub struct UpdateChatInferenceDatapointRequest {
    pub function_name: String,
    #[serde(default)]
    pub episode_id: Option<Uuid>,
    pub input: Input,
    #[serde(default)]
    #[serde(deserialize_with = "deserialize_optional_json_value")]
    pub output: Option<serde_json::Value>,
    pub tool_params: Option<LegacyToolCallConfigDatabaseInsert>,
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

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct StaleDatasetResponse {
    pub num_staled_datapoints: u64,
}

/// Helper function for staling a dataset. Used by the Rust client.
pub async fn stale_dataset(
    clickhouse: &impl DatasetQueries,
    dataset_name: &str,
) -> Result<StaleDatasetResponse, Error> {
    let num_staled_datapoints = clickhouse.delete_datapoints(dataset_name, None).await?;
    Ok(StaleDatasetResponse {
        num_staled_datapoints,
    })
}

/// Deprecated in favor of `delete_dataset_handler` in `v1/datasets/v1/mod.rs`.
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
        // Test with tool config fields flattened at top level (new Migration 0041 format)
        let json_str = r#"{
            "function_name": "test_function",
            "input": {"system": {"assistant_name": "Test"}, "messages": []},
            "output": [{"type": "text", "value": "Hello"}],
            "tool_params": {"tools_available": [], "tool_choice": "auto", "parallel_tool_calls": false},
            "dynamic_tools": [],
            "dynamic_provider_tools": [],
            "allowed_tools": {"tools": [], "choice": "function_default"},
            "tool_choice": "auto",
            "parallel_tool_calls": false,
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
