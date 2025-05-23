use axum::extract::{Path, Query, State};
use axum::Json;
use futures::future;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, future::Future, pin::Pin};
use tracing::instrument;
use uuid::Uuid;

use crate::{
    clickhouse::{ClickHouseConnectionInfo, ExternalDataInfo},
    config_parser::Config,
    error::{Error, ErrorDetails},
    function::FunctionConfig,
    gateway_util::{AppState, StructuredJson},
    inference::types::{
        batch::{deserialize_json_string, deserialize_optional_json_string},
        ChatInferenceDatabaseInsert, ContentBlockChatOutput, FetchContext, Input,
        JsonInferenceDatabaseInsert, JsonInferenceOutput, ResolvedInput,
    },
    tool::{DynamicToolParams, ToolCallConfigDatabaseInsert},
    uuid_util::validate_tensorzero_uuid,
};

#[cfg(debug_assertions)]
use crate::gateway_util::AppStateData;

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
            r#"
        SELECT
          id,
          inference_id,
          value,
          formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
        FROM DemonstrationFeedbackByInferenceId
        WHERE inference_id = {inference_id:String}
        ORDER BY toUInt128(id) DESC
        LIMIT {page_size:UInt32}
        FORMAT JSONEachRow;"#
                .to_string(),
            Some(&HashMap::from([
                ("inference_id", inference_id.to_string().as_str()),
                ("page_size", page_size.to_string().as_str()),
            ])),
        )
        .await?;
    if result.is_empty() {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!("No demonstration found for inference `{inference_id}`"),
        }));
    }
    let demonstration: Demonstration = serde_json::from_str(&result).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to deserialize demonstration ClickHouse response: {e}"),
        })
    })?;
    Ok(demonstration)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, tag = "function_type", rename_all = "snake_case")]
pub enum TaggedInferenceDatabaseInsert {
    Chat(ChatInferenceDatabaseInsert),
    Json(JsonInferenceDatabaseInsert),
}

async fn query_inference_for_datapoint(
    clickhouse: &ClickHouseConnectionInfo,
    inference_id: Uuid,
) -> Result<TaggedInferenceDatabaseInsert, Error> {
    let result: String = clickhouse
        .run_query_synchronous(
            r#"
            SELECT
  uint_to_uuid(i.id_uint) AS id,

  -- Common columns (pick via IF)
  IF(i.function_type = 'chat', c.function_name, j.function_name) AS function_name,
  IF(i.function_type = 'chat', c.variant_name,   j.variant_name)   AS variant_name,
  IF(i.function_type = 'chat', c.episode_id,     j.episode_id)     AS episode_id,
  IF(i.function_type = 'chat', c.input,          j.input)          AS input,
  IF(i.function_type = 'chat', c.output,         j.output)         AS output,

  -- Chat-specific columns
  IF(i.function_type = 'chat', c.tool_params, '') AS tool_params,

  -- Inference params (common name in the union)
  IF(i.function_type = 'chat', c.inference_params, j.inference_params) AS inference_params,

  -- Processing time
  IF(i.function_type = 'chat', c.processing_time_ms, j.processing_time_ms) AS processing_time_ms,

  -- JSON-specific columns
  IF(i.function_type = 'json', j.output_schema, '') AS output_schema,
  IF(i.function_type = 'json', j.auxiliary_content, '') AS auxiliary_content,

  -- Timestamps & tags
  IF(i.function_type = 'chat',
   formatDateTime(c.timestamp, '%Y-%m-%dT%H:%i:%SZ'),
   formatDateTime(j.timestamp, '%Y-%m-%dT%H:%i:%SZ')
) AS timestamp,
  IF(i.function_type = 'chat', c.tags,      j.tags)      AS tags,

  -- Discriminator itself
  i.function_type

FROM InferenceById i FINAL
LEFT JOIN ChatInference c
  ON i.id_uint = toUInt128(c.id)
LEFT JOIN JsonInference j
  ON i.id_uint = toUInt128(j.id)
WHERE uint_to_uuid(i.id_uint) = {id:String}
FORMAT JSONEachRow;"#
                .to_string(),
            Some(&HashMap::from([("id", inference_id.to_string().as_str())])),
        )
        .await?;

    if result.is_empty() {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!("Inference `{inference_id}` not found"),
        }));
    }

    let inference_data: TaggedInferenceDatabaseInsert =
        serde_json::from_str(&result).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to deserialize inference data: {e}"),
            })
        })?;
    Ok(inference_data)
}

async fn insert_from_existing(
    clickhouse: &ClickHouseConnectionInfo,
    path_params: InsertPathParams,
    existing: &ExistingInferenceInfo,
) -> Result<Uuid, Error> {
    let inference_data = query_inference_for_datapoint(clickhouse, existing.inference_id).await?;
    let datapoint_id = Uuid::now_v7();

    match inference_data {
        TaggedInferenceDatabaseInsert::Json(inference) => {
            let output = match existing.output {
                OutputKind::Inherit => Some(inference.output),
                OutputKind::Demonstration => {
                    let demonstration =
                        query_demonstration(clickhouse, existing.inference_id, 1).await?;
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
            let datapoint = JsonInferenceDatapoint {
                dataset_name: path_params.dataset_name,
                function_name: inference.function_name,
                id: datapoint_id,
                episode_id: Some(inference.episode_id),
                input: inference.input,
                output,
                output_schema: inference.output_schema,
                tags: Some(inference.tags),
                auxiliary: "{}".to_string(),
                is_deleted: false,
                source_inference_id: Some(existing.inference_id),
                staled_at: None,
            };
            let rows_written = put_deduped_json_datapoints(clickhouse, &[datapoint]).await?;
            if rows_written == 0 {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: "Datapoint with this source_inference_id already exists".to_string(),
                }));
            }
        }
        TaggedInferenceDatabaseInsert::Chat(inference) => {
            let output = match existing.output {
                OutputKind::Inherit => Some(inference.output),
                OutputKind::Demonstration => {
                    let demonstration =
                        query_demonstration(clickhouse, existing.inference_id, 1).await?;
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
            let datapoint = ChatInferenceDatapoint {
                dataset_name: path_params.dataset_name,
                function_name: inference.function_name,
                id: datapoint_id,
                episode_id: Some(inference.episode_id),
                input: inference.input,
                output,
                tool_params: inference.tool_params,
                tags: Some(inference.tags),
                auxiliary: "{}".to_string(),
                is_deleted: false,
                source_inference_id: Some(existing.inference_id),
                staled_at: None,
            };
            let rows_written = put_deduped_chat_datapoints(clickhouse, &[datapoint]).await?;
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
#[instrument(name = "insert_datapoint", skip(app_state))]
pub async fn insert_from_existing_datapoint_handler(
    State(app_state): AppState,
    Path(path_params): Path<InsertPathParams>,
    StructuredJson(existing_inference_info): StructuredJson<ExistingInferenceInfo>,
) -> Result<Json<InsertDatapointResponse>, Error> {
    validate_dataset_name(&path_params.dataset_name)?;
    let datapoint_id = insert_from_existing(
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
#[instrument(name = "update_datapoint", skip(app_state))]
pub async fn update_datapoint_handler(
    State(app_state): AppState,
    Path(path_params): Path<UpdatePathParams>,
    // This is deserialized as either a `SyntheticChatInferenceDatapoint` or `SyntheticJsonInferenceDatapoint`,
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

    match **function_config {
        FunctionConfig::Chat(_) => {
            let chat: SyntheticChatInferenceDatapoint =
                serde_json::from_value(params).map_err(|e| {
                    Error::new(ErrorDetails::InvalidRequest {
                        message: format!("Failed to deserialize chat datapoint: {e}"),
                    })
                })?;

            let resolved_input = chat.input.clone().resolve(&fetch_context).await?;
            function_config.validate_input(&chat.input)?;
            // If there are no tool params in the SyntheticChatInferenceDatapoint, we use the default tool params (empty tools).
            // This is consistent with how they are serialized at inference time.
            let dynamic_demonstration_info = DynamicDemonstrationInfo::Chat(
                chat.tool_params
                    .clone()
                    .map(|x| x.into())
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
                id: path_params.datapoint_id,
                episode_id: None,
                input: resolved_input,
                output,
                tool_params: chat.tool_params,
                tags: chat.tags,
                auxiliary: chat.auxiliary,
                is_deleted: false,
                source_inference_id: chat.source_inference_id,
                staled_at: None,
            };
            let rows_written =
                put_deduped_chat_datapoints(&app_state.clickhouse_connection_info, &[datapoint])
                    .await?;
            if rows_written == 0 {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: "Datapoint with this source_inference_id already exists".to_string(),
                }));
            }
        }
        FunctionConfig::Json(_) => {
            let json: SyntheticJsonInferenceDatapoint =
                serde_json::from_value(params).map_err(|e| {
                    Error::new(ErrorDetails::InvalidRequest {
                        message: format!("Failed to deserialize JSON datapoint: {e}"),
                    })
                })?;
            let resolved_input = json.input.clone().resolve(&fetch_context).await?;
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
                id: path_params.datapoint_id,
                episode_id: None,
                input: resolved_input,
                output,
                output_schema: json.output_schema,
                tags: json.tags,
                auxiliary: json.auxiliary,
                is_deleted: false,
                source_inference_id: json.source_inference_id,
                staled_at: None,
            };
            let rows_written =
                put_deduped_json_datapoints(&app_state.clickhouse_connection_info, &[datapoint])
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

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InsertDatapointParams {
    pub datapoints: Vec<Value>,
}

#[derive(Debug, Deserialize)]
pub struct InsertDatapointPathParams {
    pub dataset_name: String,
}

// The handler for the POST `/datasets/:dataset_name/datapoints/bulk` endpoint.
/// This inserts a new datapoint into `ChatInferenceDatapoint`/`JsonInferenceDatapoint`/
#[tracing::instrument(name = "bulk_insert_datapoints_handler", skip(app_state, params))]
pub async fn bulk_insert_datapoints_handler(
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

pub async fn insert_datapoint(
    dataset_name: String,
    params: InsertDatapointParams,
    config: &Config<'_>,
    http_client: &Client,
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
                let resolved_input = chat.input.resolve(&fetch_context).await.map_err(|e| {
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
                    id: datapoint_id,
                    episode_id: None,
                    input: resolved_input,
                    output,
                    tool_params: tool_config.as_ref().map(|x| x.clone().into()),
                    tags: chat.tags,
                    auxiliary: "".to_string(),
                    is_deleted: false,
                    source_inference_id: None,
                    staled_at: None,
                })
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
                let resolved_input = json.input.resolve(&fetch_context).await.map_err(|e| {
                    Error::new(ErrorDetails::InternalError {
                        message: format!("Failed to resolve input for datapoint {i}: {e}"),
                    })
                })?;
                // Validate the outputs against the output schema
                let output_schema = json
                    .output_schema
                    .unwrap_or(json_function_config.output_schema.value.clone());
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
                            .and_then(|v| v.as_str().map(|s| s.to_string())),
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
                    id: datapoint_id,
                    episode_id: None,
                    input: resolved_input,
                    output,
                    output_schema,
                    tags: json.tags,
                    auxiliary: "".to_string(),
                    is_deleted: false,
                    source_inference_id: None,
                    staled_at: None,
                };
                json_datapoints.push(datapoint);
            }
        }
    }

    #[expect(clippy::type_complexity)]
    let mut futures_vec: Vec<Pin<Box<dyn Future<Output = Result<u64, Error>> + Send>>> = Vec::new();

    if !chat_datapoints.is_empty() {
        futures_vec.push(Box::pin(put_deduped_chat_datapoints(
            clickhouse,
            &chat_datapoints,
        )));
    }
    if !json_datapoints.is_empty() {
        futures_vec.push(Box::pin(put_deduped_json_datapoints(
            clickhouse,
            &json_datapoints,
        )));
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
    let json_delete_query = r#"
    INSERT INTO JsonInferenceDatapoint
    (dataset_name, function_name, id, episode_id, input, output, output_schema,
     tags, auxiliary, is_deleted, source_inference_id, updated_at, staled_at)
    SELECT dataset_name, function_name, id, episode_id, input, output, output_schema,
           tags, auxiliary, is_deleted, source_inference_id, now64(), now64()
    FROM JsonInferenceDatapoint
    WHERE id = {datapoint_id: UUID} AND dataset_name = {dataset_name: String}
"#;
    let chat_delete_query = r#"
    INSERT INTO ChatInferenceDatapoint
    (dataset_name, function_name, id, episode_id, input, output, tool_params,
     tags, auxiliary, is_deleted, source_inference_id, updated_at, staled_at)
    SELECT dataset_name, function_name, id, episode_id, input, output, tool_params,
           tags, auxiliary, is_deleted, source_inference_id, now64(), now64()
    FROM ChatInferenceDatapoint
    WHERE id = {datapoint_id: UUID} AND dataset_name = {dataset_name: String}
"#;
    let datapoint_id = datapoint_id.to_string();
    let json_params = HashMap::from([
        ("datapoint_id", datapoint_id.as_str()),
        ("dataset_name", dataset_name.as_str()),
    ]);

    let chat_params = HashMap::from([
        ("datapoint_id", datapoint_id.as_str()),
        ("dataset_name", dataset_name.as_str()),
    ]);

    let json_future =
        clickhouse.run_query_synchronous(json_delete_query.to_string(), Some(&json_params));

    let chat_future =
        clickhouse.run_query_synchronous(chat_delete_query.to_string(), Some(&chat_params));

    let (json_result, chat_result) = tokio::join!(json_future, chat_future);

    json_result?;
    chat_result?;

    Ok(())
}

#[derive(Debug, Deserialize)]
pub struct ListDatapointsQueryParams {
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
    limit: Option<u32>,
    offset: Option<u32>,
) -> Result<Vec<Datapoint>, Error> {
    let query = r#"
    WITH dataset as (
        SELECT
            'chat' as type,
            dataset_name,
            function_name,
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
            staled_at
        FROM ChatInferenceDatapoint FINAL
        WHERE dataset_name = {dataset_name: String}
        AND staled_at IS NULL
        UNION ALL
        SELECT
            'json' as type,
            dataset_name,
            function_name,
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
            staled_at
        FROM JsonInferenceDatapoint FINAL
        WHERE dataset_name = {dataset_name: String}
        AND staled_at IS NULL
    )
    SELECT * FROM dataset
    ORDER BY id DESC
    LIMIT {limit: UInt32}
    OFFSET {offset: UInt32}
    FORMAT JSONEachRow
    "#;
    let limit = limit.unwrap_or(100);
    let offset = offset.unwrap_or(0);
    let limit_str = limit.to_string();
    let offset_str = offset.to_string();

    let params = HashMap::from([
        ("dataset_name", dataset_name.as_str()),
        ("limit", limit_str.as_str()),
        ("offset", offset_str.as_str()),
    ]);

    let result = clickhouse
        .run_query_synchronous(query.to_string(), Some(&params))
        .await?;
    if result.is_empty() {
        return Ok(vec![]);
    }
    let result_lines = result.trim().split("\n").collect::<Vec<&str>>();

    let datapoints: Result<Vec<ClickHouseDatapoint>, _> = result_lines
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

    let datapoints: Vec<Datapoint> = datapoints.into_iter().map(Datapoint::from).collect();

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
            if output_len != value.size {
                Err(Error::new(ErrorDetails::InvalidRequest {
                    message: format!(
                        "Output size ({output_len}) does not match number of datapoints ({size})",
                    ),
                }))
            } else {
                Ok(output)
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
    get_datapoint(
        path_params.dataset_name,
        path_params.datapoint_id,
        &app_state.clickhouse_connection_info,
    )
    .await
    .map(Json)
}

#[tracing::instrument(name = "get_datapoint", skip(clickhouse))]
pub async fn get_datapoint(
    dataset_name: String,
    datapoint_id: Uuid,
    clickhouse: &ClickHouseConnectionInfo,
) -> Result<Datapoint, Error> {
    let query = r#"
    WITH dataset as (
        SELECT
            'chat' as type,
            dataset_name,
            function_name,
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
            staled_at
        FROM ChatInferenceDatapoint FINAL
        WHERE dataset_name = {dataset_name: String}
        AND staled_at IS NULL
        UNION ALL
        SELECT
            'json' as type,
            dataset_name,
            function_name,
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
            staled_at
        FROM JsonInferenceDatapoint FINAL
        WHERE dataset_name = {dataset_name: String}
        AND staled_at IS NULL
    )
    SELECT * FROM dataset
    WHERE id = {datapoint_id: UUID}
    LIMIT 1
    FORMAT JSONEachRow
    "#;
    let datapoint_id_str = datapoint_id.to_string();
    let params = HashMap::from([
        ("dataset_name", dataset_name.as_str()),
        ("datapoint_id", datapoint_id_str.as_str()),
    ]);

    let result = clickhouse
        .run_query_synchronous(query.to_string(), Some(&params))
        .await?;
    if result.is_empty() {
        return Err(Error::new(ErrorDetails::DatapointNotFound {
            dataset_name,
            datapoint_id,
        }));
    }
    let datapoint: ClickHouseDatapoint = serde_json::from_str(&result).map_err(|e| {
        Error::new(ErrorDetails::ClickHouseDeserialization {
            message: format!("Failed to deserialize datapoint: {e}"),
        })
    })?;

    Ok(Datapoint::from(datapoint))
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
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatapointKind {
    Chat,
    Json,
}

impl DatapointKind {
    pub fn table_name(&self) -> &'static str {
        match self {
            DatapointKind::Chat => "ChatInferenceDatapoint",
            DatapointKind::Json => "JsonInferenceDatapoint",
        }
    }
}

#[derive(Debug, Serialize)]
pub struct InsertDatapointResponse {
    id: Uuid,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
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

    pub fn input(&self) -> &ResolvedInput {
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
}

/// These input datapoints are used as input typesby the `insert_datapoint` endpoint
/// The distinction here is that they do not include the `dataset_name` field,
/// which is instead specified as a path parameter.
/// We also use Input rather than ResolvedInput because the input is not resolved
/// when created.
/// We also do not allow users to specify the `id` or `episode_id` fields.
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatDatapointInsert {
    pub function_name: String,
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
    pub input: Input,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
    pub output_schema: Option<Value>, // Default to the function's output schema
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
pub struct ChatInferenceDatapoint {
    pub dataset_name: String,
    pub function_name: String,
    pub id: Uuid,
    pub episode_id: Option<Uuid>,
    pub input: ResolvedInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Vec<ContentBlockChatOutput>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_params: Option<ToolCallConfigDatabaseInsert>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<HashMap<String, String>>,
    #[serde(skip_serializing, default)] // this will become an object
    pub auxiliary: String,
    pub is_deleted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_inference_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub staled_at: Option<String>,
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
pub struct JsonInferenceDatapoint {
    pub dataset_name: String,
    pub function_name: String,
    pub id: Uuid,
    pub episode_id: Option<Uuid>,
    pub input: ResolvedInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<JsonInferenceOutput>,
    pub output_schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<HashMap<String, String>>,
    #[serde(skip_serializing, default)] // this will become an object
    pub auxiliary: String,
    pub is_deleted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_inference_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub staled_at: Option<String>,
}

/// We need to be able to deserialize Datapoints from both ClickHouse and
/// from strings. Since the strings will be properly serialized and we want
/// to be able to handle them naturally, we duplicated the types so that we
/// can effectively deserialize from ClickHouse as well.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClickHouseDatapoint {
    Chat(ClickHouseChatInferenceDatapoint),
    Json(ClickHouseJsonInferenceDatapoint),
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ClickHouseChatInferenceDatapoint {
    pub dataset_name: String,
    function_name: String,
    pub id: Uuid,
    episode_id: Option<Uuid>,
    #[serde(deserialize_with = "deserialize_json_string")]
    input: ResolvedInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(deserialize_with = "deserialize_optional_json_string")]
    output: Option<Vec<ContentBlockChatOutput>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(deserialize_with = "deserialize_optional_json_string")]
    tool_params: Option<ToolCallConfigDatabaseInsert>,
    tags: Option<HashMap<String, String>>,
    auxiliary: String,
    is_deleted: bool,
    source_inference_id: Option<Uuid>,
    staled_at: Option<String>,
}

impl From<ClickHouseChatInferenceDatapoint> for ChatInferenceDatapoint {
    fn from(value: ClickHouseChatInferenceDatapoint) -> Self {
        ChatInferenceDatapoint {
            dataset_name: value.dataset_name,
            function_name: value.function_name,
            id: value.id,
            episode_id: value.episode_id,
            input: value.input,
            output: value.output,
            tool_params: value.tool_params,
            tags: value.tags,
            auxiliary: value.auxiliary,
            is_deleted: value.is_deleted,
            source_inference_id: value.source_inference_id,
            staled_at: value.staled_at,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ClickHouseJsonInferenceDatapoint {
    pub dataset_name: String,
    function_name: String,
    pub id: Uuid,
    episode_id: Option<Uuid>,
    #[serde(deserialize_with = "deserialize_json_string")]
    input: ResolvedInput,
    #[serde(deserialize_with = "deserialize_optional_json_string")]
    output: Option<JsonInferenceOutput>,
    #[serde(deserialize_with = "deserialize_json_string")]
    output_schema: serde_json::Value,
    tags: Option<HashMap<String, String>>,
    auxiliary: String,
    is_deleted: bool,
    source_inference_id: Option<Uuid>,
    staled_at: Option<String>,
}

impl From<ClickHouseJsonInferenceDatapoint> for JsonInferenceDatapoint {
    fn from(value: ClickHouseJsonInferenceDatapoint) -> Self {
        JsonInferenceDatapoint {
            dataset_name: value.dataset_name,
            function_name: value.function_name,
            id: value.id,
            episode_id: value.episode_id,
            input: value.input,
            output: value.output,
            output_schema: value.output_schema,
            tags: value.tags,
            auxiliary: value.auxiliary,
            is_deleted: value.is_deleted,
            source_inference_id: value.source_inference_id,
            staled_at: value.staled_at,
        }
    }
}

impl From<ClickHouseDatapoint> for Datapoint {
    fn from(value: ClickHouseDatapoint) -> Self {
        match value {
            ClickHouseDatapoint::Chat(datapoint) => Datapoint::Chat(datapoint.into()),
            ClickHouseDatapoint::Json(datapoint) => Datapoint::Json(datapoint.into()),
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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticChatInferenceDatapoint {
    pub function_name: String,
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
    pub source_inference_id: Option<Uuid>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticJsonInferenceDatapoint {
    pub function_name: String,
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
    pub source_inference_id: Option<Uuid>,
}

fn validate_dataset_name(dataset_name: &str) -> Result<(), Error> {
    if dataset_name == "builder" || dataset_name.starts_with("tensorzero::") {
        Err(Error::new(ErrorDetails::InvalidDatasetName {
            dataset_name: dataset_name.to_string(),
        }))
    } else {
        Ok(())
    }
}

/// Puts a chat datapoint into ClickHouse but only
/// if it doesn't have a source_inference_id that already exists for this dataset.
/// Returns the number of rows written to ClickHouse
async fn put_deduped_chat_datapoints(
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

    let query = r#"
    INSERT INTO ChatInferenceDatapoint
        (
            dataset_name,
            function_name,
            id,
            episode_id,
            input,
            output,
            tool_params,
            tags,
            auxiliary,
            is_deleted,
            source_inference_id
        )
        SELECT
            new_data.dataset_name,
            new_data.function_name,
            new_data.id,
            new_data.episode_id,
            new_data.input,
            new_data.output,
            new_data.tool_params,
            new_data.tags,
            new_data.auxiliary,
            new_data.is_deleted,
            new_data.source_inference_id,
        FROM new_data
        LEFT JOIN ChatInferenceDatapoint AS existing FINAL
          ON new_data.dataset_name = existing.dataset_name
             AND new_data.function_name = existing.function_name
             AND new_data.source_inference_id = existing.source_inference_id
             AND new_data.id != existing.id -- this is to allow us to update the datapoint and keep the same source_inference_id
             AND existing.staled_at IS NULL
        WHERE existing.source_inference_id IS NULL
        "#;

    let external_data = ExternalDataInfo {
        external_data_name: "new_data".to_string(),
        structure: "dataset_name LowCardinality(String), function_name LowCardinality(String), id UUID, episode_id Nullable(UUID), input String, output Nullable(String), tool_params String, tags Map(String, String), auxiliary String, is_deleted Bool, source_inference_id Nullable(UUID)".to_string(),
        format: "JSONEachRow".to_string(),
        data: serialized_datapoints.join("\n"),
    };
    let result = clickhouse
        .run_query_with_external_data(external_data, query.to_string())
        .await?;
    Ok(result.metadata.written_rows)
}

async fn put_deduped_json_datapoints(
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

    let query = r#"
        INSERT INTO JsonInferenceDatapoint
        (
            dataset_name,
            function_name,
            id,
            episode_id,
            input,
            output,
            output_schema,
            tags,
            auxiliary,
            is_deleted,
            source_inference_id
        )
        SELECT
            new_data.dataset_name,
            new_data.function_name,
            new_data.id,
            new_data.episode_id,
            new_data.input,
            new_data.output,
            new_data.output_schema,
            new_data.tags,
            new_data.auxiliary,
            new_data.is_deleted,
            new_data.source_inference_id
        FROM new_data
        LEFT JOIN JsonInferenceDatapoint AS existing FINAL
          ON new_data.dataset_name = existing.dataset_name
             AND new_data.function_name = existing.function_name
             AND new_data.source_inference_id = existing.source_inference_id
             AND new_data.id != existing.id -- this is to allow us to update the datapoint and keep the same source_inference_id
             AND existing.staled_at IS NULL
        WHERE existing.source_inference_id IS NULL
        "#;

    let external_data = ExternalDataInfo {
        external_data_name: "new_data".to_string(),
        structure: "dataset_name LowCardinality(String), function_name LowCardinality(String), id UUID, episode_id Nullable(UUID), input String, output Nullable(String), output_schema Nullable(String), tags Map(String, String), auxiliary String, is_deleted Bool, source_inference_id Nullable(UUID)".to_string(),
        format: "JSONEachRow".to_string(),
        data: serialized_datapoints.join("\n"),
    };
    let result = clickhouse
        .run_query_with_external_data(external_data, query.to_string())
        .await?;
    Ok(result.metadata.written_rows)
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
            "auxiliary": ""
        }"#;

        let datapoint: SyntheticChatInferenceDatapoint = serde_json::from_str(json_str).unwrap();
        assert_eq!(datapoint.function_name, "test_function");
        assert_eq!(datapoint.output, None);
        assert_eq!(datapoint.tool_params, None);
        assert_eq!(datapoint.tags, None);
        assert_eq!(datapoint.auxiliary, "");
    }

    #[test]
    fn test_synthetic_chat_datapoint_with_some_output() {
        let json_str = r#"{
            "function_name": "test_function",
            "input": {"system": {"assistant_name": "Test"}, "messages": []},
            "output": [{"type": "text", "value": "Hello"}],
            "tool_params": {"tools_available": [], "tool_choice": "auto", "parallel_tool_calls": false},
            "tags": {"source": "test"},
            "auxiliary": "extra data"
        }"#;

        let datapoint: SyntheticChatInferenceDatapoint = serde_json::from_str(json_str).unwrap();
        assert_eq!(datapoint.function_name, "test_function");
        assert!(datapoint.output.is_some());
        assert!(datapoint.tool_params.is_some());
        assert_eq!(
            datapoint.tags,
            Some(HashMap::from([("source".to_string(), "test".to_string())]))
        );
        assert_eq!(datapoint.auxiliary, "extra data");
    }

    #[test]
    fn test_synthetic_json_datapoint_with_none_output() {
        let json_str = r#"{
            "function_name": "test_json_function",
            "input": {"system": {"assistant_name": "Test"}, "messages": []},
            "output": null,
            "output_schema": {},
            "tags": null,
            "auxiliary": ""
        }"#;

        let datapoint: SyntheticJsonInferenceDatapoint = serde_json::from_str(json_str).unwrap();
        assert_eq!(datapoint.function_name, "test_json_function");
        assert_eq!(datapoint.output, None);
        assert_eq!(datapoint.output_schema, json!({}));
        assert_eq!(datapoint.tags, None);
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
            "auxiliary": "extra data"
        }"#;

        let datapoint: SyntheticJsonInferenceDatapoint = serde_json::from_str(json_str).unwrap();
        assert_eq!(datapoint.function_name, "test_json_function");
        assert!(datapoint.output.is_some());
        assert_eq!(datapoint.output.as_ref().unwrap()["answer"], "Hello");
        assert_eq!(
            datapoint.tags,
            Some(HashMap::from([("source".to_string(), "test".to_string())]))
        );
        assert_eq!(datapoint.auxiliary, "extra data");
    }

    #[test]
    fn test_synthetic_json_datapoint_with_missing_output() {
        let json_str = r#"{
            "function_name": "test_json_function",
            "input": {"system": {"assistant_name": "Test"}, "messages": []},
            "output_schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
            "tags": {"source": "test"},
            "auxiliary": "extra data"
        }"#;

        let datapoint: SyntheticJsonInferenceDatapoint = serde_json::from_str(json_str).unwrap();
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
}
