use std::collections::HashMap;

use axum::{
    extract::{Path, State},
    Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    clickhouse::ClickHouseConnectionInfo,
    error::{Error, ErrorDetails},
    gateway_util::{AppState, StructuredJson},
    inference::types::batch::{deserialize_json_string, deserialize_optional_json_string},
    inference::types::{
        ChatInferenceDatabaseInsert, ContentBlockChatOutput, JsonInferenceDatabaseInsert,
        JsonInferenceOutput, ResolvedInput,
    },
    tool::ToolCallConfigDatabaseInsert,
};
use tracing::instrument;

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
    #[allow(dead_code)]
    id: Uuid,
    #[allow(dead_code)]
    inference_id: Uuid,
    value: String,
}

async fn query_demonstration(
    clickhouse: &ClickHouseConnectionInfo,
    inference_id: Uuid,
    page_size: u32,
) -> Result<Demonstration, Error> {
    let result = clickhouse
        .run_query(
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
            message: format!("No demonstration found for inference `{}`", inference_id),
        }));
    }
    let demonstration: Demonstration = serde_json::from_str(&result).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!(
                "Failed to deserialize demonstration ClickHouse response: {}",
                e
            ),
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
        .run_query(
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

  -- JSON-specific column
  IF(i.function_type = 'json', j.output_schema, '') AS output_schema,

  -- Timestamps & tags
  IF(i.function_type = 'chat',
   formatDateTime(c.timestamp, '%Y-%m-%dT%H:%i:%SZ'),
   formatDateTime(j.timestamp, '%Y-%m-%dT%H:%i:%SZ')
) AS timestamp,
  IF(i.function_type = 'chat', c.tags,      j.tags)      AS tags,

  -- Discriminator itself
  i.function_type

FROM InferenceById i
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
            message: format!("Inference `{}` not found", inference_id),
        }));
    }

    let inference_data: TaggedInferenceDatabaseInsert =
        serde_json::from_str(&result).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to deserialize inference data: {}", e),
            })
        })?;
    Ok(inference_data)
}

/// The handler for the `/datasets/:dataset/datapoints` endpoint.
/// This inserts a new datapoint into `ChatInferenceDataset`/`JsonInferenceDataset`
/// based on an existing inference (specified by `inference_id`).
///
/// The inference is mostly copied as-is, except for the 'output' field.
/// Based on the 'output' parameter, the output is copied, ignored, or fetched from a demonstration.
#[instrument(name = "create_datapoint", skip(app_state))]
pub async fn create_datapoint_handler(
    State(app_state): AppState,
    Path(path_params): Path<PathParams>,
    StructuredJson(params): StructuredJson<CreateDatapointParams>,
) -> Result<Json<CreateDatapointResponse>, Error> {
    let inference_data =
        query_inference_for_datapoint(&app_state.clickhouse_connection_info, params.inference_id)
            .await?;

    match inference_data {
        TaggedInferenceDatabaseInsert::Json(inference) => {
            let output = match params.output {
                OutputKind::Inherit => Some(inference.output),
                OutputKind::Demonstration => {
                    let demonstration = query_demonstration(
                        &app_state.clickhouse_connection_info,
                        params.inference_id,
                        1,
                    )
                    .await?;
                    Some(serde_json::from_str(&demonstration.value).map_err(|e| {
                        Error::new(ErrorDetails::Serialization {
                            message: format!(
                                "Failed to deserialize JSON demonstration output: {}",
                                e
                            ),
                        })
                    })?)
                }
                OutputKind::None => None,
            };
            let datapoint = JsonInferenceDatapoint {
                dataset_name: path_params.dataset,
                function_name: inference.function_name,
                id: inference.id,
                episode_id: inference.episode_id,
                input: inference.input,
                output,
                output_schema: inference.output_schema,
                tags: Some(inference.tags),
                auxiliary: "{}".to_string(),
                is_deleted: false,
            };
            app_state
                .clickhouse_connection_info
                .write(&[datapoint], "JsonInferenceDataset")
                .await?;
        }
        TaggedInferenceDatabaseInsert::Chat(inference) => {
            let output = match params.output {
                OutputKind::Inherit => Some(inference.output),
                OutputKind::Demonstration => {
                    let demonstration = query_demonstration(
                        &app_state.clickhouse_connection_info,
                        params.inference_id,
                        1,
                    )
                    .await?;
                    Some(serde_json::from_str(&demonstration.value).map_err(|e| {
                        Error::new(ErrorDetails::Serialization {
                            message: format!(
                                "Failed to deserialize chat demonstration output: {}",
                                e
                            ),
                        })
                    })?)
                }
                OutputKind::None => None,
            };
            let datapoint = ChatInferenceDatapoint {
                dataset_name: path_params.dataset,
                function_name: inference.function_name,
                id: inference.id,
                episode_id: inference.episode_id,
                input: inference.input,
                output,
                tool_params: inference.tool_params,
                tags: Some(inference.tags),
                auxiliary: "{}".to_string(),
                is_deleted: false,
            };
            app_state
                .clickhouse_connection_info
                .write(&[datapoint], "ChatInferenceDataset")
                .await?;
        }
    }
    Ok(Json(CreateDatapointResponse {}))
}

#[derive(Debug, Deserialize)]
pub struct PathParams {
    pub dataset: String,
}

#[derive(Debug, Deserialize)]
pub struct CreateDatapointParams {
    pub output: OutputKind,
    pub inference_id: Uuid,
}

#[derive(Debug, Serialize)]
pub struct CreateDatapointResponse {}

#[derive(Debug, Deserialize, Serialize)]
pub enum Datapoint {
    ChatInference(ChatInferenceDatapoint),
    JsonInference(JsonInferenceDatapoint),
}

impl Datapoint {
    pub fn dataset_name(&self) -> &str {
        match self {
            Datapoint::ChatInference(datapoint) => &datapoint.dataset_name,
            Datapoint::JsonInference(datapoint) => &datapoint.dataset_name,
        }
    }

    pub fn input(&self) -> &ResolvedInput {
        match self {
            Datapoint::ChatInference(datapoint) => &datapoint.input,
            Datapoint::JsonInference(datapoint) => &datapoint.input,
        }
    }
}
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatInferenceDatapoint {
    pub dataset_name: String,
    pub function_name: String,
    pub id: Uuid,
    pub episode_id: Uuid,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub input: ResolvedInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(deserialize_with = "deserialize_optional_json_string")]
    pub output: Option<Vec<ContentBlockChatOutput>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(deserialize_with = "deserialize_optional_json_string")]
    pub tool_params: Option<ToolCallConfigDatabaseInsert>,
    pub tags: Option<HashMap<String, String>>,
    pub auxiliary: String,
    pub is_deleted: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct JsonInferenceDatapoint {
    pub dataset_name: String,
    pub function_name: String,
    pub id: Uuid,
    pub episode_id: Uuid,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub input: ResolvedInput,
    #[serde(deserialize_with = "deserialize_optional_json_string")]
    pub output: Option<JsonInferenceOutput>,
    pub output_schema: serde_json::Value,
    pub tags: Option<HashMap<String, String>>,
    pub auxiliary: String,
    pub is_deleted: bool,
}
