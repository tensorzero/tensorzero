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
    function::FunctionConfig,
    gateway_util::{AppState, StructuredJson},
    inference::types::{
        batch::{deserialize_json_string, deserialize_optional_json_string},
        ChatInferenceDatabaseInsert, ContentBlockChatOutput, FetchContext, Input,
        JsonInferenceDatabaseInsert, JsonInferenceOutput, ResolvedInput,
    },
    tool::ToolCallConfigDatabaseInsert,
    uuid_util::validate_tensorzero_uuid,
};
use tracing::instrument;

pub const CLICKHOUSE_DATETIME_FORMAT: &str = "%Y-%m-%d %H:%M:%S%.6f";
use super::feedback::{
    validate_parse_demonstration, DemonstrationOutput, DynamicDemonstrationInfo,
};

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

async fn insert_from_existing(
    clickhouse: &ClickHouseConnectionInfo,
    path_params: CreatePathParams,
    existing: &ExistingInference,
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
                id: datapoint_id,
                episode_id: Some(inference.episode_id),
                input: inference.input,
                output,
                output_schema: inference.output_schema,
                tags: Some(inference.tags),
                auxiliary: "{}".to_string(),
                is_deleted: false,
            };
            clickhouse
                .write(&[datapoint], "JsonInferenceDatapoint")
                .await?;
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
                id: datapoint_id,
                episode_id: Some(inference.episode_id),
                input: inference.input,
                output,
                tool_params: inference.tool_params,
                tags: Some(inference.tags),
                auxiliary: "{}".to_string(),
                is_deleted: false,
            };
            clickhouse
                .write(&[datapoint], "ChatInferenceDatapoint")
                .await?;
        }
    }
    Ok(datapoint_id)
}

#[derive(Deserialize)]
struct WithFunctionName {
    function_name: String,
}

/// The handler for the POST `/datasets/:dataset/datapoints` endpoint.
/// This inserts a new datapoint into `ChatInferenceDatapoint`/`JsonInferenceDatapoint`
/// based on an existing inference (specified by `inference_id`).
///
/// The inference is mostly copied as-is, except for the 'output' field.
/// Based on the 'output' parameter, the output is copied, ignored, or fetched from a demonstration.
#[instrument(name = "create_datapoint", skip(app_state))]
pub async fn create_datapoint_handler(
    State(app_state): AppState,
    Path(path_params): Path<CreatePathParams>,
    StructuredJson(existing_inference): StructuredJson<ExistingInference>,
) -> Result<Json<CreateDatapointResponse>, Error> {
    let datapoint_id = insert_from_existing(
        &app_state.clickhouse_connection_info,
        path_params,
        &existing_inference,
    )
    .await?;
    Ok(Json(CreateDatapointResponse { id: datapoint_id }))
}

/// The handler for the PUT `/datasets/:dataset/datapoints/:id"` endpoint.
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
) -> Result<Json<CreateDatapointResponse>, Error> {
    validate_tensorzero_uuid(path_params.id, "Datapoint")?;
    let fetch_context = FetchContext {
        client: &app_state.http_client,
        object_store_info: &app_state.config.object_store_info,
    };
    let function_data: WithFunctionName = serde_json::from_value(params.clone()).map_err(|e| {
        Error::new(ErrorDetails::InvalidRequest {
            message: format!("Failed to deserialize `function_name``: {}", e),
        })
    })?;
    let function_config = app_state
        .config
        .get_function(&function_data.function_name)?;
    match &**function_config {
        FunctionConfig::Chat(_) => {
            let chat: SyntheticChatInferenceDatapoint =
                serde_json::from_value(params).map_err(|e| {
                    Error::new(ErrorDetails::InvalidRequest {
                        message: format!("Failed to deserialize chat datapoint: {}", e),
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
            let validated_output = validate_parse_demonstration(
                function_config,
                &chat.output,
                dynamic_demonstration_info,
            )
            .await?;

            let DemonstrationOutput::Chat(output) = validated_output else {
                return Err(Error::new(ErrorDetails::InternalError {
                    message: "Expected chat output from validate_parse_demonstration".to_string(),
                }));
            };

            let datapoint = ChatInferenceDatapoint {
                dataset_name: path_params.dataset,
                function_name: chat.function_name,
                id: path_params.id,
                episode_id: None,
                input: resolved_input,
                output: Some(output),
                tool_params: chat.tool_params,
                tags: chat.tags,
                auxiliary: chat.auxiliary,
                is_deleted: false,
            };
            app_state
                .clickhouse_connection_info
                .write(&[datapoint], "ChatInferenceDatapoint")
                .await?;
        }
        FunctionConfig::Json(_) => {
            let json: SyntheticJsonInferenceDatapoint =
                serde_json::from_value(params).map_err(|e| {
                    Error::new(ErrorDetails::InvalidRequest {
                        message: format!("Failed to deserialize JSON datapoint: {}", e),
                    })
                })?;
            let resolved_input = json.input.clone().resolve(&fetch_context).await?;
            function_config.validate_input(&json.input)?;
            let dynamic_demonstration_info = DynamicDemonstrationInfo::Json(json.output_schema);
            let validated_json = validate_parse_demonstration(
                function_config,
                &json.output,
                dynamic_demonstration_info,
            )
            .await?;
            let DemonstrationOutput::Json(json_out) = validated_json else {
                return Err(Error::new(ErrorDetails::InternalError {
                    message: "Expected JSON output from validate_parse_demonstration".to_string(),
                }));
            };
            let json_out: JsonInferenceOutput = serde_json::from_value(json_out).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Failed to deserialize validated JSON output: {}", e),
                })
            })?;
            let datapoint = JsonInferenceDatapoint {
                dataset_name: path_params.dataset,
                function_name: json.function_name,
                id: path_params.id,
                episode_id: None,
                input: resolved_input,
                output: Some(json_out),
                // We currently don't support creating synthetic datapoints with 'output_schema'
                output_schema: serde_json::Value::Object(Default::default()),
                tags: json.tags,
                auxiliary: json.auxiliary,
                is_deleted: false,
            };
            app_state
                .clickhouse_connection_info
                .write(&[datapoint], "JsonInferenceDatapoint")
                .await?;
        }
    }
    Ok(Json(CreateDatapointResponse { id: path_params.id }))
}

#[instrument(name = "delete_datapoint", skip(app_state))]
pub async fn delete_datapoint_handler(
    State(app_state): AppState,
    Path(path_params): Path<DeletePathParams>,
) -> Result<Json<DeleteDatapointResponse>, Error> {
    let datapoint = app_state.clickhouse_connection_info.run_query(
        "SELECT * FROM {table_name:Identifier} WHERE dataset_name={dataset_name:String} AND function_name={function_name:String} AND id = {id:String} ORDER BY updated_at DESC LIMIT 1 FORMAT JSONEachRow;".to_string(),
        Some(&HashMap::from([
            ("table_name", path_params.kind.table_name()),
            ("function_name", path_params.function.as_str()),
            ("dataset_name", path_params.dataset.as_str()),
            ("id", path_params.id.to_string().as_str())
        ]))).await?;

    if datapoint.is_empty() {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!("Datapoint not found with params {path_params:?}",),
        }));
    }

    let mut datapoint_json: serde_json::Value = serde_json::from_str(&datapoint).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to deserialize datapoint: {}", e),
        })
    })?;

    // We delete datapoints by writing a new row (which ClickHouse will merge)
    // with the 'is_deleted' and 'updated_at' fields modified.
    datapoint_json["is_deleted"] = serde_json::Value::Bool(true);
    datapoint_json["updated_at"] =
        format!("{}", chrono::Utc::now().format(CLICKHOUSE_DATETIME_FORMAT)).into();

    app_state
        .clickhouse_connection_info
        .write(&[datapoint_json], path_params.kind.table_name())
        .await?;

    Ok(Json(DeleteDatapointResponse {}))
}

#[derive(Debug, Serialize)]
pub struct DeleteDatapointResponse {}

#[derive(Debug, Deserialize)]
pub struct CreatePathParams {
    pub dataset: String,
}

#[derive(Debug, Deserialize)]
pub struct UpdatePathParams {
    pub dataset: String,
    pub id: Uuid,
}

#[derive(Debug, Deserialize)]
pub struct DeletePathParams {
    pub dataset: String,
    pub function: String,
    pub kind: DatapointKind,
    pub id: Uuid,
}

#[derive(Debug, Deserialize)]
pub struct ExistingInference {
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
    fn table_name(&self) -> &'static str {
        match self {
            DatapointKind::Chat => "ChatInferenceDatapoint",
            DatapointKind::Json => "JsonInferenceDatapoint",
        }
    }
}

#[derive(Debug, Serialize)]
pub struct CreateDatapointResponse {
    id: Uuid,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
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

    pub fn tool_call_config(&self) -> Option<&ToolCallConfigDatabaseInsert> {
        match self {
            Datapoint::ChatInference(datapoint) => datapoint.tool_params.as_ref(),
            Datapoint::JsonInference(_datapoint) => None,
        }
    }

    pub fn output_schema(&self) -> Option<&serde_json::Value> {
        match self {
            Datapoint::ChatInference(_datapoint) => None,
            Datapoint::JsonInference(datapoint) => Some(&datapoint.output_schema),
        }
    }

    pub fn id(&self) -> Uuid {
        match self {
            Datapoint::ChatInference(datapoint) => datapoint.id,
            Datapoint::JsonInference(datapoint) => datapoint.id,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
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
    pub auxiliary: String,
    pub is_deleted: bool,
}

#[derive(Debug, Deserialize, Serialize)]
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
    pub auxiliary: String,
    pub is_deleted: bool,
}

/// We need to be able to deserialize Datapoints from both ClickHouse and
/// from strings. Since the strings will be properly serialized and we want
/// to be able to handle them naturally, we duplicated the types so that we
/// can effectively deserialize from ClickHouse as well.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ClickHouseDatapoint {
    Chat(ClickHouseChatInferenceDatapoint),
    Json(ClickHouseJsonInferenceDatapoint),
}

#[derive(Debug, Deserialize)]
pub struct ClickHouseChatInferenceDatapoint {
    dataset_name: String,
    function_name: String,
    id: Uuid,
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
}

#[derive(Debug, Deserialize)]
pub struct ClickHouseJsonInferenceDatapoint {
    dataset_name: String,
    function_name: String,
    id: Uuid,
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
}

impl From<ClickHouseDatapoint> for Datapoint {
    fn from(value: ClickHouseDatapoint) -> Self {
        match value {
            ClickHouseDatapoint::Chat(datapoint) => {
                Datapoint::ChatInference(ChatInferenceDatapoint {
                    dataset_name: datapoint.dataset_name,
                    function_name: datapoint.function_name,
                    id: datapoint.id,
                    episode_id: datapoint.episode_id,
                    input: datapoint.input,
                    output: datapoint.output,
                    tool_params: datapoint.tool_params,
                    tags: datapoint.tags,
                    auxiliary: datapoint.auxiliary,
                    is_deleted: datapoint.is_deleted,
                })
            }
            ClickHouseDatapoint::Json(datapoint) => {
                Datapoint::JsonInference(JsonInferenceDatapoint {
                    dataset_name: datapoint.dataset_name,
                    function_name: datapoint.function_name,
                    id: datapoint.id,
                    episode_id: datapoint.episode_id,
                    input: datapoint.input,
                    output: datapoint.output,
                    output_schema: datapoint.output_schema,
                    tags: datapoint.tags,
                    auxiliary: datapoint.auxiliary,
                    is_deleted: datapoint.is_deleted,
                })
            }
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticChatInferenceDatapoint {
    pub function_name: String,
    pub input: Input,
    pub output: serde_json::Value,
    #[serde(default)]
    pub tool_params: Option<ToolCallConfigDatabaseInsert>,
    #[serde(default)]
    pub tags: Option<HashMap<String, String>>,
    #[serde(default)]
    pub auxiliary: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticJsonInferenceDatapoint {
    pub function_name: String,
    pub input: Input,
    pub output: serde_json::Value,
    pub output_schema: serde_json::Value,
    #[serde(default)]
    pub tags: Option<HashMap<String, String>>,
    #[serde(default)]
    pub auxiliary: String,
}
