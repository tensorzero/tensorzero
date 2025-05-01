use std::{collections::HashMap, sync::Arc};

use axum::{
    extract::{Path, State},
    Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    clickhouse::{ClickHouseConnectionInfo, ExternalDataInfo},
    config_parser::Config,
    error::{Error, ErrorDetails},
    function::{FunctionConfig, FunctionConfigChat},
    gateway_util::{AppState, StructuredJson},
    inference::types::{
        batch::{deserialize_json_string, deserialize_optional_json_string},
        ChatInferenceDatabaseInsert, ContentBlockChatOutput, FetchContext, Input,
        JsonInferenceDatabaseInsert, JsonInferenceOutput, ResolvedInput,
    },
    tool::{ToolCallConfigDatabaseInsert, ToolChoice},
    uuid_util::validate_tensorzero_uuid,
};
use tracing::instrument;

pub const CLICKHOUSE_DATETIME_FORMAT: &str = "%Y-%m-%d %H:%M:%S%.6f";
use super::{
    feedback::{validate_parse_demonstration, DemonstrationOutput, DynamicDemonstrationInfo},
    inference::DEFAULT_FUNCTION_NAME,
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
                                "Failed to deserialize JSON demonstration output: {e}"
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
                source_inference_id: Some(existing.inference_id),
                staled_at: None,
            };
            let rows_written = put_deduped_json_datapoint(clickhouse, &datapoint).await?;
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
                source_inference_id: Some(existing.inference_id),
                staled_at: None,
            };
            let rows_written = put_deduped_chat_datapoint(clickhouse, &datapoint).await?;
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
    validate_dataset_name(&path_params.dataset)?;
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
    validate_dataset_name(&path_params.dataset)?;
    let fetch_context = FetchContext {
        client: &app_state.http_client,
        object_store_info: &app_state.config.object_store_info,
    };
    let function_data: WithFunctionName = serde_json::from_value(params.clone()).map_err(|e| {
        Error::new(ErrorDetails::InvalidRequest {
            message: format!("Failed to deserialize `function_name`: {e}"),
        })
    })?;
    let function_config =
        get_possibly_default_function(&function_data.function_name, &app_state.config)?;

    match *function_config {
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
                dataset_name: path_params.dataset,
                function_name: chat.function_name,
                id: path_params.id,
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
                put_deduped_chat_datapoint(&app_state.clickhouse_connection_info, &datapoint)
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
                dataset_name: path_params.dataset,
                function_name: json.function_name,
                id: path_params.id,
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
                put_deduped_json_datapoint(&app_state.clickhouse_connection_info, &datapoint)
                    .await?;
            if rows_written == 0 {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: "Datapoint with this source_inference_id already exists".to_string(),
                }));
            }
        }
    }
    Ok(Json(CreateDatapointResponse { id: path_params.id }))
}

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
    pub fn table_name(&self) -> &'static str {
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_inference_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub staled_at: Option<String>,
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
#[serde(untagged)]
pub enum ClickHouseDatapoint {
    Chat(ClickHouseChatInferenceDatapoint),
    Json(ClickHouseJsonInferenceDatapoint),
}

#[derive(Debug, Deserialize)]
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

#[derive(Debug, Deserialize)]
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
                    source_inference_id: datapoint.source_inference_id,
                    staled_at: datapoint.staled_at,
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
                    source_inference_id: datapoint.source_inference_id,
                    staled_at: datapoint.staled_at,
                })
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
    #[serde(default)]
    pub auxiliary: String,
    #[serde(default)]
    pub source_inference_id: Option<Uuid>,
}

fn get_possibly_default_function(
    function_name: &str,
    config: &Config,
) -> Result<Arc<FunctionConfig>, Error> {
    if function_name == DEFAULT_FUNCTION_NAME {
        Ok(Arc::new(FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            tools: vec![],
            tool_choice: ToolChoice::None,
            parallel_tool_calls: None,
            description: None,
        })))
    } else {
        config.get_function(function_name).cloned()
    }
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
async fn put_deduped_chat_datapoint(
    clickhouse: &ClickHouseConnectionInfo,
    datapoint: &ChatInferenceDatapoint,
) -> Result<u64, Error> {
    let serialized_datapoint = serde_json::to_string(datapoint).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize datapoint: {e}"),
        })
    })?;

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
        data: serialized_datapoint,
    };
    let result = clickhouse
        .run_query_with_external_data(external_data, query.to_string())
        .await?;
    Ok(result.metadata.written_rows)
}

async fn put_deduped_json_datapoint(
    clickhouse: &ClickHouseConnectionInfo,
    datapoint: &JsonInferenceDatapoint,
) -> Result<u64, Error> {
    let serialized_datapoint = serde_json::to_string(datapoint).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize datapoint: {e}"),
        })
    })?;

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
        data: serialized_datapoint,
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
