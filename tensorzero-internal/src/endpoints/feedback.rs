use std::cmp::max;
use std::collections::HashMap;
use std::time::Duration;

use axum::extract::State;
use axum::{debug_handler, Json};
use metrics::counter;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::time::Instant;
use tracing::instrument;
use uuid::Uuid;

use crate::clickhouse::ClickHouseConnectionInfo;
use crate::config_parser::{Config, MetricConfigLevel, MetricConfigType};
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfig;
use crate::gateway_util::{AppState, AppStateData, StructuredJson};
use crate::inference::types::batch::deserialize_optional_json_string;
use crate::inference::types::{
    parse_chat_output, ContentBlockChatOutput, ContentBlockOutput, Text,
};
use crate::jsonschema_util::JSONSchemaFromPath;
use crate::tool::{ToolCall, ToolCallConfig, ToolCallConfigDatabaseInsert};
use crate::uuid_util::uuid_elapsed;

use super::validate_tags;

/// There is a potential issue here where if we write an inference and then immediately write feedback for it,
/// we might not be able to find the inference in the database because it hasn't been written yet.
///
/// This is the amount of time we want to wait after the target was supposed to have been written
/// before we decide that the target was actually not written because we can't find it in the database.
const FEEDBACK_COOLDOWN_PERIOD: Duration = Duration::from_secs(5);
/// Since we can't be sure that an inference actually completed when the ID says it was
/// (the ID is generated at the start of the inference), we wait a minimum amount of time
/// before we decide that the target was actually not written because we can't find it in the database.
const FEEDBACK_MINIMUM_WAIT_TIME: Duration = Duration::from_millis(1200);

/// The expected payload is a JSON object with the following fields:
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    // the episode ID client is providing feedback for (either this or `inference_id` must be set but not both)
    pub episode_id: Option<Uuid>,
    // the inference ID client is providing feedback for (either this or `episode_id` must be set but not both)
    pub inference_id: Option<Uuid>,
    // the name of the Metric to provide feedback for (this can always also be "comment" or "demonstration")
    pub metric_name: String,
    // the value of the feedback being provided
    pub value: Value,
    // if true, the feedback will be internal and validation of tags will be skipped
    #[serde(default)]
    pub internal: bool,
    // the tags to add to the feedback
    #[serde(default)]
    pub tags: HashMap<String, String>,
    // if true, the feedback will not be stored
    pub dryrun: Option<bool>,
}

#[derive(Debug, PartialEq)]
enum FeedbackType {
    Comment,
    Demonstration,
    Float,
    Boolean,
}

impl From<&MetricConfigType> for FeedbackType {
    fn from(value: &MetricConfigType) -> Self {
        match value {
            MetricConfigType::Float => FeedbackType::Float,
            MetricConfigType::Boolean => FeedbackType::Boolean,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FeedbackResponse {
    pub feedback_id: Uuid,
}

/// A handler for the feedback endpoint
#[instrument(name="feedback",
  skip_all,
  fields(
    metric_name = %params.metric_name,
  )
)]
#[debug_handler(state = AppStateData)]
pub async fn feedback_handler(
    State(app_state): AppState,
    StructuredJson(params): StructuredJson<Params>,
) -> Result<Json<FeedbackResponse>, Error> {
    feedback(app_state, params).await
}

// Helper function to avoid requiring axum types in the client
pub async fn feedback(
    AppStateData {
        config,
        clickhouse_connection_info,
        ..
    }: AppStateData,
    params: Params,
) -> Result<Json<FeedbackResponse>, Error> {
    validate_tags(&params.tags, params.internal)?;
    // Get the metric config or return an error if it doesn't exist
    let feedback_metadata = get_feedback_metadata(
        &config,
        &params.metric_name,
        params.episode_id,
        params.inference_id,
    )?;

    let feedback_id = Uuid::now_v7();

    let dryrun = params.dryrun.unwrap_or(false);

    // Increment the request count if we're not in dryrun mode
    if !dryrun {
        counter!(
            "request_count",
            "endpoint" => "feedback",
            "metric_name" => params.metric_name.to_string()
        )
        .increment(1);
    }

    match feedback_metadata.r#type {
        FeedbackType::Comment => {
            write_comment(
                clickhouse_connection_info,
                &params,
                feedback_metadata.target_id,
                feedback_metadata.level,
                feedback_id,
                dryrun,
            )
            .await?
        }
        FeedbackType::Demonstration => {
            write_demonstration(
                clickhouse_connection_info,
                &config,
                &params,
                feedback_metadata.target_id,
                feedback_id,
                dryrun,
            )
            .await?
        }
        FeedbackType::Float => {
            write_float(
                clickhouse_connection_info,
                &config,
                &params,
                feedback_metadata.target_id,
                feedback_id,
                dryrun,
            )
            .await?
        }
        FeedbackType::Boolean => {
            write_boolean(
                clickhouse_connection_info,
                &config,
                &params,
                feedback_metadata.target_id,
                feedback_id,
                dryrun,
            )
            .await?
        }
    }

    Ok(Json(FeedbackResponse { feedback_id }))
}

#[derive(Debug)]
struct FeedbackMetadata<'a> {
    r#type: FeedbackType,
    level: &'a MetricConfigLevel,
    target_id: Uuid,
}

fn get_feedback_metadata<'a>(
    config: &'a Config,
    metric_name: &str,
    episode_id: Option<Uuid>,
    inference_id: Option<Uuid>,
) -> Result<FeedbackMetadata<'a>, Error> {
    if let (Some(_), Some(_)) = (episode_id, inference_id) {
        return Err(ErrorDetails::InvalidRequest {
            message: "Both episode_id and inference_id cannot be provided".to_string(),
        }
        .into());
    }
    let metric = config.get_metric(metric_name);
    let feedback_type = match metric.as_ref() {
        Some(metric) => {
            let feedback_type: FeedbackType = (&metric.r#type).into();
            Ok(feedback_type)
        }
        None => match metric_name {
            "comment" => Ok(FeedbackType::Comment),
            "demonstration" => Ok(FeedbackType::Demonstration),
            _ => Err(Error::new(ErrorDetails::UnknownMetric {
                name: metric_name.to_string(),
            })),
        },
    }?;
    let feedback_level = match metric {
        Some(metric) => Ok(&metric.level),
        None => match feedback_type {
            FeedbackType::Demonstration => Ok(&MetricConfigLevel::Inference),
            _ => match (inference_id, episode_id) {
                (Some(_), None) => Ok(&MetricConfigLevel::Inference),
                (None, Some(_)) => Ok(&MetricConfigLevel::Episode),
                _ => Err(Error::new(ErrorDetails::InvalidRequest {
                    message: "Exactly one of inference_id or episode_id must be provided"
                        .to_string(),
                })),
            },
        },
    }?;
    let target_id = match feedback_level {
        MetricConfigLevel::Inference => inference_id,
        MetricConfigLevel::Episode => episode_id,
    }
    .ok_or_else(|| ErrorDetails::InvalidRequest {
        message: format!(
            r#"Correct ID was not provided for feedback level "{}"."#,
            feedback_level
        ),
    })?;
    Ok(FeedbackMetadata {
        r#type: feedback_type,
        level: feedback_level,
        target_id,
    })
}

async fn write_comment(
    connection_info: ClickHouseConnectionInfo,
    params: &Params,
    target_id: Uuid,
    level: &MetricConfigLevel,
    feedback_id: Uuid,
    dryrun: bool,
) -> Result<(), Error> {
    let Params { value, tags, .. } = params;
    // Verify that the function name exists.
    let _ = throttled_get_function_name(&connection_info, level, &target_id).await?;
    let value = value.as_str().ok_or_else(|| ErrorDetails::InvalidRequest {
        message: "Feedback value for a comment must be a string".to_string(),
    })?;
    let payload = json!({
        "target_type": level,
        "target_id": target_id,
        "value": value,
        "id": feedback_id,
        "tags": tags
    });
    if !dryrun {
        tokio::spawn(async move {
            let _ = connection_info.write(&[payload], "CommentFeedback").await;
        });
    }
    Ok(())
}

async fn write_demonstration(
    connection_info: ClickHouseConnectionInfo,
    config: &Config<'_>,
    params: &Params,
    inference_id: Uuid,
    feedback_id: Uuid,
    dryrun: bool,
) -> Result<(), Error> {
    let Params { value, tags, .. } = params;
    let function_name = throttled_get_function_name(
        &connection_info,
        &MetricConfigLevel::Inference,
        &inference_id,
    )
    .await?;
    let function_config = config.get_function(&function_name)?;
    let dynamic_demonstration_info = get_dynamic_demonstration_info(
        &connection_info,
        inference_id,
        &function_name,
        function_config,
    )
    .await?;
    let parsed_value =
        validate_parse_demonstration(function_config, value, dynamic_demonstration_info).await?;
    let string_value = serde_json::to_string(&parsed_value).map_err(|e| {
        Error::new(ErrorDetails::InvalidRequest {
            message: format!("Failed to serialize parsed value to json: {}", e),
        })
    })?;
    let payload = json!({"inference_id": inference_id, "value": string_value, "id": feedback_id, "tags": tags});
    if !dryrun {
        tokio::spawn(async move {
            let _ = connection_info
                .write(&[payload], "DemonstrationFeedback")
                .await;
        });
    }
    Ok(())
}

async fn write_float(
    connection_info: ClickHouseConnectionInfo,
    config: &Config<'_>,
    params: &Params,
    target_id: Uuid,
    feedback_id: Uuid,
    dryrun: bool,
) -> Result<(), Error> {
    let Params {
        metric_name,
        value,
        tags,
        ..
    } = params;
    let metric_config: &crate::config_parser::MetricConfig =
        config.get_metric_or_err(metric_name)?;
    // Verify that the function name exists.
    let _ = throttled_get_function_name(&connection_info, &metric_config.level, &target_id).await?;

    let value = value.as_f64().ok_or_else(|| {
        Error::new(ErrorDetails::InvalidRequest {
            message: format!("Feedback value for metric `{metric_name}` must be a number"),
        })
    })?;
    let payload = json!({"target_id": target_id, "value": value, "metric_name": metric_name, "id": feedback_id, "tags": tags});
    if !dryrun {
        tokio::spawn(async move {
            let _ = connection_info
                .write(&[payload], "FloatMetricFeedback")
                .await;
        });
    }
    Ok(())
}

async fn write_boolean(
    connection_info: ClickHouseConnectionInfo,
    config: &Config<'_>,
    params: &Params,
    target_id: Uuid,
    feedback_id: Uuid,
    dryrun: bool,
) -> Result<(), Error> {
    let Params {
        metric_name,
        value,
        tags,
        ..
    } = params;
    let metric_config = config.get_metric_or_err(metric_name)?;
    // Verify that the function name exists.
    let _ = throttled_get_function_name(&connection_info, &metric_config.level, &target_id).await?;
    let value = value.as_bool().ok_or_else(|| {
        Error::new(ErrorDetails::InvalidRequest {
            message: format!("Feedback value for metric `{metric_name}` must be a boolean"),
        })
    })?;
    let payload = json!({"target_id": target_id, "value": value, "metric_name": metric_name, "id": feedback_id, "tags": tags});
    if !dryrun {
        tokio::spawn(async move {
            let _ = connection_info
                .write(&[payload], "BooleanMetricFeedback")
                .await;
        });
    }
    Ok(())
}

/// This function throttles the check that an id is valid if it was created very recently.
/// This is to avoid a race condition where the id was created (e.g. an inference was made)
/// but the feedback is received before the id is written to the database.
///
/// We compute an amount of time to wait by max(FEEDBACK_COOLDOWN_PERIOD - elapsed_from_target_id, FEEDBACK_MINIMUM_WAIT_TIME)
/// We then poll every 500ms until that time has passed.
/// If the time has passed and the id is still not found, we return an error.
async fn throttled_get_function_name(
    connection_info: &ClickHouseConnectionInfo,
    metric_config_level: &MetricConfigLevel,
    target_id: &Uuid,
) -> Result<String, Error> {
    // Compute how long ago the target_id was created.
    let elapsed = uuid_elapsed(target_id)?;

    // Calculate the remaining cooldown (which may be zero) and ensure we wait at least FEEDBACK_MINIMUM_WAIT_TIME.
    let wait_time = max(
        FEEDBACK_COOLDOWN_PERIOD.saturating_sub(elapsed),
        FEEDBACK_MINIMUM_WAIT_TIME,
    );
    let deadline = Instant::now() + wait_time;

    // Poll every 500ms until the deadline is reached.
    loop {
        match get_function_name(connection_info, metric_config_level, target_id).await {
            Ok(identifier) => return Ok(identifier),
            Err(err) => {
                if Instant::now() >= deadline {
                    return Err(err);
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

/// Retrieves the function name associated with a given `target_id` of the inference or episode.
///
/// # Arguments
///
/// * `connection_info` - Connection details for the ClickHouse database.
/// * `metric_config_level` - The level of metric configuration, either `Inference` or `Episode`.
/// * `target_id` - The UUID of the target to be validated and retrieved.
///
/// # Returns
///
/// * On success:
///   - Returns the `function_name` associated with the `target_id`.
/// * On failure:
///   - Returns an `Error` if the `target_id` is invalid or does not exist.
async fn get_function_name(
    connection_info: &ClickHouseConnectionInfo,
    metric_config_level: &MetricConfigLevel,
    target_id: &Uuid,
) -> Result<String, Error> {
    let table_name = match metric_config_level {
        MetricConfigLevel::Inference => "InferenceById",
        MetricConfigLevel::Episode => "InferenceByEpisodeId",
    };
    let identifier_type = match metric_config_level {
        MetricConfigLevel::Inference => "Inference",
        MetricConfigLevel::Episode => "Episode",
    };
    let identifier_key = match metric_config_level {
        MetricConfigLevel::Inference => "id_uint",
        MetricConfigLevel::Episode => "episode_id_uint",
    };
    let query = format!(
        "SELECT function_name FROM {} FINAL WHERE {} = toUInt128(toUUID('{}'))",
        table_name, identifier_key, target_id
    );
    let function_name = connection_info
        .run_query(query, None)
        .await?
        .trim()
        .to_string();
    if function_name.is_empty() {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!("{identifier_type} ID: {target_id} does not exist"),
        }));
    };
    Ok(function_name)
}

#[derive(Debug, Deserialize, PartialEq)]
struct DemonstrationToolCall {
    name: String,
    arguments: Value,
}

impl TryFrom<DemonstrationToolCall> for ToolCall {
    type Error = Error;
    fn try_from(value: DemonstrationToolCall) -> Result<Self, Self::Error> {
        Ok(ToolCall {
            name: value.name,
            arguments: serde_json::to_string(&value.arguments).map_err(|e| {
                Error::new(ErrorDetails::InvalidRequest {
                    message: format!(
                        "Failed to serialize demonstration tool call arguments: {}",
                        e
                    ),
                })
            })?,
            id: "".to_string(),
        })
    }
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
enum DemonstrationContentBlock {
    Text(Text),
    ToolCall(DemonstrationToolCall),
}

impl TryFrom<DemonstrationContentBlock> for ContentBlockOutput {
    type Error = Error;
    fn try_from(value: DemonstrationContentBlock) -> Result<Self, Self::Error> {
        match value {
            DemonstrationContentBlock::Text(text) => Ok(ContentBlockOutput::Text(text)),
            DemonstrationContentBlock::ToolCall(tool_call) => {
                Ok(ContentBlockOutput::ToolCall(tool_call.try_into()?))
            }
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum DemonstrationOutput {
    Chat(Vec<ContentBlockChatOutput>),
    Json(Value),
}

// Validates that the demonstration is correct.
// For chat functions, the value should be a string or a list of valid content blocks.
// For json functions, the value is validated against the output schema. If it passes,
// we construct the usual {"raw": str, "parsed": parsed_value} object, serialize it, and return it.
pub async fn validate_parse_demonstration(
    function_config: &FunctionConfig,
    value: &Value,
    dynamic_demonstration_info: DynamicDemonstrationInfo,
) -> Result<DemonstrationOutput, Error> {
    match (function_config, dynamic_demonstration_info) {
        (FunctionConfig::Chat(_), DynamicDemonstrationInfo::Chat(tool_call_config)) => {
            // For chat functions, the value should either be a string or a list of valid content blocks.
            let content_blocks = match value {
                Value::String(s) => {
                    vec![DemonstrationContentBlock::Text(Text {
                        text: s.to_string(),
                    })]
                }
                Value::Array(content_blocks) => content_blocks
                    .iter()
                    .map(|block| {
                        serde_json::from_value::<DemonstrationContentBlock>(block.clone()).map_err(
                            |e| {
                                Error::new(ErrorDetails::InvalidRequest {
                                    message: format!("Invalid demonstration content block: {}", e),
                                })
                            },
                        )
                    })
                    .collect::<Result<Vec<DemonstrationContentBlock>, Error>>()?,
                _ => {
                    return Err(ErrorDetails::InvalidRequest {
                        message: "Demonstration must be a string or an array of content blocks"
                            .to_string(),
                    }
                    .into());
                }
            };
            let content_blocks: Vec<ContentBlockOutput> = content_blocks
                .into_iter()
                .map(|block| block.try_into())
                .collect::<Result<Vec<ContentBlockOutput>, Error>>()?;
            let parsed_value = parse_chat_output(content_blocks, Some(&tool_call_config)).await;
            for block in &parsed_value {
                if let ContentBlockChatOutput::ToolCall(tool_call) = block {
                    if tool_call.name.is_none() {
                        return Err(ErrorDetails::InvalidRequest {
                            message: "Demonstration contains invalid tool name".to_string(),
                        }
                        .into());
                    }
                    if tool_call.arguments.is_none() {
                        return Err(ErrorDetails::InvalidRequest {
                            message: "Demonstration contains invalid tool call arguments"
                                .to_string(),
                        }
                        .into());
                    }
                }
            }
            Ok(DemonstrationOutput::Chat(parsed_value))
        }
        (FunctionConfig::Json(_), DynamicDemonstrationInfo::Json(output_schema)) => {
            // For json functions, the value should be a valid json object.
            JSONSchemaFromPath::from_value(&output_schema)?
                .validate(value)
                .map_err(|e| {
                    Error::new(ErrorDetails::InvalidRequest {
                        message: format!(
                            "Demonstration does not fit function output schema: {}",
                            e
                        ),
                    })
                })?;
            let raw = serde_json::to_string(value).map_err(|e| {
                Error::new(ErrorDetails::InvalidRequest {
                    message: format!("Failed to serialize demonstration to json: {}", e),
                })
            })?;
            let json_inference_response = json!({"raw": raw, "parsed": value});
            Ok(DemonstrationOutput::Json(json_inference_response))
        }
        _ => Err(ErrorDetails::Inference { message: "The DynamicDemonstrationInfo does not match the function type. This should never happen. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports.".to_string()}.into())
    }
}

/// Represents the different types of dynamic demonstration information that can be retrieved
#[derive(Debug)]
pub enum DynamicDemonstrationInfo {
    Chat(ToolCallConfig),
    Json(Value),
}

#[derive(Debug, Deserialize)]
struct ToolParamsResult {
    #[serde(deserialize_with = "deserialize_optional_json_string")]
    tool_params: Option<ToolCallConfigDatabaseInsert>,
}

/// In order to properly validate demonstration data we need to fetch the information that was
/// passed to the inference at runtime.
/// If we don't do this then we might allow some e.g. tool calls or output schemas that would not
/// have been valid. Similarly, we might reject some tools or output schemas that were actually
/// valid.
/// This function grabs either the tool call or output schema information that was used at the
/// time of the actual inference in order to validate the demonstration data.
async fn get_dynamic_demonstration_info(
    clickhouse_client: &ClickHouseConnectionInfo,
    inference_id: Uuid,
    function_name: &str,
    function_config: &FunctionConfig,
) -> Result<DynamicDemonstrationInfo, Error> {
    match function_config {
        FunctionConfig::Chat(..) => {
            let parameterized_query = "SELECT tool_params FROM ChatInference WHERE function_name={function_name:String} and id={inference_id:String} FORMAT JSONEachRow".to_string();
            let result = clickhouse_client
                .run_query(
                    parameterized_query,
                    Some(&HashMap::from([
                        ("function_name", function_name),
                        ("inference_id", &inference_id.to_string()),
                    ])),
                )
                .await?;

            let tool_params_result =
                serde_json::from_str::<ToolParamsResult>(&result).map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseQuery {
                        message: format!("Failed to parse demonstration result: {}", e),
                    })
                })?;

            Ok(DynamicDemonstrationInfo::Chat(
                // If the tool params are not present in the database, we use the default tool params (empty tools).
                // This is consistent with how they are serialized at inference time.
                tool_params_result
                    .tool_params
                    .map(|x| x.into())
                    .unwrap_or_default(),
            ))
        }
        FunctionConfig::Json(..) => {
            let parameterized_query = "SELECT output_schema FROM JsonInference WHERE function_name={function_name:String} and id={inference_id:String} FORMAT JSONEachRow".to_string();
            let result = clickhouse_client
                .run_query(
                    parameterized_query,
                    Some(&HashMap::from([
                        ("function_name", function_name),
                        ("inference_id", &inference_id.to_string()),
                    ])),
                )
                .await?;
            let result_value = serde_json::from_str::<Value>(&result).map_err(|e| {
                Error::new(ErrorDetails::ClickHouseQuery {
                    message: format!("Failed to parse demonstration result: {}", e),
                })
            })?;
            let output_schema_str = result_value
                .get("output_schema")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    Error::new(ErrorDetails::ClickHouseQuery {
                        message: "Failed to get output schema from demonstration result"
                            .to_string(),
                    })
                })?;

            let output_schema = serde_json::from_str::<Value>(output_schema_str).map_err(|e| {
                Error::new(ErrorDetails::ClickHouseQuery {
                    message: format!("Failed to parse output schema: {}", e),
                })
            })?;
            Ok(DynamicDemonstrationInfo::Json(output_schema))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    use crate::config_parser::{Config, MetricConfig, MetricConfigOptimize};
    use crate::function::{FunctionConfigChat, FunctionConfigJson};
    use crate::jsonschema_util::JSONSchemaFromPath;
    use crate::testing::get_unit_test_app_state_data;
    use crate::tool::{StaticToolConfig, ToolCallOutput, ToolChoice, ToolConfig};

    #[tokio::test]
    async fn test_get_feedback_metadata() {
        // Case 1.1: Metric exists with name, is inference-level and a Float
        let mut metrics = HashMap::new();
        metrics.insert(
            "test_metric".to_string(),
            MetricConfig {
                r#type: MetricConfigType::Float,
                level: MetricConfigLevel::Inference,
                optimize: MetricConfigOptimize::Max,
            },
        );
        let config = Config {
            metrics,
            ..Default::default()
        };
        let inference_id = Uuid::now_v7();
        let metadata =
            get_feedback_metadata(&config, "test_metric", None, Some(inference_id)).unwrap();
        assert_eq!(metadata.r#type, FeedbackType::Float);
        assert_eq!(metadata.level, &MetricConfigLevel::Inference);
        assert_eq!(metadata.target_id, inference_id);

        // Case 1.2: ID not provided
        let metadata = get_feedback_metadata(&config, "test_metric", None, None).unwrap_err();
        let details = metadata.get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InvalidRequest {
                message: "Correct ID was not provided for feedback level \"inference\"."
                    .to_string(),
            }
        );
        // Case 1.3: ID provided but not for the correct level
        let metadata =
            get_feedback_metadata(&config, "test_metric", Some(Uuid::now_v7()), None).unwrap_err();
        let details = metadata.get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InvalidRequest {
                message: "Correct ID was not provided for feedback level \"inference\"."
                    .to_string(),
            }
        );

        // Case 1.4: Both ids are provided
        let metadata = get_feedback_metadata(
            &config,
            "test_metric",
            Some(Uuid::now_v7()),
            Some(Uuid::now_v7()),
        )
        .unwrap_err();
        let details = metadata.get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InvalidRequest {
                message: "Both episode_id and inference_id cannot be provided".to_string(),
            }
        );

        // Case 2.1: Comment Feedback, episode-level
        let episode_id = Uuid::now_v7();
        let metadata = get_feedback_metadata(&config, "comment", Some(episode_id), None).unwrap();
        assert_eq!(metadata.r#type, FeedbackType::Comment);
        assert_eq!(metadata.level, &MetricConfigLevel::Episode);
        assert_eq!(metadata.target_id, episode_id);

        // Case 2.2: Comment Feedback, inference-level
        let metadata = get_feedback_metadata(&config, "comment", None, Some(inference_id)).unwrap();
        assert_eq!(metadata.r#type, FeedbackType::Comment);
        assert_eq!(metadata.level, &MetricConfigLevel::Inference);
        assert_eq!(metadata.target_id, inference_id);

        // Case 2.3: Comment feedback but both ids are provided
        let metadata =
            get_feedback_metadata(&config, "comment", Some(episode_id), Some(inference_id))
                .unwrap_err();
        let details = metadata.get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InvalidRequest {
                message: "Both episode_id and inference_id cannot be provided".to_string(),
            }
        );

        // Case 3.1 Demonstration Feedback with only inference id
        let metadata =
            get_feedback_metadata(&config, "demonstration", None, Some(inference_id)).unwrap();
        assert_eq!(metadata.r#type, FeedbackType::Demonstration);
        assert_eq!(metadata.level, &MetricConfigLevel::Inference);
        assert_eq!(metadata.target_id, inference_id);

        // Case 3.2 Demonstration Feedback with only episode id
        let metadata =
            get_feedback_metadata(&config, "demonstration", Some(episode_id), None).unwrap_err();
        let details = metadata.get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InvalidRequest {
                message: "Correct ID was not provided for feedback level \"inference\"."
                    .to_string(),
            }
        );

        // Case 3.3 Demonstration Feedback with both IDs
        let metadata = get_feedback_metadata(
            &config,
            "demonstration",
            Some(episode_id),
            Some(inference_id),
        )
        .unwrap_err();
        let details = metadata.get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InvalidRequest {
                message: "Both episode_id and inference_id cannot be provided".to_string(),
            }
        );

        // Case 4.1 Boolean Feedback with episode level and episode id
        let metric_config = MetricConfig {
            r#type: MetricConfigType::Boolean,
            level: MetricConfigLevel::Episode,
            optimize: MetricConfigOptimize::Max,
        };
        let mut metrics = HashMap::new();
        metrics.insert("test_metric".to_string(), metric_config);
        let config = Config {
            metrics,
            ..Default::default()
        };
        let episode_id = Uuid::now_v7();
        let metadata =
            get_feedback_metadata(&config, "test_metric", Some(episode_id), None).unwrap();
        assert_eq!(metadata.r#type, FeedbackType::Boolean);
        assert_eq!(metadata.level, &MetricConfigLevel::Episode);
        assert_eq!(metadata.target_id, episode_id);

        // Case 4.2 Boolean Feedback with both ids
        let metadata =
            get_feedback_metadata(&config, "test_metric", Some(episode_id), Some(inference_id))
                .unwrap_err();
        let details = metadata.get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InvalidRequest {
                message: "Both episode_id and inference_id cannot be provided".to_string(),
            }
        );
    }

    #[tokio::test]
    async fn test_feedback_missing_metric() {
        let metrics = HashMap::new();
        let config = Config {
            metrics,
            ..Default::default()
        };
        let inference_id = Uuid::now_v7();
        let metadata_err =
            get_feedback_metadata(&config, "missing_metric_name", None, Some(inference_id))
                .unwrap_err();
        let details = metadata_err.get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::UnknownMetric {
                name: "missing_metric_name".to_string(),
            }
        );
    }
    #[tokio::test]
    async fn test_feedback_handler_comment() {
        let config = Arc::new(Config {
            ..Default::default()
        });
        let app_state_data = get_unit_test_app_state_data(config, true);
        let timestamp = uuid::Timestamp::from_unix_time(1579751960, 0, 0, 0);
        let episode_id = Uuid::new_v7(timestamp);
        let value = json!("test comment");
        let params = Params {
            episode_id: Some(episode_id),
            inference_id: None,
            metric_name: "comment".to_string(),
            value: value.clone(),
            tags: HashMap::from([("foo".to_string(), "bar".to_string())]),
            internal: false,
            dryrun: Some(false),
        };
        let response =
            feedback_handler(State(app_state_data.clone()), StructuredJson(params)).await;
        let details = response.unwrap_err().get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InvalidRequest {
                message: format!("Episode ID: {episode_id} does not exist"),
            }
        );
    }

    #[tokio::test]
    async fn test_feedback_handler_demonstration() {
        let config = Arc::new(Config {
            ..Default::default()
        });
        let app_state_data = get_unit_test_app_state_data(config, true);
        let timestamp = uuid::Timestamp::from_unix_time(1579751960, 0, 0, 0);
        let episode_id = Uuid::new_v7(timestamp);
        let value = json!("test demonstration");

        // Test with episode_id (should fail)
        let params = Params {
            episode_id: Some(episode_id),
            inference_id: None,
            metric_name: "demonstration".to_string(),
            value: value.clone(),
            tags: HashMap::from([("baz".to_string(), "bat".to_string())]),
            dryrun: Some(false),
            internal: false,
        };
        let response = feedback_handler(State(app_state_data.clone()), StructuredJson(params))
            .await
            .unwrap_err();
        let details = response.get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InvalidRequest {
                message: "Correct ID was not provided for feedback level \"inference\"."
                    .to_string(),
            }
        );

        // Test with inference_id (should fail with non-existent ID)
        let inference_id = Uuid::new_v7(timestamp);
        let params = Params {
            episode_id: None,
            inference_id: Some(inference_id),
            metric_name: "demonstration".to_string(),
            value: value.clone(),
            tags: HashMap::from([("bat".to_string(), "man".to_string())]),
            dryrun: Some(false),
            internal: false,
        };
        let response =
            feedback_handler(State(app_state_data.clone()), StructuredJson(params)).await;
        let details = response.unwrap_err().get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InvalidRequest {
                message: format!("Inference ID: {inference_id} does not exist"),
            }
        );
    }

    #[tokio::test]
    async fn test_feedback_handler_float_episode_level() {
        let mut metrics = HashMap::new();
        metrics.insert(
            "test_float".to_string(),
            MetricConfig {
                r#type: MetricConfigType::Float,
                level: MetricConfigLevel::Episode,
                optimize: MetricConfigOptimize::Max,
            },
        );
        let config = Arc::new(Config {
            metrics,
            ..Default::default()
        });
        let app_state_data = get_unit_test_app_state_data(config.clone(), true);
        let value = json!(4.5);
        let timestamp = uuid::Timestamp::from_unix_time(1579751960, 0, 0, 0);
        let inference_id = Uuid::new_v7(timestamp);
        let episode_id = Uuid::new_v7(timestamp);

        // Test with inference_id (should fail)
        let params = Params {
            episode_id: None,
            inference_id: Some(inference_id),
            metric_name: "test_float".to_string(),
            value: value.clone(),
            tags: HashMap::from([("boo".to_string(), "far".to_string())]),
            dryrun: Some(false),
            internal: false,
        };
        let response = feedback_handler(State(app_state_data.clone()), StructuredJson(params))
            .await
            .unwrap_err();
        let details = response.get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InvalidRequest {
                message: "Correct ID was not provided for feedback level \"episode\".".to_string(),
            }
        );

        // Test with episode_id (should fail with non-existent ID)
        let params = Params {
            episode_id: Some(episode_id),
            inference_id: None,
            metric_name: "test_float".to_string(),
            value: value.clone(),
            tags: HashMap::from([("poo".to_string(), "bar".to_string())]),
            dryrun: Some(false),
            internal: false,
        };
        let response =
            feedback_handler(State(app_state_data.clone()), StructuredJson(params)).await;
        let details = response.unwrap_err().get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InvalidRequest {
                message: format!("Episode ID: {episode_id} does not exist"),
            }
        );
    }

    #[tokio::test]
    async fn test_feedback_handler_boolean_inference_level() {
        let mut metrics = HashMap::new();
        metrics.insert(
            "test_boolean".to_string(),
            MetricConfig {
                r#type: MetricConfigType::Boolean,
                level: MetricConfigLevel::Inference,
                optimize: MetricConfigOptimize::Max,
            },
        );
        let config = Arc::new(Config {
            metrics,
            ..Default::default()
        });
        let app_state_data = get_unit_test_app_state_data(config.clone(), true);
        let value = json!(true);
        let timestamp = uuid::Timestamp::from_unix_time(1579751960, 0, 0, 0);
        let inference_id = Uuid::new_v7(timestamp);
        let params = Params {
            episode_id: None,
            inference_id: Some(inference_id),
            metric_name: "test_boolean".to_string(),
            value: value.clone(),
            tags: HashMap::from([("new".to_string(), "car".to_string())]),
            dryrun: None,
            internal: false,
        };
        let response =
            feedback_handler(State(app_state_data.clone()), StructuredJson(params)).await;
        let details = response.unwrap_err().get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InvalidRequest {
                message: format!("Inference ID: {inference_id} does not exist"),
            }
        );
    }

    #[tokio::test]
    async fn test_validate_parse_demonstration() {
        let weather_tool_config_static = StaticToolConfig {
            name: "get_temperature".to_string(),
            description: "Get the current temperature in a given location".to_string(),
            parameters: JSONSchemaFromPath::from_value(&json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }))
            .unwrap(),
            strict: false,
        };
        let tools = HashMap::from([(
            "get_temperature".to_string(),
            Arc::new(weather_tool_config_static),
        )]);
        let function_config_chat_tools =
            Box::leak(Box::new(FunctionConfig::Chat(FunctionConfigChat {
                variants: HashMap::new(),
                system_schema: None,
                user_schema: None,
                assistant_schema: None,
                tools: vec!["get_temperature".to_string()],
                tool_choice: ToolChoice::Auto,
                parallel_tool_calls: None,
            })));

        // Case 1: a string passed to a chat function
        let value = json!("Hello, world!");
        let dynamic_demonstration_info = DynamicDemonstrationInfo::Chat(ToolCallConfig {
            tools_available: tools.values().cloned().map(ToolConfig::Static).collect(),
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        });
        let parsed_value = serde_json::to_string(
            &validate_parse_demonstration(
                function_config_chat_tools,
                &value,
                dynamic_demonstration_info,
            )
            .await
            .unwrap(),
        )
        .unwrap();
        let expected_parsed_value = serde_json::to_string(&vec![ContentBlockOutput::Text(Text {
            text: "Hello, world!".to_string(),
        })])
        .unwrap();
        assert_eq!(expected_parsed_value, parsed_value);

        // Case 2: a tool call to get_temperature, which exists
        let value = json!([{"type": "tool_call", "name": "get_temperature", "arguments": {"location": "London", "unit": "celsius"}}]
        );
        let dynamic_demonstration_info = DynamicDemonstrationInfo::Chat(ToolCallConfig {
            tools_available: tools.values().cloned().map(ToolConfig::Static).collect(),
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        });
        let parsed_value = serde_json::to_string(
            &validate_parse_demonstration(
                function_config_chat_tools,
                &value,
                dynamic_demonstration_info,
            )
            .await
            .unwrap(),
        )
        .unwrap();
        let expected_parsed_value =
            serde_json::to_string(&vec![ContentBlockChatOutput::ToolCall(ToolCallOutput {
                id: "".to_string(),
                name: Some("get_temperature".to_string()),
                raw_name: "get_temperature".to_string(),
                arguments: Some(json!({"location": "London", "unit": "celsius"})),
                raw_arguments: serde_json::to_string(
                    &json!({"location": "London", "unit": "celsius"}),
                )
                .unwrap(),
            })])
            .unwrap();
        assert_eq!(expected_parsed_value, parsed_value);

        // Case 3: a tool call to get_humidity, which does not exist
        let value = json!([{"type": "tool_call", "name": "get_humidity", "arguments": {"location": "London", "unit": "celsius"}}]
        );
        let dynamic_demonstration_info = DynamicDemonstrationInfo::Chat(ToolCallConfig {
            tools_available: tools.values().cloned().map(ToolConfig::Static).collect(),
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        });
        let err = validate_parse_demonstration(
            function_config_chat_tools,
            &value,
            dynamic_demonstration_info,
        )
        .await
        .unwrap_err();
        let details = err.get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InvalidRequest {
                message: "Demonstration contains invalid tool name".to_string(),
            }
        );

        // Case 4: a tool call to get_temperature, which exists but has bad arguments (place instead of location)
        let value = json!([{"type": "tool_call", "name": "get_temperature", "arguments": {"place": "London", "unit": "celsius"}}]
        );
        let dynamic_demonstration_info = DynamicDemonstrationInfo::Chat(ToolCallConfig {
            tools_available: tools.values().cloned().map(ToolConfig::Static).collect(),
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        });
        let err = validate_parse_demonstration(
            function_config_chat_tools,
            &value,
            dynamic_demonstration_info,
        )
        .await
        .unwrap_err();
        let details = err.get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InvalidRequest {
                message: "Demonstration contains invalid tool call arguments".to_string(),
            }
        );

        // Let's try a JSON function
        let output_schema = json!({
          "$schema": "http://json-schema.org/draft-07/schema#",
          "type": "object",
          "properties": {
            "name": {
              "type": "string"
            },
            "age": {
              "type": "integer",
              "minimum": 0
            }
          },
          "required": ["name", "age"],
          "additionalProperties": false
        });
        let implicit_tool_call_config = ToolCallConfig::implicit_from_value(&output_schema);
        let function_config = Box::leak(Box::new(FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            output_schema: JSONSchemaFromPath::from_value(&output_schema).unwrap(),
            implicit_tool_call_config,
        })));

        // Case 5: a JSON function with correct output
        let value = json!({
            "name": "John",
            "age": 30
        });
        let dynamic_demonstration_info = DynamicDemonstrationInfo::Json(output_schema.clone());
        let parsed_value = serde_json::to_string(
            &validate_parse_demonstration(function_config, &value, dynamic_demonstration_info)
                .await
                .unwrap(),
        )
        .unwrap();
        let expected_parsed_value = serde_json::to_string(&json!({
            "raw": serde_json::to_string(&value).unwrap(),
            "parsed": value
        }))
        .unwrap();
        assert_eq!(expected_parsed_value, parsed_value);
        // Case 6: a JSON function with incorrect output
        let value = json!("Hello, world!");
        let dynamic_demonstration_info = DynamicDemonstrationInfo::Json(output_schema.clone());
        let err = validate_parse_demonstration(function_config, &value, dynamic_demonstration_info)
            .await
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("Demonstration does not fit function output schema"));

        // Case 7: Mismatched function type - Chat function with JSON demonstration info
        let value = json!("Hello, world!");
        let dynamic_demonstration_info = DynamicDemonstrationInfo::Json(output_schema.clone());
        let err = validate_parse_demonstration(
            function_config_chat_tools,
            &value,
            dynamic_demonstration_info,
        )
        .await
        .unwrap_err();
        let details = err.get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::Inference {
                message: "The DynamicDemonstrationInfo does not match the function type. This should never happen. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports.".to_string()
            }
        );

        // Case 8: Mismatched function type - JSON function with Chat demonstration info
        let value = json!({
            "name": "John",
            "age": 30
        });
        let dynamic_demonstration_info = DynamicDemonstrationInfo::Chat(ToolCallConfig {
            tools_available: tools.values().cloned().map(ToolConfig::Static).collect(),
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        });
        let err = validate_parse_demonstration(function_config, &value, dynamic_demonstration_info)
            .await
            .unwrap_err();
        let details = err.get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::Inference {
                message: "The DynamicDemonstrationInfo does not match the function type. This should never happen. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports.".to_string()
            }
        );
    }
}
