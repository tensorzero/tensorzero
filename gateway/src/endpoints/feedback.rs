use std::collections::HashMap;
use std::time::Duration;

use axum::extract::State;
use axum::{debug_handler, Json};
use metrics::counter;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::instrument;
use uuid::Uuid;

use crate::clickhouse::ClickHouseConnectionInfo;
use crate::config_parser::{Config, MetricConfigLevel, MetricConfigType};
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfig;
use crate::gateway_util::{AppState, AppStateData, StructuredJson};
use crate::inference::types::{parse_chat_output, ContentBlock, ContentBlockOutput, Text};
use crate::tool::{DynamicToolParams, StaticToolConfig, ToolCall, ToolCallConfig};
use crate::uuid_util::uuid_elapsed;

/// There is a potential issue here where if we write an inference and then immediately write feedback for it,
/// we might not be able to find the inference in the database because it hasn't been written yet.
///
/// This is the amount of time we want to wait after the target was supposed to have been written
/// before we decide that the target was actually not written because we can't find it in the database.
const FEEDBACK_COOLDOWN_PERIOD: Duration = Duration::from_secs(5);

/// The expected payload is a JSON object with the following fields:
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    // the episode ID client is providing feedback for (either this or `inference_id` must be set but not both)
    episode_id: Option<Uuid>,
    // the inference ID client is providing feedback for (either this or `episode_id` must be set but not both)
    inference_id: Option<Uuid>,
    // the name of the Metric to provide feedback for (this can always also be "comment" or "demonstration")
    metric_name: String,
    // the value of the feedback being provided
    value: Value,
    // the tags to add to the feedback
    #[serde(default)]
    tags: HashMap<String, String>,
    // if true, the feedback will not be stored
    dryrun: Option<bool>,
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

/// A handler for the feedback endpoint
#[instrument(name="feedback",
  skip_all,
  fields(
    metric_name = %params.metric_name,
  )
)]
#[debug_handler(state = AppStateData)]
pub async fn feedback_handler(
    State(AppStateData {
        config,
        clickhouse_connection_info,
        ..
    }): AppState,
    StructuredJson(params): StructuredJson<Params>,
) -> Result<Json<Value>, Error> {
    // Get the metric config or return an error if it doesn't exist
    let feedback_metadata = get_feedback_metadata(
        config,
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
                config,
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
                config,
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
                config,
                &params,
                feedback_metadata.target_id,
                feedback_id,
                dryrun,
            )
            .await?
        }
    }

    Ok(Json(json!({"feedback_id": feedback_id})))
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
    config: &'static Config<'_>,
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
    let parsed_value = validate_parse_demonstration(function_config, &config.tools, value).await?;
    let payload = json!({"inference_id": inference_id, "value": parsed_value, "id": feedback_id, "tags": tags});
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
    config: &'static Config<'_>,
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
    config: &'static Config<'_>,
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
/// The current behavior is to immediately check and return the result if the id was created
/// more than FEEDBACK_COOLDOWN_PERIOD ago.
///
/// Otherwise, it will poll the database every second until the cooldown period has passed.
/// Then, if the id has still not been created, it will return an error.
async fn throttled_get_function_name(
    connection_info: &ClickHouseConnectionInfo,
    metric_config_level: &MetricConfigLevel,
    target_id: &Uuid,
) -> Result<String, Error> {
    let mut elapsed = uuid_elapsed(target_id)?;
    if elapsed > FEEDBACK_COOLDOWN_PERIOD {
        return get_function_name(connection_info, metric_config_level, target_id).await;
    }

    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;
        elapsed = uuid_elapsed(target_id)?;
        match get_function_name(connection_info, metric_config_level, target_id).await {
            Ok(identifier) => return Ok(identifier),
            Err(e) => {
                if elapsed > FEEDBACK_COOLDOWN_PERIOD {
                    return Err(e);
                }
            }
        }
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
        MetricConfigLevel::Inference => "id",
        MetricConfigLevel::Episode => "episode_id",
    };
    let query = format!(
        "SELECT function_name FROM {} WHERE {} = '{}'",
        table_name, identifier_key, target_id
    );
    let function_name = connection_info.run_query(query).await?.trim().to_string();
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

impl TryFrom<DemonstrationContentBlock> for ContentBlock {
    type Error = Error;
    fn try_from(value: DemonstrationContentBlock) -> Result<Self, Self::Error> {
        match value {
            DemonstrationContentBlock::Text(text) => Ok(ContentBlock::Text(text)),
            DemonstrationContentBlock::ToolCall(tool_call) => {
                Ok(ContentBlock::ToolCall(tool_call.try_into()?))
            }
        }
    }
}

// Validates that the demonstration is correct.
// For chat functions, the value should be a string or a list of valid content blocks.
// For json functions, the value is validated against the output schema. If it passes,
// we construct the usual {"raw": str, "parsed": parsed_value} object, serialize it, and return it.
async fn validate_parse_demonstration(
    function_config: &'static FunctionConfig,
    tools: &'static HashMap<String, StaticToolConfig>,
    value: &Value,
) -> Result<String, Error> {
    match function_config {
        FunctionConfig::Chat(chat_config) => {
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
            let tool_config = ToolCallConfig::new(
                &chat_config.tools,
                &chat_config.tool_choice,
                chat_config.parallel_tool_calls,
                tools,
                // TODO (#348): Eventually we should allow dynamic tools here as well.
                DynamicToolParams {
                    allowed_tools: None,
                    additional_tools: None,
                    tool_choice: None,
                    parallel_tool_calls: None,
                },
            )?;
            let content_blocks: Vec<ContentBlock> = content_blocks
                .into_iter()
                .map(|block| block.try_into())
                .collect::<Result<Vec<ContentBlock>, Error>>()?;
            let parsed_value = parse_chat_output(content_blocks, tool_config.as_ref()).await;
            for block in &parsed_value {
                if let ContentBlockOutput::ToolCall(tool_call) = block {
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
            let serialized_parsed_content_blocks =
                serde_json::to_string(&parsed_value).map_err(|e| {
                    Error::new(ErrorDetails::InvalidRequest {
                        message: format!("Failed to serialize parsed value to json: {}", e),
                    })
                })?;
            Ok(serialized_parsed_content_blocks)
        }
        FunctionConfig::Json(json_config) => {
            // For json functions, the value should be a valid json object.
            // TODO (#348): Eventually we should allow dynamic output_schema here as well.
            json_config.output_schema.validate(value).map_err(|e| {
                Error::new(ErrorDetails::InvalidRequest {
                    message: format!("Demonstration does not fit function output schema: {}", e),
                })
            })?;
            let raw = serde_json::to_string(value).map_err(|e| {
                Error::new(ErrorDetails::InvalidRequest {
                    message: format!("Failed to serialize demonstration to json: {}", e),
                })
            })?;
            let json_inference_response = json!({"raw": raw, "parsed": value});
            let serialized_json_inference_response =
                serde_json::to_string(&json_inference_response).map_err(|e| {
                    Error::new(ErrorDetails::InvalidRequest {
                        message: format!(
                            "Failed to serialize json_inference_response to json: {}",
                            e
                        ),
                    })
                })?;
            Ok(serialized_json_inference_response)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use crate::config_parser::{Config, GatewayConfig, MetricConfig, MetricConfigOptimize};
    use crate::function::{FunctionConfigChat, FunctionConfigJson};
    use crate::jsonschema_util::JSONSchemaFromPath;
    use crate::minijinja_util::TemplateConfig;
    use crate::model::ModelTable;
    use crate::testing::get_unit_test_app_state_data;
    use crate::tool::{ToolCallOutput, ToolChoice};

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
            gateway: GatewayConfig::default(),
            models: ModelTable::default(),
            embedding_models: HashMap::new(),
            metrics,
            functions: HashMap::new(),
            tools: HashMap::new(),
            templates: TemplateConfig::new(),
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
            gateway: GatewayConfig::default(),
            models: ModelTable::default(),
            embedding_models: HashMap::new(),
            metrics,
            functions: HashMap::new(),
            tools: HashMap::new(),
            templates: TemplateConfig::new(),
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
            gateway: GatewayConfig::default(),
            models: ModelTable::default(),
            embedding_models: HashMap::new(),
            metrics,
            functions: HashMap::new(),
            tools: HashMap::new(),
            templates: TemplateConfig::new(),
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
    async fn test_feedback_handler() {
        // Test a Comment Feedback
        let config = Box::leak(Box::new(Config {
            gateway: GatewayConfig::default(),
            models: ModelTable::default(),
            embedding_models: HashMap::new(),
            metrics: HashMap::new(),
            functions: HashMap::new(),
            tools: HashMap::new(),
            templates: TemplateConfig::new(),
        }));
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

        // Test a Demonstration Feedback
        let value = json!("test demonstration");
        let params = Params {
            // Demonstrations shouldn't work with episode id
            episode_id: Some(episode_id),
            inference_id: None,
            metric_name: "demonstration".to_string(),
            value: value.clone(),
            tags: HashMap::from([("baz".to_string(), "bat".to_string())]),
            dryrun: Some(false),
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

        let inference_id = Uuid::new_v7(timestamp);
        let params = Params {
            episode_id: None,
            inference_id: Some(inference_id),
            metric_name: "demonstration".to_string(),
            value: value.clone(),
            tags: HashMap::from([("bat".to_string(), "man".to_string())]),
            dryrun: Some(false),
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

        // Test a Float Feedback (episode level)
        let mut metrics = HashMap::new();
        metrics.insert(
            "test_float".to_string(),
            MetricConfig {
                r#type: MetricConfigType::Float,
                level: MetricConfigLevel::Episode,
                optimize: MetricConfigOptimize::Max,
            },
        );
        let config = Box::leak(Box::new(Config {
            gateway: GatewayConfig::default(),
            models: ModelTable::default(),
            embedding_models: HashMap::new(),
            metrics,
            functions: HashMap::new(),
            tools: HashMap::new(),
            templates: TemplateConfig::new(),
        }));
        let app_state_data = get_unit_test_app_state_data(config, true);
        let value = json!(4.5);
        let inference_id = Uuid::new_v7(timestamp);
        let episode_id = Uuid::new_v7(timestamp);
        let params = Params {
            episode_id: None,
            inference_id: Some(inference_id),
            metric_name: "test_float".to_string(),
            value: value.clone(),
            tags: HashMap::from([("boo".to_string(), "far".to_string())]),
            dryrun: Some(false),
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

        let params = Params {
            episode_id: Some(episode_id),
            inference_id: None,
            metric_name: "test_float".to_string(),
            value: value.clone(),
            tags: HashMap::from([("poo".to_string(), "bar".to_string())]),
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

        // Test Boolean feedback with invalid inference-level.
        let mut metrics = HashMap::new();
        metrics.insert(
            "test_boolean".to_string(),
            MetricConfig {
                r#type: MetricConfigType::Boolean,
                level: MetricConfigLevel::Inference,
                optimize: MetricConfigOptimize::Max,
            },
        );
        let config = Box::leak(Box::new(Config {
            gateway: GatewayConfig::default(),
            models: ModelTable::default(),
            embedding_models: HashMap::new(),
            metrics,
            functions: HashMap::new(),
            tools: HashMap::new(),
            templates: TemplateConfig::new(),
        }));
        let app_state_data = get_unit_test_app_state_data(config, true);
        let value = json!(true);
        let inference_id = Uuid::new_v7(timestamp);
        let params = Params {
            episode_id: None,
            inference_id: Some(inference_id),
            metric_name: "test_boolean".to_string(),
            value: value.clone(),
            tags: HashMap::from([("new".to_string(), "car".to_string())]),
            dryrun: None,
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
        let tools = Box::leak(Box::new(HashMap::from([(
            "get_temperature".to_string(),
            weather_tool_config_static,
        )])));
        let function_config_chat_tools =
            Box::leak(Box::new(FunctionConfig::Chat(FunctionConfigChat {
                variants: HashMap::new(),
                system_schema: None,
                user_schema: None,
                assistant_schema: None,
                tools: vec!["get_temperature".to_string()],
                tool_choice: ToolChoice::Auto,
                parallel_tool_calls: false,
            })));

        // Case 1: a string passed to a chat function
        let value = json!("Hello, world!");
        let parsed_value = validate_parse_demonstration(function_config_chat_tools, tools, &value)
            .await
            .unwrap();
        let expected_parsed_value = serde_json::to_string(&vec![ContentBlock::Text(Text {
            text: "Hello, world!".to_string(),
        })])
        .unwrap();
        assert_eq!(expected_parsed_value, parsed_value);

        // Case 2: a tool call to get_temperature, which exists
        let value = json!([{"type": "tool_call", "name": "get_temperature", "arguments": {"location": "London", "unit": "celsius"}}]
        );
        let parsed_value = validate_parse_demonstration(function_config_chat_tools, tools, &value)
            .await
            .unwrap();
        let expected_parsed_value =
            serde_json::to_string(&vec![ContentBlockOutput::ToolCall(ToolCallOutput {
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
        let err = validate_parse_demonstration(function_config_chat_tools, tools, &value)
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
        let err = validate_parse_demonstration(function_config_chat_tools, tools, &value)
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
        let output_schema = JSONSchemaFromPath::from_value(&output_schema).unwrap();
        let function_config = Box::leak(Box::new(FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            output_schema,
            implicit_tool_call_config,
        })));

        // Case 5: a JSON function with correct output
        let value = json!({
            "name": "John",
            "age": 30
        });
        let parsed_value = validate_parse_demonstration(function_config, tools, &value)
            .await
            .unwrap();
        let expected_parsed_value = serde_json::to_string(&json!({
            "raw": serde_json::to_string(&value).unwrap(),
            "parsed": value
        }))
        .unwrap();
        assert_eq!(expected_parsed_value, parsed_value);

        // Case 6: a JSON function with incorrect output
        let value = json!("Hello, world!");
        validate_parse_demonstration(function_config, tools, &value)
            .await
            .unwrap_err();
    }
}
