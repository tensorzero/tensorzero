use std::cmp::max;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use axum::extract::State;
use axum::{Extension, Json, debug_handler};
use human_feedback::write_static_evaluation_human_feedback_if_necessary;
use metrics::counter;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tensorzero_derive::TensorZeroDeserialize;
use tokio::{time::Instant, try_join};
use tokio_util::task::TaskTracker;
use tracing::instrument;
use uuid::Uuid;

use crate::config::snapshot::SnapshotHash;
use crate::config::{Config, MetricConfigLevel, MetricConfigType};
use crate::db::delegating_connection::DelegatingDatabaseConnection;
use crate::db::feedback::{
    BooleanMetricFeedbackInsert, CommentFeedbackInsert, CommentTargetType,
    DemonstrationFeedbackInsert, FeedbackQueries, FloatMetricFeedbackInsert,
};
use crate::db::inferences::{FunctionInfo, InferenceQueries};
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfig;
use crate::inference::types::{
    ContentBlockChatOutput, ContentBlockOutput, Text, parse_chat_output,
};
use crate::jsonschema_util::JSONSchema;
use crate::tool::{StaticToolConfig, ToolCall, ToolCallConfig};
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};
use crate::utils::uuid::uuid_elapsed;
use tensorzero_auth::middleware::RequestApiKeyExtension;
use tracing_opentelemetry::OpenTelemetrySpanExt;

use super::validate_tags;

pub mod human_feedback;
pub mod internal;

/// There is a potential issue here where if we write an inference and then immediately write feedback for it,
/// we might not be able to find the inference in the database because it hasn't been written yet.
///
/// This is the amount of time we want to wait after the target was supposed to have been written
/// before we decide that the target was actually not written because we can't find it in the database.
/// This should really be read at 5000ms but since there might be some jitter we want to make sure there's
/// a read at ~5s
const FEEDBACK_COOLDOWN_PERIOD: Duration = Duration::from_millis(6000);
/// Since we can't be sure that an inference actually completed when the ID says it was
/// (the ID is generated at the start of the inference), we wait a minimum amount of time
/// before we decide that the target was actually not written because we can't find it in the database.
const FEEDBACK_MINIMUM_WAIT_TIME: Duration = Duration::from_millis(1000);
/// We also poll in the intermediate time so that we can return as soon as we find a target entry.
const FEEDBACK_TARGET_POLL_INTERVAL: Duration = Duration::from_millis(2000);

// TODO(shuyangli): rename this to CreateFeedbackRequest and export
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
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

// TODO(shuyangli): rename this to CreateFeedbackResponse and export
#[derive(Debug, Serialize, Deserialize)]
pub struct FeedbackResponse {
    pub feedback_id: Uuid,
}

/// A handler for the feedback endpoint
#[debug_handler(state = AppStateData)]
pub async fn feedback_handler(
    State(app_state): AppState,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
    StructuredJson(params): StructuredJson<Params>,
) -> Result<Json<FeedbackResponse>, Error> {
    Ok(Json(feedback(app_state, params, api_key_ext).await?))
}

// Helper function to avoid requiring axum types in the client
#[instrument(name="feedback",
  skip_all,
  fields(
    inference_id,
    episode_id,
    metric_name = %params.metric_name,
    otel.name = "feedback"
  )
)]
pub async fn feedback(
    AppStateData {
        config,
        clickhouse_connection_info,
        postgres_connection_info,
        deferred_tasks,
        ..
    }: AppStateData,
    mut params: Params,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
) -> Result<FeedbackResponse, Error> {
    let span = tracing::Span::current();
    if let Some(inference_id) = params.inference_id {
        span.record("inference_id", inference_id.to_string());
    }
    if let Some(episode_id) = params.episode_id {
        span.record("episode_id", episode_id.to_string());
    }

    // Automatically add internal tag when internal=true
    if params.internal {
        params
            .tags
            .insert("tensorzero::internal".to_string(), "true".to_string());
    }

    for (tag_key, tag_value) in &params.tags {
        span.set_attribute(format!("tags.{tag_key}"), tag_value.clone());
    }
    validate_tags(&params.tags, params.internal)?;
    validate_feedback_specific_tags(&params.tags)?;
    if let Some(api_key_ext) = api_key_ext {
        params.tags.insert(
            "tensorzero::api_key_public_id".to_string(),
            api_key_ext.0.api_key.get_public_id().into(),
        );
    }
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
            "tensorzero_requests_total",
            "endpoint" => "feedback",
            "metric_name" => params.metric_name.to_string()
        )
        .increment(1);
    }

    // Note: InferenceQueries is only implemented for ClickHouse currently.
    // When Postgres implements InferenceQueries, we can use ENABLE_POSTGRES_READ to select.
    //
    // TODO(shuyangli): Also implement InferenceQueries for DelegatingDatabaseConnection and only pass one in
    let read_database: Arc<dyn InferenceQueries + Send + Sync> =
        Arc::new(clickhouse_connection_info.clone());
    let write_database: Arc<dyn FeedbackQueries + Send + Sync> =
        Arc::new(DelegatingDatabaseConnection::new(
            clickhouse_connection_info.clone(),
            postgres_connection_info.clone(),
        ));

    match feedback_metadata.r#type {
        FeedbackType::Comment => {
            write_comment(
                read_database,
                write_database,
                &deferred_tasks,
                &params,
                feedback_metadata.target_id,
                feedback_metadata.level,
                feedback_id,
                dryrun,
                config.gateway.unstable_disable_feedback_target_validation,
                config.hash.clone(),
            )
            .await?;
        }
        FeedbackType::Demonstration => {
            write_demonstration(
                read_database,
                write_database,
                &deferred_tasks,
                &config,
                &params,
                feedback_metadata.target_id,
                feedback_id,
                dryrun,
            )
            .await?;
        }
        FeedbackType::Float => {
            write_float(
                read_database,
                write_database,
                &deferred_tasks,
                &config,
                params,
                feedback_metadata.target_id,
                feedback_id,
                dryrun,
                config.gateway.unstable_disable_feedback_target_validation,
            )
            .await?;
        }
        FeedbackType::Boolean => {
            write_boolean(
                read_database,
                write_database,
                &deferred_tasks,
                &config,
                params,
                feedback_metadata.target_id,
                feedback_id,
                dryrun,
                config.gateway.unstable_disable_feedback_target_validation,
            )
            .await?;
        }
    }

    Ok(FeedbackResponse { feedback_id })
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
        message: format!(r#"Correct ID was not provided for feedback level "{feedback_level}"."#),
    })?;
    Ok(FeedbackMetadata {
        r#type: feedback_type,
        level: feedback_level,
        target_id,
    })
}

#[expect(clippy::too_many_arguments)]
async fn write_comment(
    read_database: Arc<dyn InferenceQueries + Send + Sync>,
    write_database: Arc<dyn FeedbackQueries + Send + Sync>,
    deferred_tasks: &TaskTracker,
    params: &Params,
    target_id: Uuid,
    level: &MetricConfigLevel,
    feedback_id: Uuid,
    dryrun: bool,
    disable_validation: bool,
    snapshot_hash: SnapshotHash,
) -> Result<(), Error> {
    let Params { value, tags, .. } = params;
    // Verify that the function name exists.
    if !disable_validation {
        let _ = throttled_get_function_info(read_database.as_ref(), level, &target_id).await?;
    }
    let value = value.as_str().ok_or_else(|| ErrorDetails::InvalidRequest {
        message: "Feedback value for a comment must be a string".to_string(),
    })?;
    let target_type = match level {
        MetricConfigLevel::Inference => CommentTargetType::Inference,
        MetricConfigLevel::Episode => CommentTargetType::Episode,
    };
    let insert = CommentFeedbackInsert {
        id: feedback_id,
        target_id,
        target_type,
        value: value.to_string(),
        tags: tags.clone(),
        snapshot_hash,
    };

    if !dryrun {
        deferred_tasks.spawn(async move {
            let _ = write_database.insert_comment_feedback(&insert).await;
        });
    }
    Ok(())
}

#[expect(clippy::too_many_arguments)]
async fn write_demonstration(
    read_database: Arc<dyn InferenceQueries + Send + Sync>,
    write_database: Arc<dyn FeedbackQueries + Send + Sync>,
    deferred_tasks: &TaskTracker,
    config: &Config,
    params: &Params,
    inference_id: Uuid,
    feedback_id: Uuid,
    dryrun: bool,
) -> Result<(), Error> {
    let Params { value, tags, .. } = params;
    let function_info = throttled_get_function_info(
        read_database.as_ref(),
        &MetricConfigLevel::Inference,
        &inference_id,
    )
    .await?;
    let function_config = config.get_function(&function_info.function_name)?;
    let dynamic_demonstration_info = get_dynamic_demonstration_info(
        read_database.as_ref(),
        inference_id,
        &function_info.function_name,
        &function_config,
        &config.tools,
    )
    .await?;
    let parsed_value =
        validate_parse_demonstration(&function_config, value, dynamic_demonstration_info).await?;
    let string_value = serde_json::to_string(&parsed_value).map_err(|e| {
        Error::new(ErrorDetails::InvalidRequest {
            message: format!("Failed to serialize parsed value to json: {e}"),
        })
    })?;
    let insert = DemonstrationFeedbackInsert {
        id: feedback_id,
        inference_id,
        value: string_value,
        tags: tags.clone(),
        snapshot_hash: config.hash.clone(),
    };

    if !dryrun {
        deferred_tasks.spawn(async move {
            let _ = write_database.insert_demonstration_feedback(&insert).await;
        });
    }
    Ok(())
}

#[expect(clippy::too_many_arguments)]
async fn write_float(
    read_database: Arc<dyn InferenceQueries + Send + Sync>,
    write_database: Arc<dyn FeedbackQueries + Send + Sync>,
    deferred_tasks: &TaskTracker,
    config: &Config,
    params: Params,
    target_id: Uuid,
    feedback_id: Uuid,
    dryrun: bool,
    disable_validation: bool,
) -> Result<(), Error> {
    let Params {
        metric_name,
        value,
        tags,
        ..
    } = params;
    let metric_config: &crate::config::MetricConfig = config.get_metric_or_err(&metric_name)?;
    let maybe_function_info = if disable_validation {
        None
    } else {
        // This will also throw if the function does not exist.
        Some(
            throttled_get_function_info(read_database.as_ref(), &metric_config.level, &target_id)
                .await?,
        )
    };

    let float_value = value.as_f64().ok_or_else(|| {
        Error::new(ErrorDetails::InvalidRequest {
            message: format!("Feedback value for metric `{metric_name}` must be a number"),
        })
    })?;
    let insert = FloatMetricFeedbackInsert {
        id: feedback_id,
        target_id,
        metric_name: metric_name.clone(),
        value: float_value,
        tags: tags.clone(),
        snapshot_hash: config.hash.clone(),
    };

    if !dryrun {
        deferred_tasks.spawn(async move {
            let _ = try_join!(
                write_static_evaluation_human_feedback_if_necessary(
                    read_database.as_ref(),
                    write_database.as_ref(),
                    maybe_function_info,
                    &metric_name,
                    &tags,
                    feedback_id,
                    &value,
                    target_id
                ),
                write_database.insert_float_feedback(&insert)
            );
        });
    }
    Ok(())
}

#[expect(clippy::too_many_arguments)]
async fn write_boolean(
    read_database: Arc<dyn InferenceQueries + Send + Sync>,
    write_database: Arc<dyn FeedbackQueries + Send + Sync>,
    deferred_tasks: &TaskTracker,
    config: &Config,
    params: Params,
    target_id: Uuid,
    feedback_id: Uuid,
    dryrun: bool,
    disable_validation: bool,
) -> Result<(), Error> {
    let Params {
        metric_name,
        value,
        tags,
        ..
    } = params;
    let metric_config = config.get_metric_or_err(&metric_name)?;
    let maybe_function_info = if disable_validation {
        None
    } else {
        // This will also throw if the function does not exist.
        Some(
            throttled_get_function_info(read_database.as_ref(), &metric_config.level, &target_id)
                .await?,
        )
    };
    let bool_value = value.as_bool().ok_or_else(|| {
        Error::new(ErrorDetails::InvalidRequest {
            message: format!("Feedback value for metric `{metric_name}` must be a boolean"),
        })
    })?;
    let insert = BooleanMetricFeedbackInsert {
        id: feedback_id,
        target_id,
        metric_name: metric_name.clone(),
        value: bool_value,
        tags: tags.clone(),
        snapshot_hash: config.hash.clone(),
    };

    if !dryrun {
        deferred_tasks.spawn(async move {
            let _ = try_join!(
                write_static_evaluation_human_feedback_if_necessary(
                    read_database.as_ref(),
                    write_database.as_ref(),
                    maybe_function_info,
                    &metric_name,
                    &tags,
                    feedback_id,
                    &value,
                    target_id
                ),
                write_database.insert_boolean_feedback(&insert)
            );
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
async fn throttled_get_function_info(
    db_client: &(dyn InferenceQueries + Sync),
    metric_config_level: &MetricConfigLevel,
    target_id: &Uuid,
) -> Result<FunctionInfo, Error> {
    // Compute how long ago the target_id was created.
    let elapsed = match uuid_elapsed(target_id) {
        Ok(elapsed) => elapsed,
        Err(e) => {
            // Some UUIDs are in the future, e.g. for dynamic evaluation runs.
            // In this case we should be conservative and assume no time has passed.
            if matches!(e.get_details(), ErrorDetails::UuidInFuture { .. }) {
                // We don't log anything, since this is an expected case.
                e.suppress_logging_of_error_message();
                Duration::from_secs(0)
            } else {
                return Err(e.log());
            }
        }
    };

    // Calculate the remaining cooldown (which may be zero) and ensure we wait at least FEEDBACK_MINIMUM_WAIT_TIME.
    let wait_time = max(
        FEEDBACK_COOLDOWN_PERIOD.saturating_sub(elapsed),
        FEEDBACK_MINIMUM_WAIT_TIME,
    );
    let deadline = Instant::now() + wait_time;

    // Poll every 500ms until the deadline is reached.
    loop {
        // If an error occurs during lookup (distinct from the target_id not existing), we bail out immediately.
        let feedback_target_info = db_client
            .get_function_info(target_id, metric_config_level.clone())
            .await?;
        match feedback_target_info {
            Some(feedback_target_info) => return Ok(feedback_target_info),
            None => {
                if Instant::now() >= deadline {
                    let identifier_type = match metric_config_level {
                        MetricConfigLevel::Inference => "Inference",
                        MetricConfigLevel::Episode => "Episode",
                    };
                    // We log here since this means we were not able to find the target_id in the database
                    // and are timing out.
                    return Err(Error::new(ErrorDetails::InvalidRequest {
                        message: format!("{identifier_type} ID: {target_id} does not exist"),
                    }));
                } else {
                    tracing::info!(
                        "Failed to find function name for target_id: {target_id}. Retrying..."
                    );
                }
            }
        }
        tokio::time::sleep(FEEDBACK_TARGET_POLL_INTERVAL).await;
    }
}

#[derive(Debug, Deserialize, PartialEq)]
struct DemonstrationToolCall {
    name: String,
    arguments: Value,
    /// Demonstration tool calls require an ID to match up with tool call responses. See #4058.
    id: String,
}

impl TryFrom<DemonstrationToolCall> for ToolCall {
    type Error = Error;
    fn try_from(value: DemonstrationToolCall) -> Result<Self, Self::Error> {
        Ok(ToolCall {
            name: value.name,
            arguments: serde_json::to_string(&value.arguments).map_err(|e| {
                Error::new(ErrorDetails::InvalidRequest {
                    message: format!("Failed to serialize demonstration tool call arguments: {e}"),
                })
            })?,
            id: value.id,
        })
    }
}

#[derive(Debug, PartialEq, TensorZeroDeserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
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
                                    message: format!("Invalid demonstration content block: {e}"),
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
                .map(DemonstrationContentBlock::try_into)
                .collect::<Result<Vec<ContentBlockOutput>, Error>>()?;
            let parsed_value = parse_chat_output(content_blocks, tool_call_config.as_ref(), None).await;
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
            JSONSchema::from_value(output_schema)?
                .validate(value)
                .await
                .map_err(|e| {
                    Error::new(ErrorDetails::InvalidRequest {
                        message: format!(
                            "Demonstration does not fit function output schema: {e}"
                        ),
                    })
                })?;
            let raw = serde_json::to_string(value).map_err(|e| {
                Error::new(ErrorDetails::InvalidRequest {
                    message: format!("Failed to serialize demonstration to json: {e}"),
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
    Chat(Option<ToolCallConfig>),
    Json(Value),
}

/// In order to properly validate demonstration data we need to fetch the information that was
/// passed to the inference at runtime.
/// If we don't do this then we might allow some e.g. tool calls or output schemas that would not
/// have been valid. Similarly, we might reject some tools or output schemas that were actually
/// valid.
/// This function grabs either the tool call or output schema information that was used at the
/// time of the actual inference in order to validate the demonstration data.
async fn get_dynamic_demonstration_info(
    db_client: &(dyn InferenceQueries + Sync),
    inference_id: Uuid,
    function_name: &str,
    function_config: &FunctionConfig,
    static_tools: &HashMap<String, Arc<StaticToolConfig>>,
) -> Result<DynamicDemonstrationInfo, Error> {
    match function_config {
        FunctionConfig::Chat(..) => {
            let tool_params = db_client
                .get_chat_inference_tool_params(function_name, inference_id)
                .await?;

            Ok(DynamicDemonstrationInfo::Chat(
                // If the tool params are not present in the database, we use the default tool params (empty tools).
                // This is consistent with how they are serialized at inference time.
                tool_params
                    .unwrap_or_default()
                    .into_tool_call_config(function_config, static_tools)?,
            ))
        }
        FunctionConfig::Json(..) => {
            let output_schema = db_client
                .get_json_inference_output_schema(function_name, inference_id)
                .await?
                .ok_or_else(|| {
                    Error::new(ErrorDetails::ClickHouseQuery {
                        message: "Failed to get output schema from demonstration result"
                            .to_string(),
                    })
                })?;

            let output_schema = if function_name.starts_with("tensorzero::llm_judge") {
                handle_llm_judge_output_schema(output_schema)
            } else {
                output_schema
            };
            Ok(DynamicDemonstrationInfo::Json(output_schema))
        }
    }
}

static OLD_LLM_JUDGE_OUTPUT_SCHEMA_FLOAT: OnceLock<Value> = OnceLock::new();
static OLD_LLM_JUDGE_OUTPUT_SCHEMA_BOOLEAN: OnceLock<Value> = OnceLock::new();
static NEW_LLM_JUDGE_OUTPUT_SCHEMA_FLOAT: OnceLock<Value> = OnceLock::new();
static NEW_LLM_JUDGE_OUTPUT_SCHEMA_BOOLEAN: OnceLock<Value> = OnceLock::new();

/// When we first introduced LLM Judges, we used a slightly different output schema
/// for them that explicitly included a "thinking" field.
/// We want to be able to take demonstrations that do not include this field.
/// This function handles the conversion of the old schema to the new schema for validation purposes.
fn handle_llm_judge_output_schema(output_schema: Value) -> Value {
    let old_float_schema = OLD_LLM_JUDGE_OUTPUT_SCHEMA_FLOAT.get_or_init(|| {
        json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["thinking", "score"],
            "additionalProperties": false,
            "properties": {
              "thinking": {
                "type": "string",
                "description": "The reasoning or thought process behind the judgment"
              },
              "score": {
                "type": "number",
                "description": "The score assigned as a number"
              }
            }
        })
    });

    let old_boolean_schema = OLD_LLM_JUDGE_OUTPUT_SCHEMA_BOOLEAN.get_or_init(|| {
        json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["thinking", "score"],
            "additionalProperties": false,
            "properties": {
              "thinking": {
                "type": "string",
                "description": "The reasoning or thought process behind the judgment"
              },
              "score": {
                "type": "boolean",
                "description": "The LLM judge's score as a boolean"
              }
            }
        })
    });

    let new_float_schema = NEW_LLM_JUDGE_OUTPUT_SCHEMA_FLOAT.get_or_init(|| {
        json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["score"],
            "additionalProperties": false,
            "properties": {
              "score": {
                "type": "number",
                "description": "The score assigned as a number"
              }
            }
        })
    });

    let new_boolean_schema = NEW_LLM_JUDGE_OUTPUT_SCHEMA_BOOLEAN.get_or_init(|| {
        json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["score"],
            "additionalProperties": false,
            "properties": {
              "score": {
                "type": "boolean",
                "description": "The LLM judge's score as a boolean"
              }
            }
        })
    });

    if output_schema == *old_float_schema {
        new_float_schema.clone()
    } else if output_schema == *old_boolean_schema {
        new_boolean_schema.clone()
    } else {
        output_schema
    }
}

fn validate_feedback_specific_tags(tags: &HashMap<String, String>) -> Result<(), Error> {
    if tags.contains_key("tensorzero::datapoint_id")
        && tags.contains_key("tensorzero::human_feedback")
        && !tags.contains_key("tensorzero::evaluator_inference_id")
    {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "tensorzero::evaluator_inference_id is required when tensorzero::datapoint_id and tensorzero::human_feedback are provided".to_string(),
        }));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    use crate::config::{Config, MetricConfig, MetricConfigOptimize, SchemaData};
    use crate::experimentation::ExperimentationConfig;
    use crate::function::{FunctionConfigChat, FunctionConfigJson};
    use crate::jsonschema_util::JSONSchema;
    use crate::testing::get_unit_test_gateway_handle;
    use crate::tool::{
        FunctionToolConfig, InferenceResponseToolCall, StaticToolConfig, ToolChoice,
    };

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
                description: None,
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
        let details = metadata.get_details();
        assert_eq!(
            *details,
            ErrorDetails::InvalidRequest {
                message: "Correct ID was not provided for feedback level \"inference\"."
                    .to_string(),
            }
        );
        // Case 1.3: ID provided but not for the correct level
        let metadata =
            get_feedback_metadata(&config, "test_metric", Some(Uuid::now_v7()), None).unwrap_err();
        let details = metadata.get_details();
        assert_eq!(
            *details,
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
        let details = metadata.get_details();
        assert_eq!(
            *details,
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
        let details = metadata.get_details();
        assert_eq!(
            *details,
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
        let details = metadata.get_details();
        assert_eq!(
            *details,
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
        let details = metadata.get_details();
        assert_eq!(
            *details,
            ErrorDetails::InvalidRequest {
                message: "Both episode_id and inference_id cannot be provided".to_string(),
            }
        );

        // Case 4.1 Boolean Feedback with episode level and episode id
        let metric_config = MetricConfig {
            r#type: MetricConfigType::Boolean,
            level: MetricConfigLevel::Episode,
            optimize: MetricConfigOptimize::Max,
            description: None,
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
        let details = metadata.get_details();
        assert_eq!(
            *details,
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
        let details = metadata_err.get_details();
        assert_eq!(
            *details,
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
        let gateway_handle = get_unit_test_gateway_handle(config);
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
        let response = feedback_handler(
            State(gateway_handle.app_state.clone()),
            None,
            StructuredJson(params),
        )
        .await;
        let error = response.unwrap_err();
        let details = error.get_details();
        assert_eq!(
            *details,
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
        let gateway_handle = get_unit_test_gateway_handle(config);
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
        let response = feedback_handler(
            State(gateway_handle.app_state.clone()),
            None,
            StructuredJson(params),
        )
        .await
        .unwrap_err();
        let details = response.get_details();
        assert_eq!(
            *details,
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
        let response = feedback_handler(
            State(gateway_handle.app_state.clone()),
            None,
            StructuredJson(params),
        )
        .await;
        let error = response.unwrap_err();
        let details = error.get_details();
        assert_eq!(
            *details,
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
                description: None,
            },
        );
        let config = Arc::new(Config {
            metrics,
            ..Default::default()
        });
        let gateway_handle = get_unit_test_gateway_handle(config);
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
        let response = feedback_handler(
            State(gateway_handle.app_state.clone()),
            None,
            StructuredJson(params),
        )
        .await
        .unwrap_err();
        let details = response.get_details();
        assert_eq!(
            *details,
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
        let response = feedback_handler(
            State(gateway_handle.app_state.clone()),
            None,
            StructuredJson(params),
        )
        .await;
        let error = response.unwrap_err();
        let details = error.get_details();
        assert_eq!(
            *details,
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
                description: None,
            },
        );
        let config = Arc::new(Config {
            metrics,
            ..Default::default()
        });
        let gateway_handle = get_unit_test_gateway_handle(config.clone());
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
        let response = feedback_handler(
            State(gateway_handle.app_state.clone()),
            None,
            StructuredJson(params),
        )
        .await;
        let error = response.unwrap_err();
        let details = error.get_details();
        assert_eq!(
            *details,
            ErrorDetails::InvalidRequest {
                message: format!("Inference ID: {inference_id} does not exist"),
            }
        );
    }

    #[tokio::test]
    async fn test_validate_parse_demonstration() {
        let weather_tool_config_static = StaticToolConfig {
            name: "get_temperature".to_string(),
            key: "get_temperature".to_string(),
            description: "Get the current temperature in a given location".to_string(),
            parameters: JSONSchema::from_value(json!({
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
                schemas: SchemaData::default(),
                tools: vec!["get_temperature".to_string()],
                tool_choice: ToolChoice::Auto,
                parallel_tool_calls: None,
                description: None,
                all_explicit_templates_names: HashSet::new(),
                experimentation: ExperimentationConfig::default(),
            })));

        // Case 1: a string passed to a chat function
        let value = json!("Hello, world!");
        let dynamic_demonstration_info =
            DynamicDemonstrationInfo::Chat(Some(ToolCallConfig::with_tools_available(
                tools
                    .values()
                    .cloned()
                    .map(FunctionToolConfig::Static)
                    .collect(),
                vec![],
            )));
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
        let value = json!([{"type": "tool_call", "id": "get_temperature_123", "name": "get_temperature", "arguments": {"location": "London", "unit": "celsius"}}]
        );
        let dynamic_demonstration_info =
            DynamicDemonstrationInfo::Chat(Some(ToolCallConfig::with_tools_available(
                tools
                    .values()
                    .cloned()
                    .map(FunctionToolConfig::Static)
                    .collect(),
                vec![],
            )));
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
        let expected_parsed_value = serde_json::to_string(&vec![ContentBlockChatOutput::ToolCall(
            InferenceResponseToolCall {
                id: "get_temperature_123".to_string(),
                name: Some("get_temperature".to_string()),
                raw_name: "get_temperature".to_string(),
                arguments: Some(json!({"location": "London", "unit": "celsius"})),
                raw_arguments: serde_json::to_string(
                    &json!({"location": "London", "unit": "celsius"}),
                )
                .unwrap(),
            },
        )])
        .unwrap();
        assert_eq!(expected_parsed_value, parsed_value);

        // Case 3: a tool call to get_humidity, which does not exist
        let value = json!([{"type": "tool_call", "id": "get_humidity_123", "name": "get_humidity", "arguments": {"location": "London", "unit": "celsius"}}]
        );
        let dynamic_demonstration_info =
            DynamicDemonstrationInfo::Chat(Some(ToolCallConfig::with_tools_available(
                tools
                    .values()
                    .cloned()
                    .map(FunctionToolConfig::Static)
                    .collect(),
                vec![],
            )));
        let err = validate_parse_demonstration(
            function_config_chat_tools,
            &value,
            dynamic_demonstration_info,
        )
        .await
        .unwrap_err();
        let details = err.get_details();
        assert_eq!(
            *details,
            ErrorDetails::InvalidRequest {
                message: "Demonstration contains invalid tool name".to_string(),
            }
        );

        // Case 4: a tool call to get_temperature, which exists but has bad arguments (place instead of location)
        let value = json!([{"type": "tool_call", "id": "get_temperature_123", "name": "get_temperature", "arguments": {"place": "London", "unit": "celsius"}}]
        );
        let dynamic_demonstration_info =
            DynamicDemonstrationInfo::Chat(Some(ToolCallConfig::with_tools_available(
                tools
                    .values()
                    .cloned()
                    .map(FunctionToolConfig::Static)
                    .collect(),
                vec![],
            )));
        let err = validate_parse_demonstration(
            function_config_chat_tools,
            &value,
            dynamic_demonstration_info,
        )
        .await
        .unwrap_err();
        let details = err.get_details();
        assert_eq!(
            *details,
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
        let json_mode_tool_call_config = ToolCallConfig::implicit_from_value(&output_schema);
        let function_config = Box::leak(Box::new(FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            output_schema: JSONSchema::from_value(output_schema.clone()).unwrap(),
            json_mode_tool_call_config,
            description: None,
            all_explicit_template_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
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
        assert!(
            err.to_string()
                .contains("Demonstration does not fit function output schema")
        );

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
        let details = err.get_details();
        assert_eq!(
            *details,
            ErrorDetails::Inference {
                message: "The DynamicDemonstrationInfo does not match the function type. This should never happen. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports.".to_string()
            }
        );

        // Case 8: Mismatched function type - JSON function with Chat demonstration info
        let value = json!({
            "name": "John",
            "age": 30
        });
        let dynamic_demonstration_info =
            DynamicDemonstrationInfo::Chat(Some(ToolCallConfig::with_tools_available(
                tools
                    .values()
                    .cloned()
                    .map(FunctionToolConfig::Static)
                    .collect(),
                vec![],
            )));
        let err = validate_parse_demonstration(function_config, &value, dynamic_demonstration_info)
            .await
            .unwrap_err();
        let details = err.get_details();
        assert_eq!(
            *details,
            ErrorDetails::Inference {
                message: "The DynamicDemonstrationInfo does not match the function type. This should never happen. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports.".to_string()
            }
        );
    }

    #[test]
    fn test_validate_feedback_specific_tags() {
        // Case 1: Empty tags should be valid
        let tags = HashMap::new();
        assert!(validate_feedback_specific_tags(&tags).is_ok());

        // Case 2: Tags with only datapoint_id should be valid
        let mut tags = HashMap::new();
        tags.insert("tensorzero::datapoint_id".to_string(), "123".to_string());
        assert!(validate_feedback_specific_tags(&tags).is_ok());

        // Case 3: Tags with only human_feedback should be valid
        let mut tags = HashMap::new();
        tags.insert("tensorzero::human_feedback".to_string(), "true".to_string());
        assert!(validate_feedback_specific_tags(&tags).is_ok());

        // Case 4: Tags with datapoint_id and human_feedback but without evaluator_inference_id should be invalid
        let mut tags = HashMap::new();
        tags.insert("tensorzero::datapoint_id".to_string(), "123".to_string());
        tags.insert("tensorzero::human_feedback".to_string(), "true".to_string());
        assert!(validate_feedback_specific_tags(&tags).is_err());
        let err = validate_feedback_specific_tags(&tags).unwrap_err();
        let details = err.get_details();
        assert_eq!(
            *details,
            ErrorDetails::InvalidRequest {
                message: "tensorzero::evaluator_inference_id is required when tensorzero::datapoint_id and tensorzero::human_feedback are provided".to_string(),
            }
        );

        // Case 5: Tags with all three required keys should be valid
        let mut tags = HashMap::new();
        tags.insert("tensorzero::datapoint_id".to_string(), "123".to_string());
        tags.insert("tensorzero::human_feedback".to_string(), "true".to_string());
        tags.insert(
            "tensorzero::evaluator_inference_id".to_string(),
            "456".to_string(),
        );
        assert!(validate_feedback_specific_tags(&tags).is_ok());
    }
}
