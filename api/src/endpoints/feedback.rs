use axum::debug_handler;
use axum::extract::State;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use uuid::Uuid;

use crate::api_util::{AppState, AppStateData, StructuredJson};
use crate::clickhouse::{clickhouse_write, ClickHouseConnectionInfo};
use crate::config_parser::{Config, MetricConfigLevel, MetricConfigType};
use crate::error::Error;

// TODO: function or function_name or ...? variant?

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
    // if true, the feedback will not be stored
    dryrun: Option<bool>,
}

#[derive(Debug)]
enum FeedbackType {
    Comment,
    Demonstration,
    Float,
    Boolean,
}

impl From<MetricConfigType> for FeedbackType {
    fn from(value: MetricConfigType) -> Self {
        match value {
            MetricConfigType::Float => FeedbackType::Float,
            MetricConfigType::Boolean => FeedbackType::Boolean,
        }
    }
}

/// A handler for the feedback endpoint
#[debug_handler(state = AppStateData)]
pub async fn feedback_handler(
    State(AppStateData {
        config,
        http_client,
        clickhouse_connection_info,
    }): AppState,
    StructuredJson(params): StructuredJson<Params>,
) -> Result<(), Error> {
    // Get the metric config or return an error if it doesn't exist
    let feedback_metadata = get_feedback_metadata(
        &config,
        &params.metric_name,
        params.episode_id,
        params.inference_id,
    )?;
    match feedback_metadata.r#type {
        FeedbackType::Comment => {
            write_comment(
                http_client,
                clickhouse_connection_info,
                feedback_metadata.target_id,
                params.value,
                feedback_metadata.level,
                params.dryrun.unwrap_or(false),
            )
            .await?
        }
        FeedbackType::Demonstration => {
            write_demonstration(
                http_client,
                clickhouse_connection_info,
                feedback_metadata.target_id,
                params.value,
                params.dryrun.unwrap_or(false),
            )
            .await?
        }
        FeedbackType::Float => {
            write_float(
                http_client,
                clickhouse_connection_info,
                &params.metric_name,
                feedback_metadata.target_id,
                params.value,
                feedback_metadata.level,
                params.dryrun.unwrap_or(false),
            )
            .await?
        }
        FeedbackType::Boolean => {
            write_boolean(
                http_client,
                clickhouse_connection_info,
                &params.metric_name,
                feedback_metadata.target_id,
                params.value,
                feedback_metadata.level,
                params.dryrun.unwrap_or(false),
            )
            .await?
        }
    }
    Ok(())
}

struct FeedbackMetadata {
    r#type: FeedbackType,
    level: MetricConfigLevel,
    target_id: Uuid,
}

fn get_feedback_metadata(
    config: &Config,
    metric_name: &str,
    episode_id: Option<Uuid>,
    inference_id: Option<Uuid>,
) -> Result<FeedbackMetadata, Error> {
    let metric = config.get_metric(metric_name);
    let feedback_type = match metric.as_ref() {
        Ok(metric) => {
            let feedback_type: FeedbackType = metric.r#type.clone().into();
            Ok(feedback_type)
        }
        Err(e) => match metric_name {
            "comment" => Ok(FeedbackType::Comment),
            "demonstration" => Ok(FeedbackType::Demonstration),
            _ => Err(Error::InvalidRequest {
                message: e.to_string(),
            }),
        },
    }?;
    let feedback_level = match metric {
        Ok(metric) => Ok(metric.level.clone()),
        Err(_) => match feedback_type {
            FeedbackType::Demonstration => Ok(MetricConfigLevel::Inference),
            _ => match (inference_id, episode_id) {
                (Some(_), None) => Ok(MetricConfigLevel::Inference),
                (None, Some(_)) => Ok(MetricConfigLevel::Episode),
                _ => Err(Error::InvalidRequest {
                    message: "Exactly one of inference_id or episode_id must be provided"
                        .to_string(),
                }),
            },
        },
    }?;
    let target_id = match feedback_level {
        MetricConfigLevel::Inference => inference_id,
        MetricConfigLevel::Episode => episode_id,
    }
    .ok_or(Error::InvalidRequest {
        message: "No feedback or episode ID provided".to_string(),
    })?;
    Ok(FeedbackMetadata {
        r#type: feedback_type,
        level: feedback_level,
        target_id,
    })
}

async fn write_comment(
    client: Client,
    connection_info: ClickHouseConnectionInfo,
    target_id: Uuid,
    value: Value,
    level: MetricConfigLevel,
    dryrun: bool,
) -> Result<(), Error> {
    let value = value.as_str().ok_or(Error::InvalidRequest {
        message: "Value for comment must be a string".to_string(),
    })?;
    let payload = json!({"target_type": level, "target_id": target_id, "value": value});
    if !dryrun {
        clickhouse_write(&client, &connection_info, &payload, "CommentFeedback").await?;
    }
    Ok(())
}

async fn write_demonstration(
    client: Client,
    connection_info: ClickHouseConnectionInfo,
    inference_id: Uuid,
    value: Value,
    dryrun: bool,
) -> Result<(), Error> {
    let value = value.as_str().ok_or(Error::InvalidRequest {
        message: "Value for demonstration must be a string".to_string(),
    })?;
    let payload = json!({"inference_id": inference_id, "value": value});
    if !dryrun {
        clickhouse_write(&client, &connection_info, &payload, "DemonstrationFeedback").await?;
    }
    Ok(())
}

async fn write_float(
    client: Client,
    connection_info: ClickHouseConnectionInfo,
    name: &str,
    target_id: Uuid,
    value: Value,
    level: MetricConfigLevel,
    dryrun: bool,
) -> Result<(), Error> {
    let value = value.as_f64().ok_or(Error::InvalidRequest {
        message: "Value for float must be a number".to_string(),
    })?;
    let payload =
        json!({"target_type": level, "target_id": target_id, "value": value, "name": name});
    if !dryrun {
        clickhouse_write(&client, &connection_info, &payload, "FloatMetricFeedback").await?;
    }
    Ok(())
}

async fn write_boolean(
    client: Client,
    connection_info: ClickHouseConnectionInfo,
    name: &str,
    target_id: Uuid,
    value: Value,
    level: MetricConfigLevel,
    dryrun: bool,
) -> Result<(), Error> {
    let value = value.as_bool().ok_or(Error::InvalidRequest {
        message: "Value for boolean must be a boolean".to_string(),
    })?;
    let payload =
        json!({"target_type": level, "target_id": target_id, "value": value, "name": name});
    if !dryrun {
        clickhouse_write(&client, &connection_info, &payload, "BooleanMetricFeedback").await?;
    }
    Ok(())
}
