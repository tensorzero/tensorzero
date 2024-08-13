use axum::extract::State;
use axum::{debug_handler, Json};
use metrics::counter;
use serde::Deserialize;
use serde_json::{json, Value};
use uuid::Uuid;

use crate::clickhouse::ClickHouseConnectionInfo;
use crate::config_parser::{Config, MetricConfigLevel, MetricConfigType, CONFIG};
use crate::error::Error;
use crate::gateway_util::{AppState, AppStateData, StructuredJson};

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
#[debug_handler(state = AppStateData)]
pub async fn feedback_handler(
    State(AppStateData {
        clickhouse_connection_info,
        ..
    }): AppState,
    StructuredJson(params): StructuredJson<Params>,
) -> Result<Json<Value>, Error> {
    // Get the metric config or return an error if it doesn't exist
    let feedback_metadata = get_feedback_metadata(
        &CONFIG,
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
                feedback_metadata.target_id,
                params.value,
                feedback_metadata.level,
                feedback_id,
                dryrun,
            )
            .await?
        }
        FeedbackType::Demonstration => {
            write_demonstration(
                clickhouse_connection_info,
                feedback_metadata.target_id,
                params.value,
                feedback_id,
                dryrun,
            )
            .await?
        }
        FeedbackType::Float => {
            write_float(
                clickhouse_connection_info,
                &params.metric_name,
                feedback_metadata.target_id,
                params.value,
                feedback_metadata.level,
                feedback_id,
                dryrun,
            )
            .await?
        }
        FeedbackType::Boolean => {
            write_boolean(
                clickhouse_connection_info,
                &params.metric_name,
                feedback_metadata.target_id,
                params.value,
                feedback_metadata.level,
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
        return Err(Error::InvalidRequest {
            message: "Both episode_id and inference_id cannot be provided".to_string(),
        });
    }
    let metric = config.get_metric(metric_name);
    let feedback_type = match metric.as_ref() {
        Ok(metric) => {
            let feedback_type: FeedbackType = (&metric.r#type).into();
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
        Ok(metric) => Ok(&metric.level),
        Err(_) => match feedback_type {
            FeedbackType::Demonstration => Ok(&MetricConfigLevel::Inference),
            _ => match (inference_id, episode_id) {
                (Some(_), None) => Ok(&MetricConfigLevel::Inference),
                (None, Some(_)) => Ok(&MetricConfigLevel::Episode),
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
    target_id: Uuid,
    value: Value,
    level: &MetricConfigLevel,
    feedback_id: Uuid,
    dryrun: bool,
) -> Result<(), Error> {
    let value = value.as_str().ok_or(Error::InvalidRequest {
        message: "Feedback value for a comment must be a string".to_string(),
    })?;
    let payload = json!({
        "target_type": level,
        "target_id": target_id,
        "value": value,
        "id": feedback_id
    });
    if !dryrun {
        connection_info.write(&payload, "CommentFeedback").await?;
    }
    Ok(())
}

async fn write_demonstration(
    connection_info: ClickHouseConnectionInfo,
    inference_id: Uuid,
    value: Value,
    feedback_id: Uuid,
    dryrun: bool,
) -> Result<(), Error> {
    let value = value.as_str().ok_or(Error::InvalidRequest {
        message: "Feedback value for a demonstration must be a string".to_string(),
    })?;
    let payload = json!({"inference_id": inference_id, "value": value, "id": feedback_id});
    if !dryrun {
        connection_info
            .write(&payload, "DemonstrationFeedback")
            .await?;
    }
    Ok(())
}

async fn write_float(
    connection_info: ClickHouseConnectionInfo,
    name: &str,
    target_id: Uuid,
    value: Value,
    level: &MetricConfigLevel,
    feedback_id: Uuid,
    dryrun: bool,
) -> Result<(), Error> {
    let value = value.as_f64().ok_or(Error::InvalidRequest {
        message: format!("Feedback value for metric `{name}` must be a number"),
    })?;
    let payload = json!({"target_type": level, "target_id": target_id, "value": value, "name": name, "id": feedback_id});
    if !dryrun {
        connection_info
            .write(&payload, "FloatMetricFeedback")
            .await?;
    }
    Ok(())
}

async fn write_boolean(
    connection_info: ClickHouseConnectionInfo,
    name: &str,
    target_id: Uuid,
    value: Value,
    level: &MetricConfigLevel,
    feedback_id: Uuid,
    dryrun: bool,
) -> Result<(), Error> {
    let value = value.as_bool().ok_or(Error::InvalidRequest {
        message: format!("Feedback value for metric `{name}` must be a boolean"),
    })?;
    let payload = json!({"target_type": level, "target_id": target_id, "value": value, "name": name, "id": feedback_id});
    if !dryrun {
        connection_info
            .write(&payload, "BooleanMetricFeedback")
            .await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use crate::config_parser::{Config, MetricConfig, MetricConfigOptimize};

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
            gateway: None,
            clickhouse: None,
            models: HashMap::new(),
            metrics,
            functions: HashMap::new(),
            tools: HashMap::new(),
        };
        let inference_id = Uuid::now_v7();
        let metadata =
            get_feedback_metadata(&config, "test_metric", None, Some(inference_id)).unwrap();
        assert_eq!(metadata.r#type, FeedbackType::Float);
        assert_eq!(metadata.level, &MetricConfigLevel::Inference);
        assert_eq!(metadata.target_id, inference_id);

        // Case 1.2: ID not provided
        let metadata = get_feedback_metadata(&config, "test_metric", None, None).unwrap_err();
        assert_eq!(
            metadata,
            Error::InvalidRequest {
                message: "Correct ID was not provided for feedback level \"inference\"."
                    .to_string(),
            }
        );
        // Case 1.3: ID provided but not for the correct level
        let metadata =
            get_feedback_metadata(&config, "test_metric", Some(Uuid::now_v7()), None).unwrap_err();
        assert_eq!(
            metadata,
            Error::InvalidRequest {
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
        assert_eq!(
            metadata,
            Error::InvalidRequest {
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
        assert_eq!(
            metadata,
            Error::InvalidRequest {
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
        assert_eq!(
            metadata,
            Error::InvalidRequest {
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
        assert_eq!(
            metadata,
            Error::InvalidRequest {
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
            gateway: None,
            clickhouse: None,
            models: HashMap::new(),
            metrics,
            functions: HashMap::new(),
            tools: HashMap::new(),
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
        assert_eq!(
            metadata,
            Error::InvalidRequest {
                message: "Both episode_id and inference_id cannot be provided".to_string(),
            }
        );
    }
}
