use super::throttled_get_function_info;
use crate::db::feedback::{FeedbackQueries, StaticEvaluationHumanFeedbackInsert};
use crate::db::inferences::{FunctionInfo, InferenceQueries};
use crate::{
    config::MetricConfigLevel,
    error::{Error, ErrorDetails},
};

use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

/// We maintain a table StaticEvaluationHumanFeedback (note: this is the actual database
/// table name, which retains "Static" prefix for backward compatibility even though
/// the feature is now called "Inference Evaluations") for datapoints which have had
/// humans label them so we can keep track of that and reuse it to "short-circuit"
/// future evaluations.
/// To do this at write time we need to make sure that that table is populated.
/// We saw memory issues in the materialized view 0028 so have dropped it.
/// Instead we read & write to the table here.
/// This is only necessary if: the feedback contains tags "tensorzero::human_feedback",
/// "tensorzero::evaluator_inference_id", and "tensorzero::datapoint_id": "uuid"
#[expect(clippy::too_many_arguments)]
pub(super) async fn write_static_evaluation_human_feedback_if_necessary(
    read_database: &(dyn InferenceQueries + Sync),
    write_database: &(dyn FeedbackQueries + Sync),
    maybe_function_info: Option<FunctionInfo>,
    metric_name: &str,
    tags: &HashMap<String, String>,
    feedback_id: Uuid,
    value: &Value,
    target_id: Uuid,
) -> Result<(), Error> {
    let Some(info) = get_static_evaluation_human_feedback_info(tags)? else {
        return Ok(());
    };
    let function_info = match maybe_function_info {
        Some(info) => info,
        None => {
            throttled_get_function_info(read_database, &MetricConfigLevel::Inference, &target_id)
                .await?
        }
    };
    let output = read_database
        .get_inference_output(&function_info, target_id)
        .await?
        .ok_or_else(|| {
            Error::new(ErrorDetails::InferenceNotFound {
                inference_id: target_id,
            })
        })?;
    let row = StaticEvaluationHumanFeedbackInsert {
        output,
        feedback_id,
        metric_name: metric_name.to_string(),
        value: serde_json::to_string(value)?,
        datapoint_id: info.datapoint_id,
        evaluator_inference_id: Some(info.evaluator_inference_id),
    };
    write_database.insert_static_eval_feedback(&row).await?;
    Ok(())
}

/// Extracts inference evaluation human feedback info from tags.
///
/// Returns `Ok(Some(InferenceEvaluationInfo))` if all 3 required tags are present:
/// - `tensorzero::datapoint_id` (must be a valid UUID)
/// - `tensorzero::evaluator_inference_id` (must be a valid UUID)
/// - `tensorzero::human_feedback` (any value, only presence is checked)
///
/// Returns `Ok(None)` if any of the 3 required tags are missing.
/// Returns `Err` if UUIDs are malformed.
fn get_static_evaluation_human_feedback_info(
    tags: &HashMap<String, String>,
) -> Result<Option<InferenceEvaluationInfo>, Error> {
    let Some(datapoint_id_str) = tags.get("tensorzero::datapoint_id") else {
        return Ok(None);
    };
    let Some(evaluator_inference_id_str) = tags.get("tensorzero::evaluator_inference_id") else {
        return Ok(None);
    };
    if !tags.contains_key("tensorzero::human_feedback") {
        return Ok(None);
    }
    let datapoint_id = Uuid::parse_str(datapoint_id_str).map_err(|e| {
        Error::new(ErrorDetails::InvalidTensorzeroUuid {
            kind: "datapoint".to_string(),
            message: format!("Invalid UUID: {e} (raw: {datapoint_id_str})"),
        })
    })?;
    let evaluator_inference_id = Uuid::parse_str(evaluator_inference_id_str).map_err(|e| {
        Error::new(ErrorDetails::InvalidTensorzeroUuid {
            kind: "evaluator_inference".to_string(),
            message: format!("Invalid UUID: {e} (raw: {evaluator_inference_id_str})"),
        })
    })?;

    Ok(Some(InferenceEvaluationInfo {
        datapoint_id,
        evaluator_inference_id,
    }))
}

#[derive(Debug)]
struct InferenceEvaluationInfo {
    datapoint_id: Uuid,
    evaluator_inference_id: Uuid,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use uuid::Uuid;

    #[test]
    fn test_get_static_evaluation_human_feedback_info_all_tags_present() {
        let datapoint_id = Uuid::now_v7();
        let evaluator_inference_id = Uuid::now_v7();

        let mut tags = HashMap::new();
        tags.insert(
            "tensorzero::datapoint_id".to_string(),
            datapoint_id.to_string(),
        );
        tags.insert(
            "tensorzero::evaluator_inference_id".to_string(),
            evaluator_inference_id.to_string(),
        );
        tags.insert("tensorzero::human_feedback".to_string(), "true".to_string());

        let result = get_static_evaluation_human_feedback_info(&tags).unwrap();

        assert!(result.is_some());
        let info = result.unwrap();
        assert_eq!(info.datapoint_id, datapoint_id);
        assert_eq!(info.evaluator_inference_id, evaluator_inference_id);
    }

    #[test]
    fn test_get_static_evaluation_human_feedback_info_missing_datapoint_id() {
        let evaluator_inference_id = Uuid::now_v7();

        let mut tags = HashMap::new();
        tags.insert(
            "tensorzero::evaluator_inference_id".to_string(),
            evaluator_inference_id.to_string(),
        );
        tags.insert("tensorzero::human_feedback".to_string(), "true".to_string());

        let result = get_static_evaluation_human_feedback_info(&tags).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_get_static_evaluation_human_feedback_info_missing_evaluator_inference_id() {
        let datapoint_id = Uuid::now_v7();

        let mut tags = HashMap::new();
        tags.insert(
            "tensorzero::datapoint_id".to_string(),
            datapoint_id.to_string(),
        );
        tags.insert("tensorzero::human_feedback".to_string(), "true".to_string());

        let result = get_static_evaluation_human_feedback_info(&tags).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_get_static_evaluation_human_feedback_info_missing_human_feedback() {
        let datapoint_id = Uuid::now_v7();
        let evaluator_inference_id = Uuid::now_v7();

        let mut tags = HashMap::new();
        tags.insert(
            "tensorzero::datapoint_id".to_string(),
            datapoint_id.to_string(),
        );
        tags.insert(
            "tensorzero::evaluator_inference_id".to_string(),
            evaluator_inference_id.to_string(),
        );

        let result = get_static_evaluation_human_feedback_info(&tags).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_get_static_evaluation_human_feedback_info_invalid_datapoint_uuid() {
        let evaluator_inference_id = Uuid::now_v7();

        let mut tags = HashMap::new();
        tags.insert(
            "tensorzero::datapoint_id".to_string(),
            "invalid-uuid".to_string(),
        );
        tags.insert(
            "tensorzero::evaluator_inference_id".to_string(),
            evaluator_inference_id.to_string(),
        );
        tags.insert("tensorzero::human_feedback".to_string(), "true".to_string());

        let result = get_static_evaluation_human_feedback_info(&tags);
        assert!(result.is_err());

        let error = result.unwrap_err();
        match error.get_details() {
            ErrorDetails::InvalidTensorzeroUuid { kind, message } => {
                assert_eq!(kind, "datapoint");
                assert!(message.contains("invalid-uuid"));
            }
            _ => panic!("Expected InvalidTensorzeroUuid error"),
        }
    }

    #[test]
    fn test_get_static_evaluation_human_feedback_info_invalid_evaluator_inference_uuid() {
        let datapoint_id = Uuid::now_v7();

        let mut tags = HashMap::new();
        tags.insert(
            "tensorzero::datapoint_id".to_string(),
            datapoint_id.to_string(),
        );
        tags.insert(
            "tensorzero::evaluator_inference_id".to_string(),
            "invalid-uuid".to_string(),
        );
        tags.insert("tensorzero::human_feedback".to_string(), "true".to_string());

        let result = get_static_evaluation_human_feedback_info(&tags);
        assert!(result.is_err());

        let error = result.unwrap_err();
        match error.get_details() {
            ErrorDetails::InvalidTensorzeroUuid { kind, message } => {
                assert_eq!(kind, "evaluator_inference");
                assert!(message.contains("invalid-uuid"));
            }
            _ => panic!("Expected InvalidTensorzeroUuid error"),
        }
    }

    #[test]
    fn test_get_static_evaluation_human_feedback_info_empty_tags() {
        let tags = HashMap::new();

        let result = get_static_evaluation_human_feedback_info(&tags).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_get_static_evaluation_human_feedback_info_human_feedback_value_variations() {
        let datapoint_id = Uuid::now_v7();
        let evaluator_inference_id = Uuid::now_v7();

        // Test different values for human_feedback tag - the function only checks presence, not value
        let test_values = vec!["true", "false", "", "some_value"];

        for value in test_values {
            let mut tags = HashMap::new();
            tags.insert(
                "tensorzero::datapoint_id".to_string(),
                datapoint_id.to_string(),
            );
            tags.insert(
                "tensorzero::evaluator_inference_id".to_string(),
                evaluator_inference_id.to_string(),
            );
            tags.insert("tensorzero::human_feedback".to_string(), value.to_string());

            let result = get_static_evaluation_human_feedback_info(&tags).unwrap();
            assert!(
                result.is_some(),
                "Should return Some regardless of human_feedback value: {value}"
            );
        }
    }

    #[test]
    fn test_get_static_evaluation_human_feedback_info_extra_tags() {
        let datapoint_id = Uuid::now_v7();
        let evaluator_inference_id = Uuid::now_v7();

        let mut tags = HashMap::new();
        tags.insert(
            "tensorzero::datapoint_id".to_string(),
            datapoint_id.to_string(),
        );
        tags.insert(
            "tensorzero::evaluator_inference_id".to_string(),
            evaluator_inference_id.to_string(),
        );
        tags.insert("tensorzero::human_feedback".to_string(), "true".to_string());
        tags.insert("extra_tag".to_string(), "extra_value".to_string());
        tags.insert("another::tag".to_string(), "another_value".to_string());

        let result = get_static_evaluation_human_feedback_info(&tags).unwrap();

        assert!(result.is_some());
        let info = result.unwrap();
        assert_eq!(info.datapoint_id, datapoint_id);
        assert_eq!(info.evaluator_inference_id, evaluator_inference_id);
    }
}
