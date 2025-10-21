use super::{throttled_get_function_info, FunctionInfo};
use crate::{
    config::MetricConfigLevel,
    db::clickhouse::{ClickHouseConnectionInfo, TableName},
    error::{Error, ErrorDetails},
};

use serde::{Deserialize, Serialize};
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
pub(super) async fn write_static_evaluation_human_feedback_if_necessary(
    clickhouse: &ClickHouseConnectionInfo,
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
            throttled_get_function_info(clickhouse, &MetricConfigLevel::Inference, &target_id)
                .await?
        }
    };
    let output = get_output(clickhouse, &function_info, target_id).await?;
    let row = StaticEvaluationHumanFeedback {
        output,
        feedback_id,
        metric_name: metric_name.to_string(),
        value: serde_json::to_string(value)?,
        datapoint_id: info.datapoint_id,
        evaluator_inference_id: Some(info.evaluator_inference_id),
    };
    clickhouse
        .write_batched(&[row], TableName::StaticEvaluationHumanFeedback)
        .await?;
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

async fn get_output(
    clickhouse: &ClickHouseConnectionInfo,
    function_info: &FunctionInfo,
    inference_id: Uuid,
) -> Result<String, Error> {
    let FunctionInfo {
        function_type,
        episode_id,
        name,
        variant_name,
    } = function_info;
    let table_name = function_type.inference_table_name();
    let output: OutputResponse = clickhouse
        .run_query_synchronous_no_params_de(format!(
            r"
    SELECT output FROM {table_name}
    WHERE
        id = '{inference_id}' AND
        episode_id = '{episode_id}' AND
        function_name = '{name}' AND
        variant_name = '{variant_name}'
    LIMIT 1
    FORMAT JSONEachRow
    SETTINGS max_threads=1"
        ))
        .await?;
    Ok(output.output)
}

/// This is so we're absolutely sure things are escaped properly.
#[derive(Debug, Deserialize)]
struct OutputResponse {
    output: String,
}

/// Represents a row in the StaticEvaluationHumanFeedback database table.
///
/// Note: The "Static" prefix is retained for backward compatibility with existing
/// database schemas. This feature is now called "Inference Evaluations" in the
/// product, configuration, and user-facing documentation.
#[derive(Debug, Deserialize, Serialize)]
pub struct StaticEvaluationHumanFeedback {
    pub metric_name: String,
    pub datapoint_id: Uuid,
    pub output: String,
    pub value: String,
    pub feedback_id: Uuid,
    pub evaluator_inference_id: Option<Uuid>,
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
