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

/// We maintain a table StaticEvaluationHumanFeedback for datapoints which
/// have had humans label them so we can keep track of that and reuse it to
/// "short-circuit" future evaluations.
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

// Returns None if **all 3** tags required are not present, then returns the datapoint id as a Uuid
fn get_static_evaluation_human_feedback_info(
    tags: &HashMap<String, String>,
) -> Result<Option<StaticEvaluationInfo>, Error> {
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

    Ok(Some(StaticEvaluationInfo {
        datapoint_id,
        evaluator_inference_id,
    }))
}

struct StaticEvaluationInfo {
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
    let response = clickhouse
        .run_query_synchronous_no_params(format!(
            r"
    SELECT output FROM {table_name}
    WHERE
        id = '{inference_id}' AND
        episode_id = '{episode_id}' AND
        function_name = '{name}' AND
        variant_name = '{variant_name}'
    LIMIT 1
    SETTINGS max_threads=1"
        ))
        .await?;
    let output = response.response;
    Ok(output)
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StaticEvaluationHumanFeedback {
    pub metric_name: String,
    pub datapoint_id: Uuid,
    pub output: String,
    pub value: String,
    pub feedback_id: Uuid,
    pub evaluator_inference_id: Option<Uuid>,
}
