use std::collections::HashMap;

use anyhow::{anyhow, Result};
use serde::Deserialize;
use serde_json::Value;
use tensorzero::{CacheParamsOptions, InferenceResponse};
use tensorzero_core::db::clickhouse::escape_string_for_clickhouse_literal;
use tensorzero_core::serde_util::deserialize_json_string;
use tensorzero_core::{cache::CacheEnabledMode, db::clickhouse::ClickHouseConnectionInfo};
use tracing::debug;
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

use crate::{Args, OutputFormat};

pub fn setup_logging(args: &Args) -> Result<()> {
    match args.format {
        OutputFormat::Jsonl => {
            let subscriber = tracing_subscriber::FmtSubscriber::builder()
                .with_writer(std::io::stderr)
                .json()
                .with_env_filter(EnvFilter::from_default_env())
                .finish();
            tracing::subscriber::set_global_default(subscriber)
                .map_err(|e| anyhow!("Failed to initialize tracing: {e}"))
        }
        OutputFormat::Pretty => {
            let subscriber = tracing_subscriber::FmtSubscriber::builder()
                .with_writer(std::io::stderr)
                .with_env_filter(EnvFilter::from_default_env())
                .finish();
            tracing::subscriber::set_global_default(subscriber)
                .map_err(|e| anyhow!("Failed to initialize tracing: {e}"))
        }
    }
}

pub fn get_cache_options(inference_cache: CacheEnabledMode) -> CacheParamsOptions {
    CacheParamsOptions {
        enabled: inference_cache,
        max_age_s: None,
    }
}

#[derive(Debug, Deserialize)]
pub struct HumanFeedbackResult {
    #[serde(deserialize_with = "deserialize_json_string")]
    pub value: Value,
    pub evaluator_inference_id: Uuid,
}

pub async fn check_inference_evaluation_human_feedback(
    clickhouse: &ClickHouseConnectionInfo,
    metric_name: &str,
    datapoint_id: Uuid,
    inference_output: &InferenceResponse,
) -> Result<Option<HumanFeedbackResult>> {
    let serialized_output = inference_output.get_serialized_output()?;
    // Note: StaticEvaluationHumanFeedback is the actual database table name,
    // retained for backward compatibility even though this feature is now
    // called "Inference Evaluations" in the product and user-facing documentation.
    let query = r"
        SELECT value, evaluator_inference_id FROM StaticEvaluationHumanFeedback
        WHERE
            metric_name = {metric_name:String}
        AND datapoint_id = {datapoint_id:UUID}
        AND output = {output:String}
        ORDER BY timestamp DESC
        LIMIT 1
        FORMAT JSONEachRow";
    debug!(query = %query, "Executing ClickHouse query");
    let escaped_serialized_output = escape_string_for_clickhouse_literal(&serialized_output);
    let result = clickhouse
        .run_query_synchronous(
            query.to_string(),
            &HashMap::from([
                ("metric_name", metric_name),
                ("datapoint_id", &datapoint_id.to_string()),
                ("output", &escaped_serialized_output),
            ]),
        )
        .await?;
    debug!(
        result_length = result.response.len(),
        "Query executed successfully"
    );
    if result.response.is_empty() {
        return Ok(None);
    }
    let human_feedback_result: HumanFeedbackResult = serde_json::from_str(&result.response)
        .map_err(|e| anyhow!("Failed to parse human feedback result: {e}"))?;
    Ok(Some(human_feedback_result))
}
