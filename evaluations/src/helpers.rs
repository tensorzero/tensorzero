use std::collections::HashMap;

use anyhow::{anyhow, Result};
use serde::Deserialize;
use serde_json::Value;
use tensorzero_core::cache::CacheParamsOptions;
use tensorzero_core::client::InferenceResponse;
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

#[cfg(test)]
mod tests {
    use serde_json::json;
    use tensorzero_core::tool::{
        AllowedTools, AllowedToolsChoice, DynamicToolParams, FunctionTool, Tool,
        ToolCallConfigDatabaseInsert, ToolChoice,
    };

    #[tokio::test]
    async fn test_get_tool_params_args() {
        // Dynamic tool params with tool_choice set to "tool_1"
        // Function has no tools, tool_1 is provided dynamically
        let tool_database_insert = ToolCallConfigDatabaseInsert::new_for_test(
            vec![Tool::ClientSideFunction(FunctionTool {
                name: "tool_1".to_string(),
                description: "Tool 1".to_string(),
                parameters: json!({}),
                strict: true,
            })],
            vec![],
            AllowedTools {
                tools: vec![], // Explicitly empty (no static tools)
                choice: AllowedToolsChoice::Explicit,
            },
            ToolChoice::Specific("tool_1".to_string()),
            None,
        );
        let tool_params_args: DynamicToolParams = tool_database_insert.into();
        assert_eq!(
            tool_params_args,
            DynamicToolParams {
                tool_choice: Some(ToolChoice::Specific("tool_1".to_string())),
                parallel_tool_calls: None,
                allowed_tools: Some(Vec::new()),
                additional_tools: Some(vec![FunctionTool {
                    name: "tool_1".to_string(),
                    description: "Tool 1".to_string(),
                    parameters: json!({}),
                    strict: true,
                }]),
                provider_tools: vec![],
            }
        );

        // Static tool params with a tool choice set to required
        // tool_1 is in function config (static), so don't store in dynamic_tools
        let tool_database_insert = ToolCallConfigDatabaseInsert::new_for_test(
            vec![], // Empty - tool_1 is static
            vec![],
            AllowedTools {
                tools: vec!["tool_1".to_string()],
                choice: AllowedToolsChoice::Explicit, // Explicit list
            },
            ToolChoice::Required,
            None,
        );
        let tool_params_args: DynamicToolParams = tool_database_insert.into();
        assert_eq!(
            tool_params_args,
            DynamicToolParams {
                tool_choice: Some(ToolChoice::Required),
                parallel_tool_calls: None,
                allowed_tools: Some(vec!["tool_1".to_string()]),
                additional_tools: None, // Empty dynamic_tools becomes None
                provider_tools: vec![],
            }
        );
    }
}
