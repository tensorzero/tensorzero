use std::collections::HashMap;

use anyhow::{anyhow, Result};
use serde::Deserialize;
use serde_json::Value;
use tensorzero_core::cache::CacheParamsOptions;
use tensorzero_core::client::{DynamicToolParams, InferenceResponse};
use tensorzero_core::db::clickhouse::escape_string_for_clickhouse_literal;
use tensorzero_core::serde_util::deserialize_json_string;
use tensorzero_core::{
    cache::CacheEnabledMode, db::clickhouse::ClickHouseConnectionInfo, function::FunctionConfig,
    tool::ToolCallConfigDatabaseInsert,
};
use tracing::debug;
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

use crate::{Args, OutputFormat};

/// Given the function config for the evaluation and the tool call config that was written to the database,
/// recover the dynamic tool params that were used to generate the tool call config.
/// This will be used to help full out the params for the inference request in this evaluation.
pub async fn get_tool_params_args(
    tool_params: &ToolCallConfigDatabaseInsert,
    function_config: &FunctionConfig,
) -> DynamicToolParams {
    match function_config {
        FunctionConfig::Chat(function_config) => {
            let mut additional_tools = Vec::new();
            let mut allowed_tools = Vec::new();
            for tool in &tool_params.tools_available {
                if function_config.tools.contains(&tool.name) {
                    allowed_tools.push(tool.name.clone());
                } else {
                    additional_tools.push(tool.clone());
                }
            }
            DynamicToolParams {
                allowed_tools: Some(allowed_tools),
                additional_tools: Some(additional_tools),
                tool_choice: Some(tool_params.tool_choice.clone()),
                parallel_tool_calls: tool_params.parallel_tool_calls,
                // TODO (Viraj): once we have this stored in the database, be sure to add it
                provider_tools: None,
            }
        }
        // This branch is actually unreachable
        FunctionConfig::Json(_function_config) => DynamicToolParams {
            allowed_tools: None,
            additional_tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            provider_tools: None,
        },
    }
}

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
    use std::collections::{HashMap, HashSet};

    use serde_json::json;
    use tensorzero_core::client::Tool;
    use tensorzero_core::{
        config::SchemaData, experimentation::ExperimentationConfig, function::FunctionConfigChat,
        tool::ToolChoice,
    };

    use super::*;

    #[tokio::test]
    async fn test_get_tool_params_args() {
        // Dynamic tool params with tool_choice set to "tool_1"
        let tool_database_insert = ToolCallConfigDatabaseInsert {
            tool_choice: ToolChoice::Specific("tool_1".to_string()),
            parallel_tool_calls: None,
            tools_available: vec![Tool {
                name: "tool_1".to_string(),
                description: "Tool 1".to_string(),
                parameters: json!({}),
                strict: true,
            }],
        };
        let function_config = FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            tools: vec![],
            tool_choice: ToolChoice::Specific("tool_1".to_string()),
            parallel_tool_calls: None,
            description: None,
            all_explicit_templates_names: HashSet::new(),
            experimentation: ExperimentationConfig::legacy_from_variants_map(&HashMap::new()),
        });
        let tool_params_args = get_tool_params_args(&tool_database_insert, &function_config).await;
        assert_eq!(
            tool_params_args,
            DynamicToolParams {
                tool_choice: Some(ToolChoice::Specific("tool_1".to_string())),
                parallel_tool_calls: None,
                allowed_tools: Some(Vec::new()),
                additional_tools: Some(vec![Tool {
                    name: "tool_1".to_string(),
                    description: "Tool 1".to_string(),
                    parameters: json!({}),
                    strict: true,
                }]),
                provider_tools: None,
            }
        );

        // Static tool params with a tool choice set to required
        let tool_database_insert = ToolCallConfigDatabaseInsert {
            tool_choice: ToolChoice::Required,
            parallel_tool_calls: None,
            tools_available: vec![Tool {
                name: "tool_1".to_string(),
                description: "Tool 1".to_string(),
                parameters: json!({}),
                strict: true,
            }],
        };
        let function_config = FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            tools: vec!["tool_1".to_string()],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
            description: None,
            all_explicit_templates_names: HashSet::new(),
            experimentation: ExperimentationConfig::legacy_from_variants_map(&HashMap::new()),
        });
        let tool_params_args = get_tool_params_args(&tool_database_insert, &function_config).await;
        assert_eq!(
            tool_params_args,
            DynamicToolParams {
                tool_choice: Some(ToolChoice::Required),
                parallel_tool_calls: None,
                allowed_tools: Some(vec!["tool_1".to_string()]),
                additional_tools: Some(vec![]),
                provider_tools: None,
            }
        );
    }
}
