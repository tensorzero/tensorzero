use anyhow::{anyhow, Result};
use tensorzero::{CacheParamsOptions, DynamicToolParams};
use tensorzero_internal::{
    cache::CacheEnabledMode, function::FunctionConfig, tool::ToolCallConfigDatabaseInsert,
};

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
            for tool in tool_params.tools_available.iter() {
                if !function_config.tools.contains(&tool.name) {
                    additional_tools.push(tool.clone());
                } else {
                    allowed_tools.push(tool.name.clone());
                }
            }
            DynamicToolParams {
                allowed_tools: Some(allowed_tools),
                additional_tools: Some(additional_tools),
                tool_choice: Some(tool_params.tool_choice.clone()),
                parallel_tool_calls: tool_params.parallel_tool_calls,
            }
        }
        // This branch is actually unreachable
        FunctionConfig::Json(_function_config) => DynamicToolParams {
            allowed_tools: None,
            additional_tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
        },
    }
}

pub fn setup_logging(args: &Args) -> Result<()> {
    match args.format {
        OutputFormat::Jsonl => {
            let subscriber = tracing_subscriber::FmtSubscriber::builder()
                .with_writer(std::io::stderr)
                .finish();
            tracing::subscriber::set_global_default(subscriber)
                .map_err(|e| anyhow!("Failed to initialize tracing: {}", e))
        }
        OutputFormat::HumanReadable => {
            let subscriber = tracing_subscriber::FmtSubscriber::new();
            tracing::subscriber::set_global_default(subscriber)
                .map_err(|e| anyhow!("Failed to initialize tracing: {}", e))
        }
    }
}

pub fn get_cache_options(skip_cache_read: bool) -> CacheParamsOptions {
    CacheParamsOptions {
        enabled: if !skip_cache_read {
            CacheEnabledMode::On
        } else {
            CacheEnabledMode::WriteOnly
        },
        max_age_s: None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;
    use tensorzero::Tool;
    use tensorzero_internal::{function::FunctionConfigChat, tool::ToolChoice};

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
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            tools: vec![],
            tool_choice: ToolChoice::Specific("tool_1".to_string()),
            parallel_tool_calls: None,
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
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            tools: vec!["tool_1".to_string()],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        });
        let tool_params_args = get_tool_params_args(&tool_database_insert, &function_config).await;
        assert_eq!(
            tool_params_args,
            DynamicToolParams {
                tool_choice: Some(ToolChoice::Required),
                parallel_tool_calls: None,
                allowed_tools: Some(vec!["tool_1".to_string()]),
                additional_tools: Some(vec![]),
            }
        );
    }
}
