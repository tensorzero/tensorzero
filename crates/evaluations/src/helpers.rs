use anyhow::{Result, anyhow};
use tensorzero_core::cache::{CacheEnabledMode, CacheParamsOptions};
use tracing_subscriber::EnvFilter;

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
            vec![Tool::Function(FunctionTool {
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
                additional_tools: Some(vec![Tool::Function(FunctionTool {
                    name: "tool_1".to_string(),
                    description: "Tool 1".to_string(),
                    parameters: json!({}),
                    strict: true,
                })]),
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
