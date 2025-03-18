use anyhow::{anyhow, Result};
use futures::future::join_all;
use tensorzero::{DynamicToolParams, Input, InputMessage, InputMessageContent};
use tensorzero_internal::{
    function::FunctionConfig,
    inference::types::{
        ResolvedInput, ResolvedInputMessage, ResolvedInputMessageContent, TextKind,
    },
    tool::ToolCallConfigDatabaseInsert,
};

use crate::{Args, OutputFormat};

/// Convert a `ResolvedInput` to an `Input`.
///
/// This function is async as it will need to fetch data from object storage for
/// types that aren't serialized in ClickHouse.
pub async fn resolved_input_to_input(resolved_input: ResolvedInput) -> Result<Input> {
    let futures = resolved_input
        .messages
        .into_iter()
        .map(resolved_input_message_to_input_message);
    let message_results = join_all(futures).await;
    let messages = message_results.into_iter().collect::<Result<Vec<_>>>()?;
    let input = Input {
        system: resolved_input.system,
        messages,
    };
    Ok(input)
}

async fn resolved_input_message_to_input_message(
    resolved_input_message: ResolvedInputMessage,
) -> Result<InputMessage> {
    let content_futures = resolved_input_message
        .content
        .into_iter()
        .map(resolved_input_message_content_to_input_message_content);

    let content_results = join_all(content_futures).await;
    let content = content_results.into_iter().collect::<Result<Vec<_>>>()?;

    Ok(InputMessage {
        role: resolved_input_message.role,
        content,
    })
}

async fn resolved_input_message_content_to_input_message_content(
    resolved_input_message_content: ResolvedInputMessageContent,
) -> Result<InputMessageContent> {
    match resolved_input_message_content {
        ResolvedInputMessageContent::Text { value } => match value {
            serde_json::Value::String(s) => {
                Ok(InputMessageContent::Text(TextKind::Text { text: s }))
            }
            serde_json::Value::Object(o) => Ok(InputMessageContent::Text(TextKind::Arguments {
                arguments: o,
            })),
            _ => Err(anyhow::anyhow!("Unsupported text content: {:?}", value)),
        },
        ResolvedInputMessageContent::ToolCall(tool_call) => {
            Ok(InputMessageContent::ToolCall(tool_call))
        }
        ResolvedInputMessageContent::ToolResult(tool_result) => {
            Ok(InputMessageContent::ToolResult(tool_result))
        }
        ResolvedInputMessageContent::RawText { value } => {
            Ok(InputMessageContent::RawText { value })
        }
        ResolvedInputMessageContent::Thought(thought) => Ok(InputMessageContent::Thought(thought)),
        ResolvedInputMessageContent::Image(_image) => {
            // TODO: Implement image support for evals
            // This will involve grabbing the image from object storage using the object storage client included in `tensorzero-internal`
            Err(anyhow::anyhow!("Image not supported for evals (yet!)"))
        }
        ResolvedInputMessageContent::Unknown {
            data,
            model_provider_name,
        } => Ok(InputMessageContent::Unknown {
            data,
            model_provider_name,
        }),
    }
}

/// Given the function config for the eval and the tool call config that was written to the database,
/// recover the dynamic tool params that were used to generate the tool call config.
/// This will be used to help full out the params for the inference request in this eval.
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
