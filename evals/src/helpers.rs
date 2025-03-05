use anyhow::Result;
use futures::future::join_all;
use tensorzero::{DynamicToolParams, Input, InputMessage, InputMessageContent};
use tensorzero_internal::{
    function::FunctionConfig,
    inference::types::{ResolvedInput, ResolvedInputMessage, ResolvedInputMessageContent},
    tool::ToolCallConfigDatabaseInsert,
};

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
        ResolvedInputMessageContent::Text { value } => Ok(InputMessageContent::Text { value }),
        ResolvedInputMessageContent::ToolCall(tool_call) => {
            Ok(InputMessageContent::ToolCall(tool_call))
        }
        ResolvedInputMessageContent::ToolResult(tool_result) => {
            Ok(InputMessageContent::ToolResult(tool_result))
        }
        ResolvedInputMessageContent::RawText { value } => {
            Ok(InputMessageContent::RawText { value })
        }
        ResolvedInputMessageContent::Image(_image) => {
            // TODO: Implement image support for evals
            // This will involve grabbing the image from object storage using the object storage client included in `tensorzero-internal`
            Err(anyhow::anyhow!("Image not supported for evals (yet!)"))
        }
    }
}

/// Given the function config for the eval and the tool call config that was written to the database,
/// recover the dynamic tool params that were used to generate the tool call config.
/// This will be used to help full out the params for the inference request in this eval.
/// TODO (Viraj): test this function
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
                parallel_tool_calls: Some(tool_params.parallel_tool_calls),
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
