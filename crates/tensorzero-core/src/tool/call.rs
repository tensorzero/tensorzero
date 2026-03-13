//! Tool call response types.
//!
//! This module contains extension traits and types for tool call responses:
//! - `InferenceResponseToolCallExt` - Extension trait for `InferenceResponseToolCall`
//! - `ToolCallChunk` - A streaming chunk of a tool call

use tensorzero_provider_types::ProviderToolCallConfig;

use super::config::ToolCallConfig;
use super::wire::ToolCall;

// Re-export from tensorzero-types
pub use tensorzero_types::InferenceResponseToolCall;

/// Extension trait for `InferenceResponseToolCall` providing core-specific functionality.
pub trait InferenceResponseToolCallExt {
    /// Validates that a ToolCall is compliant with the ToolCallConfig.
    /// First, it finds the ToolConfig for the ToolCall.
    /// Then, it validates the ToolCall arguments against the ToolConfig.
    async fn new_from_tool_call(
        tool_call: ToolCall,
        tool_cfg: Option<&ToolCallConfig>,
    ) -> InferenceResponseToolCall;

    /// Like `new_from_tool_call` but takes a `ProviderToolCallConfig`.
    /// Does not perform JSON schema validation (since `ProviderToolCallConfig` does not carry
    /// compiled schemas); instead validates that the tool name is known and arguments are valid JSON.
    async fn new_from_provider_tool_call(
        tool_call: ToolCall,
        tool_cfg: Option<&ProviderToolCallConfig>,
    ) -> InferenceResponseToolCall;
}

impl InferenceResponseToolCallExt for InferenceResponseToolCall {
    async fn new_from_tool_call(
        tool_call: ToolCall,
        tool_cfg: Option<&ToolCallConfig>,
    ) -> InferenceResponseToolCall {
        // Check if this is a function tool
        let function_tool = tool_cfg.and_then(|t| t.get_function_tool(&tool_call.name));

        // Check if this is a custom tool
        let is_custom_tool = tool_cfg
            .map(|t| {
                t.openai_custom_tools
                    .iter()
                    .any(|ct| ct.name == tool_call.name)
            })
            .unwrap_or(false);

        // Set parsed_name if tool exists (either function or custom)
        let parsed_name = if function_tool.is_some() || is_custom_tool {
            Some(tool_call.name.clone())
        } else {
            None
        };

        // Validate arguments only for function tools (custom tools don't use JSON schemas)
        let parsed_arguments = if let Some(tool) = function_tool {
            if let Ok(arguments) = serde_json::from_str(&tool_call.arguments) {
                if tool.validate_arguments(&arguments).await.is_ok() {
                    Some(arguments)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        InferenceResponseToolCall {
            arguments: parsed_arguments,
            id: tool_call.id,
            name: parsed_name,
            raw_arguments: tool_call.arguments.clone(),
            raw_name: tool_call.name.clone(),
        }
    }

    async fn new_from_provider_tool_call(
        tool_call: ToolCall,
        tool_cfg: Option<&ProviderToolCallConfig>,
    ) -> InferenceResponseToolCall {
        // Check if this is a function tool
        let function_tool = tool_cfg.and_then(|t| t.get_function_tool(&tool_call.name));

        // Check if this is a custom tool
        let is_custom_tool = tool_cfg
            .map(|t| {
                t.openai_custom_tools
                    .iter()
                    .any(|ct| ct.name == tool_call.name)
            })
            .unwrap_or(false);

        // Set parsed_name if tool exists (either function or custom)
        let parsed_name = if function_tool.is_some() || is_custom_tool {
            Some(tool_call.name.clone())
        } else {
            None
        };

        // Validate arguments as JSON (no schema validation available for ProviderToolCallConfig)
        let parsed_arguments = if function_tool.is_some() {
            serde_json::from_str(&tool_call.arguments).ok()
        } else {
            None
        };

        InferenceResponseToolCall {
            arguments: parsed_arguments,
            id: tool_call.id,
            name: parsed_name,
            raw_arguments: tool_call.arguments.clone(),
            raw_name: tool_call.name.clone(),
        }
    }
}

pub use tensorzero_provider_types::ToolCallChunk;
