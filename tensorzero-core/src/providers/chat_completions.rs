//! Common types and utilities for chat completions across OpenAI-compatible providers.
//!
//! This module centralizes the modal chat completions API implementation,
//! providing shared types and helper functions for preparing tools and tool choices
//! in OpenAI's chat completions format. These types are used by providers such as
//! OpenAI, Azure, Groq, and OpenRouter.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::Error;
use crate::inference::types::ModelInferenceRequest;
use crate::tool::{FunctionTool, FunctionToolConfig, ToolChoice};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatCompletionToolType {
    Function,
}

#[derive(Debug, PartialEq, Serialize)]
pub struct ChatCompletionFunction<'a> {
    pub name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<&'a str>,
    pub parameters: &'a Value,
}

#[derive(Debug, PartialEq, Serialize)]
pub struct ChatCompletionTool<'a> {
    pub r#type: ChatCompletionToolType,
    pub function: ChatCompletionFunction<'a>,
    pub strict: bool,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ChatCompletionToolChoice<'a> {
    String(ChatCompletionToolChoiceString),
    Specific(ChatCompletionSpecificToolChoice<'a>),
    AllowedTools(ChatCompletionAllowedToolsChoice<'a>),
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatCompletionToolChoiceString {
    None,
    Auto,
    Required,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct ChatCompletionSpecificToolChoice<'a> {
    pub r#type: ChatCompletionToolType,
    pub function: ChatCompletionSpecificToolFunction<'a>,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct ChatCompletionSpecificToolFunction<'a> {
    pub name: &'a str,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct ChatCompletionAllowedToolsChoice<'a> {
    pub r#type: &'static str, // Always "allowed_tools"
    pub allowed_tools: ChatCompletionAllowedToolsConstraint<'a>,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct ChatCompletionAllowedToolsConstraint<'a> {
    pub mode: ChatCompletionAllowedToolsMode,
    pub tools: Vec<ChatCompletionToolReference<'a>>,
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ChatCompletionAllowedToolsMode {
    Auto,
    Required,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct ChatCompletionToolReference<'a> {
    pub r#type: ChatCompletionToolType,
    pub function: ChatCompletionSpecificToolFunction<'a>,
}

// Type alias for the return type of prepare_chat_completion_tools
type PreparedChatCompletionToolsResult<'a> = (
    Option<Vec<ChatCompletionTool<'a>>>,
    Option<ChatCompletionToolChoice<'a>>,
    Option<bool>,
);

impl<'a> From<&'a FunctionToolConfig> for ChatCompletionTool<'a> {
    fn from(tool: &'a FunctionToolConfig) -> Self {
        ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: ChatCompletionFunction {
                name: tool.name(),
                description: Some(tool.description()),
                parameters: tool.parameters(),
            },
            strict: tool.strict(),
        }
    }
}

impl<'a> From<&'a FunctionTool> for ChatCompletionTool<'a> {
    fn from(tool: &'a FunctionTool) -> Self {
        ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: ChatCompletionFunction {
                name: &tool.name,
                description: Some(&tool.description),
                parameters: &tool.parameters,
            },
            strict: tool.strict,
        }
    }
}

impl<'a> From<&'a ToolChoice> for ChatCompletionToolChoice<'a> {
    fn from(tool_choice: &'a ToolChoice) -> Self {
        match tool_choice {
            ToolChoice::None => {
                ChatCompletionToolChoice::String(ChatCompletionToolChoiceString::None)
            }
            ToolChoice::Auto => {
                ChatCompletionToolChoice::String(ChatCompletionToolChoiceString::Auto)
            }
            ToolChoice::Required => {
                ChatCompletionToolChoice::String(ChatCompletionToolChoiceString::Required)
            }
            ToolChoice::Specific(tool_name) => {
                ChatCompletionToolChoice::Specific(ChatCompletionSpecificToolChoice {
                    r#type: ChatCompletionToolType::Function,
                    function: ChatCompletionSpecificToolFunction { name: tool_name },
                })
            }
        }
    }
}

/// Helper function to prepare allowed_tools constraint when dynamic allowed_tools are set.
/// This returns the ChatCompletionAllowedToolsChoice struct with the appropriate mode and tool references.
///
/// This is shared logic across OpenAI-compatible providers that support allowed_tools (Azure, Groq, OpenRouter).
fn prepare_chat_completion_allowed_tools_constraint<'a>(
    tool_config: &'a crate::tool::ToolCallConfig,
) -> Option<ChatCompletionAllowedToolsChoice<'a>> {
    // OpenAI-compatible providers don't allow both tool-choice "none" and tool-choice "allowed_tools",
    // since they're both set via the top-level "tool_choice" field.
    // We make `ToolChoice::None` take priority - that is, we allow "none" of the allowed tools.
    if tool_config.tool_choice == ToolChoice::None {
        return None;
    }
    let allowed_tools_list = tool_config.allowed_tools.as_dynamic_allowed_tools()?;

    // Construct the OpenAI spec-compliant allowed_tools structure
    let mode = match &tool_config.tool_choice {
        ToolChoice::Required => ChatCompletionAllowedToolsMode::Required,
        _ => ChatCompletionAllowedToolsMode::Auto,
    };

    let tool_refs: Vec<ChatCompletionToolReference> = allowed_tools_list
        .iter()
        .map(|name| ChatCompletionToolReference {
            r#type: ChatCompletionToolType::Function,
            function: ChatCompletionSpecificToolFunction { name },
        })
        .collect();

    Some(ChatCompletionAllowedToolsChoice {
        r#type: "allowed_tools",
        allowed_tools: ChatCompletionAllowedToolsConstraint {
            mode,
            tools: tool_refs,
        },
    })
}

/// Prepares tools and tool choice for chat completion requests in OpenAI-compatible format
///
/// If there are no tools passed or the tools are empty, return None for both tools and tool_choice
/// Otherwise convert the tool choice and tools to chat completion format
///
/// When supports_allowed_tools is true, this function will use the allowed_tools constraint
/// for providers that support OpenAI's allowed_tools format (Azure, Groq, OpenRouter).
/// When false, it will filter tools using strict_tools_available for providers that don't support it.
pub fn prepare_chat_completion_tools<'a>(
    request: &'a ModelInferenceRequest,
    supports_allowed_tools: bool,
) -> Result<PreparedChatCompletionToolsResult<'a>, Error> {
    match &request.tool_config {
        None => Ok((None, None, None)),
        Some(tool_config) => {
            if !tool_config.any_tools_available() {
                return Ok((None, None, None));
            }

            let parallel_tool_calls = tool_config.parallel_tool_calls;

            if supports_allowed_tools {
                // Provider supports OpenAI's allowed_tools constraint
                // Send all tools and use allowed_tools in tool_choice if needed
                let tools = Some(tool_config.tools_available()?.map(Into::into).collect());

                let tool_choice = if let Some(allowed_tools_choice) =
                    prepare_chat_completion_allowed_tools_constraint(tool_config)
                {
                    Some(ChatCompletionToolChoice::AllowedTools(allowed_tools_choice))
                } else {
                    // No allowed_tools constraint, use regular tool_choice
                    Some((&tool_config.tool_choice).into())
                };

                Ok((tools, tool_choice, parallel_tool_calls))
            } else {
                // Provider doesn't support allowed_tools constraint
                // Filter tools using strict_tools_available and use regular tool_choice
                let tools = Some(
                    tool_config
                        .strict_tools_available()?
                        .map(Into::into)
                        .collect(),
                );

                let tool_choice = Some((&tool_config.tool_choice).into());

                Ok((tools, tool_choice, parallel_tool_calls))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{AllowedTools, AllowedToolsChoice, ToolCallConfig};
    use serde_json::json;

    // Helper to create a test tool config
    fn create_static_tool_config() -> FunctionToolConfig {
        use crate::jsonschema_util::StaticJSONSchema;
        use crate::tool::StaticToolConfig;
        use std::sync::Arc;

        FunctionToolConfig::Static(Arc::new(StaticToolConfig {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            parameters: StaticJSONSchema::from_value(json!({
                "type": "object",
                "properties": {
                    "param1": {"type": "string"}
                }
            }))
            .unwrap(),
            strict: true,
        }))
    }

    fn create_dynamic_tool_config() -> FunctionToolConfig {
        use crate::jsonschema_util::DynamicJSONSchema;
        use crate::tool::DynamicToolConfig;

        FunctionToolConfig::Dynamic(DynamicToolConfig {
            name: "dynamic_tool".to_string(),
            description: "A dynamic tool".to_string(),
            parameters: DynamicJSONSchema::new(json!({
                "type": "object",
                "properties": {
                    "param1": {"type": "string"}
                }
            })),
            strict: true,
        })
    }

    // Test serialization of ChatCompletionToolType
    #[test]
    fn test_chat_completion_tool_type_serialization() {
        let tool_type = ChatCompletionToolType::Function;
        let serialized = serde_json::to_string(&tool_type).unwrap();
        assert_eq!(serialized, "\"function\"");
    }

    // Test serialization of ChatCompletionFunction
    #[test]
    fn test_chat_completion_function_with_description() {
        let params = json!({"type": "object"});
        let function = ChatCompletionFunction {
            name: "test_func",
            description: Some("A test function"),
            parameters: &params,
        };
        let serialized = serde_json::to_value(&function).unwrap();
        assert_eq!(serialized["name"], "test_func");
        assert_eq!(serialized["description"], "A test function");
        assert_eq!(serialized["parameters"]["type"], "object");
    }

    #[test]
    fn test_chat_completion_function_without_description() {
        let params = json!({"type": "object"});
        let function = ChatCompletionFunction {
            name: "test_func",
            description: None,
            parameters: &params,
        };
        let serialized = serde_json::to_value(&function).unwrap();
        assert_eq!(serialized["name"], "test_func");
        assert!(serialized.get("description").is_none());
    }

    // Test serialization of ChatCompletionTool
    #[test]
    fn test_chat_completion_tool_strict_true() {
        let params = json!({"type": "object"});
        let tool = ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: ChatCompletionFunction {
                name: "test",
                description: Some("test"),
                parameters: &params,
            },
            strict: true,
        };
        let serialized = serde_json::to_value(&tool).unwrap();
        assert_eq!(serialized["strict"], true);
        assert_eq!(serialized["type"], "function");
    }

    #[test]
    fn test_chat_completion_tool_strict_false() {
        let params = json!({"type": "object"});
        let tool = ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: ChatCompletionFunction {
                name: "test",
                description: Some("test"),
                parameters: &params,
            },
            strict: false,
        };
        let serialized = serde_json::to_value(&tool).unwrap();
        assert_eq!(serialized["strict"], false);
    }

    // Test ChatCompletionToolChoice serialization
    #[test]
    fn test_tool_choice_string_none() {
        let choice = ChatCompletionToolChoice::String(ChatCompletionToolChoiceString::None);
        let serialized = serde_json::to_string(&choice).unwrap();
        assert_eq!(serialized, "\"none\"");
    }

    #[test]
    fn test_tool_choice_string_auto() {
        let choice = ChatCompletionToolChoice::String(ChatCompletionToolChoiceString::Auto);
        let serialized = serde_json::to_string(&choice).unwrap();
        assert_eq!(serialized, "\"auto\"");
    }

    #[test]
    fn test_tool_choice_string_required() {
        let choice = ChatCompletionToolChoice::String(ChatCompletionToolChoiceString::Required);
        let serialized = serde_json::to_string(&choice).unwrap();
        assert_eq!(serialized, "\"required\"");
    }

    #[test]
    fn test_tool_choice_specific() {
        let choice = ChatCompletionToolChoice::Specific(ChatCompletionSpecificToolChoice {
            r#type: ChatCompletionToolType::Function,
            function: ChatCompletionSpecificToolFunction { name: "my_tool" },
        });
        let serialized = serde_json::to_value(&choice).unwrap();
        assert_eq!(serialized["type"], "function");
        assert_eq!(serialized["function"]["name"], "my_tool");
    }

    #[test]
    fn test_tool_choice_allowed_tools() {
        let choice = ChatCompletionToolChoice::AllowedTools(ChatCompletionAllowedToolsChoice {
            r#type: "allowed_tools",
            allowed_tools: ChatCompletionAllowedToolsConstraint {
                mode: ChatCompletionAllowedToolsMode::Auto,
                tools: vec![ChatCompletionToolReference {
                    r#type: ChatCompletionToolType::Function,
                    function: ChatCompletionSpecificToolFunction { name: "tool1" },
                }],
            },
        });
        let serialized = serde_json::to_value(&choice).unwrap();
        assert_eq!(serialized["type"], "allowed_tools");
        assert_eq!(serialized["allowed_tools"]["mode"], "auto");
        assert_eq!(serialized["allowed_tools"]["tools"][0]["type"], "function");
        assert_eq!(
            serialized["allowed_tools"]["tools"][0]["function"]["name"],
            "tool1"
        );
    }

    // Test From implementations
    #[test]
    fn test_from_function_tool_config_static() {
        let tool_config = create_static_tool_config();
        let chat_tool: ChatCompletionTool = (&tool_config).into();

        assert_eq!(chat_tool.function.name, "test_tool");
        assert_eq!(chat_tool.function.description, Some("A test tool"));
        assert!(chat_tool.strict);
        assert!(matches!(chat_tool.r#type, ChatCompletionToolType::Function));
    }

    #[tokio::test]
    async fn test_from_function_tool_config_dynamic() {
        let tool_config = create_dynamic_tool_config();
        let chat_tool: ChatCompletionTool = (&tool_config).into();

        assert_eq!(chat_tool.function.name, "dynamic_tool");
        assert!(matches!(chat_tool.r#type, ChatCompletionToolType::Function));
    }

    #[test]
    fn test_from_function_tool() {
        let tool = FunctionTool {
            name: "direct_tool".to_string(),
            description: "Direct tool".to_string(),
            parameters: json!({"type": "object"}),
            strict: false,
        };
        let chat_tool: ChatCompletionTool = (&tool).into();

        assert_eq!(chat_tool.function.name, "direct_tool");
        assert_eq!(chat_tool.function.description, Some("Direct tool"));
        assert!(!chat_tool.strict);
    }

    #[test]
    fn test_from_tool_choice_none() {
        let choice = ToolChoice::None;
        let chat_choice: ChatCompletionToolChoice = (&choice).into();

        match chat_choice {
            ChatCompletionToolChoice::String(ChatCompletionToolChoiceString::None) => (),
            _ => panic!("Expected None variant"),
        }
    }

    #[test]
    fn test_from_tool_choice_auto() {
        let choice = ToolChoice::Auto;
        let chat_choice: ChatCompletionToolChoice = (&choice).into();

        match chat_choice {
            ChatCompletionToolChoice::String(ChatCompletionToolChoiceString::Auto) => (),
            _ => panic!("Expected Auto variant"),
        }
    }

    #[test]
    fn test_from_tool_choice_required() {
        let choice = ToolChoice::Required;
        let chat_choice: ChatCompletionToolChoice = (&choice).into();

        match chat_choice {
            ChatCompletionToolChoice::String(ChatCompletionToolChoiceString::Required) => (),
            _ => panic!("Expected Required variant"),
        }
    }

    #[test]
    fn test_from_tool_choice_specific() {
        let choice = ToolChoice::Specific("my_specific_tool".to_string());
        let chat_choice: ChatCompletionToolChoice = (&choice).into();

        match chat_choice {
            ChatCompletionToolChoice::Specific(specific) => {
                assert_eq!(specific.function.name, "my_specific_tool");
            }
            _ => panic!("Expected Specific variant"),
        }
    }

    // Test prepare_chat_completion_allowed_tools_constraint
    #[test]
    fn test_prepare_allowed_tools_constraint_none_without_dynamic() {
        let tool_config = ToolCallConfig {
            tool_choice: ToolChoice::Auto,
            allowed_tools: AllowedTools {
                tools: vec![],
                choice: AllowedToolsChoice::FunctionDefault,
            },
            parallel_tool_calls: None,
            static_tools_available: vec![create_static_tool_config()],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
        };

        let result = prepare_chat_completion_allowed_tools_constraint(&tool_config);
        assert!(result.is_none());
    }

    #[test]
    fn test_prepare_allowed_tools_constraint_with_required() {
        let tool_config = ToolCallConfig {
            tool_choice: ToolChoice::Required,
            allowed_tools: AllowedTools {
                tools: vec!["tool1".to_string(), "tool2".to_string()]
                    .into_iter()
                    .collect(),
                choice: AllowedToolsChoice::Explicit,
            },
            parallel_tool_calls: None,
            static_tools_available: vec![create_static_tool_config()],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
        };

        let result = prepare_chat_completion_allowed_tools_constraint(&tool_config).unwrap();
        assert_eq!(result.r#type, "allowed_tools");
        assert!(matches!(
            result.allowed_tools.mode,
            ChatCompletionAllowedToolsMode::Required
        ));
        assert_eq!(result.allowed_tools.tools.len(), 2);
        let tool_names: Vec<&str> = result
            .allowed_tools
            .tools
            .iter()
            .map(|t| t.function.name)
            .collect();
        assert!(tool_names.contains(&"tool1"));
        assert!(tool_names.contains(&"tool2"));
    }

    #[test]
    fn test_prepare_allowed_tools_constraint_choice_none() {
        let tool_config = ToolCallConfig {
            tool_choice: ToolChoice::None,
            allowed_tools: AllowedTools {
                tools: vec!["tool1".to_string(), "tool2".to_string()]
                    .into_iter()
                    .collect(),
                choice: AllowedToolsChoice::Explicit,
            },
            parallel_tool_calls: None,
            static_tools_available: vec![create_static_tool_config()],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
        };

        let result = prepare_chat_completion_allowed_tools_constraint(&tool_config);
        assert!(result.is_none());
    }

    #[test]
    fn test_prepare_allowed_tools_constraint_with_auto() {
        let tool_config = ToolCallConfig {
            tool_choice: ToolChoice::Auto,
            allowed_tools: AllowedTools {
                tools: vec!["tool1".to_string()].into_iter().collect(),
                choice: AllowedToolsChoice::Explicit,
            },
            parallel_tool_calls: None,
            static_tools_available: vec![create_static_tool_config()],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
        };

        let result = prepare_chat_completion_allowed_tools_constraint(&tool_config).unwrap();
        assert!(matches!(
            result.allowed_tools.mode,
            ChatCompletionAllowedToolsMode::Auto
        ));
        assert_eq!(result.allowed_tools.tools.len(), 1);
    }

    #[test]
    fn test_prepare_allowed_tools_constraint_with_specific_uses_auto_mode() {
        let tool_config = ToolCallConfig {
            tool_choice: ToolChoice::Specific("tool1".to_string()),
            allowed_tools: AllowedTools {
                tools: vec!["tool1".to_string()].into_iter().collect(),
                choice: AllowedToolsChoice::Explicit,
            },
            parallel_tool_calls: None,
            static_tools_available: vec![],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
        };

        let result = prepare_chat_completion_allowed_tools_constraint(&tool_config).unwrap();
        assert!(matches!(
            result.allowed_tools.mode,
            ChatCompletionAllowedToolsMode::Auto
        ));
    }

    // Test prepare_chat_completion_tools
    #[test]
    fn test_prepare_tools_with_none_tool_config() {
        let request = ModelInferenceRequest {
            tool_config: None,
            ..Default::default()
        };

        let (tools, tool_choice, parallel) = prepare_chat_completion_tools(&request, true).unwrap();
        assert!(tools.is_none());
        assert!(tool_choice.is_none());
        assert!(parallel.is_none());
    }

    #[test]
    fn test_prepare_tools_with_empty_tools() {
        let tool_config = ToolCallConfig {
            tool_choice: ToolChoice::Auto,
            allowed_tools: AllowedTools {
                tools: Vec::new(),
                choice: AllowedToolsChoice::FunctionDefault,
            },
            parallel_tool_calls: None,
            static_tools_available: vec![],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
        };

        let request = ModelInferenceRequest {
            tool_config: Some(std::borrow::Cow::Owned(tool_config)),
            ..Default::default()
        };

        let (tools, tool_choice, parallel) = prepare_chat_completion_tools(&request, true).unwrap();
        assert!(tools.is_none());
        assert!(tool_choice.is_none());
        assert!(parallel.is_none());
    }

    #[test]
    fn test_prepare_tools_supports_allowed_tools_without_constraint() {
        let tool_config = ToolCallConfig {
            tool_choice: ToolChoice::Auto,
            allowed_tools: AllowedTools {
                tools: Vec::new(),
                choice: AllowedToolsChoice::FunctionDefault,
            },
            parallel_tool_calls: Some(true),
            static_tools_available: vec![create_static_tool_config()],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
        };

        let request = ModelInferenceRequest {
            tool_config: Some(std::borrow::Cow::Owned(tool_config)),
            ..Default::default()
        };

        let (tools, tool_choice, parallel) = prepare_chat_completion_tools(&request, true).unwrap();
        assert!(tools.is_some());
        assert_eq!(tools.unwrap().len(), 1);
        assert!(tool_choice.is_some());
        assert_eq!(parallel, Some(true));

        // Should use regular tool_choice, not AllowedTools
        match tool_choice.unwrap() {
            ChatCompletionToolChoice::String(ChatCompletionToolChoiceString::Auto) => (),
            _ => panic!("Expected Auto string variant"),
        }
    }

    #[test]
    fn test_prepare_tools_supports_allowed_tools_with_constraint() {
        let tool_config = ToolCallConfig {
            tool_choice: ToolChoice::Required,
            allowed_tools: AllowedTools {
                tools: vec!["test_tool".to_string()].into_iter().collect(),
                choice: AllowedToolsChoice::Explicit,
            },
            parallel_tool_calls: Some(false),
            static_tools_available: vec![create_static_tool_config()],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
        };

        let request = ModelInferenceRequest {
            tool_config: Some(std::borrow::Cow::Owned(tool_config)),
            ..Default::default()
        };

        let (tools, tool_choice, parallel) = prepare_chat_completion_tools(&request, true).unwrap();
        assert!(tools.is_some());
        assert!(tool_choice.is_some());
        assert_eq!(parallel, Some(false));

        // Should use AllowedTools constraint
        match tool_choice.unwrap() {
            ChatCompletionToolChoice::AllowedTools(allowed) => {
                assert_eq!(allowed.r#type, "allowed_tools");
                assert!(matches!(
                    allowed.allowed_tools.mode,
                    ChatCompletionAllowedToolsMode::Required
                ));
            }
            _ => panic!("Expected AllowedTools variant"),
        }
    }

    #[test]
    fn test_prepare_tools_no_allowed_tools_support() {
        let tool_config = ToolCallConfig {
            tool_choice: ToolChoice::Auto,
            allowed_tools: AllowedTools {
                tools: Vec::new(),
                choice: AllowedToolsChoice::FunctionDefault,
            },
            parallel_tool_calls: None,
            static_tools_available: vec![create_static_tool_config()],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
        };

        let request = ModelInferenceRequest {
            tool_config: Some(std::borrow::Cow::Owned(tool_config)),
            ..Default::default()
        };

        let (tools, tool_choice, parallel) =
            prepare_chat_completion_tools(&request, false).unwrap();
        assert!(tools.is_some());
        assert!(tool_choice.is_some());
        assert!(parallel.is_none());

        // Should use strict_tools_available filtering
        match tool_choice.unwrap() {
            ChatCompletionToolChoice::String(ChatCompletionToolChoiceString::Auto) => (),
            _ => panic!("Expected Auto string variant"),
        }
    }

    #[test]
    fn test_prepare_tools_no_allowed_tools_support_with_explicit_allowed() {
        let tool_config = ToolCallConfig {
            tool_choice: ToolChoice::Required,
            allowed_tools: AllowedTools {
                tools: vec!["test_tool".to_string()].into_iter().collect(),
                choice: AllowedToolsChoice::Explicit,
            },
            parallel_tool_calls: Some(true),
            static_tools_available: vec![create_static_tool_config()],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
        };

        let request = ModelInferenceRequest {
            tool_config: Some(std::borrow::Cow::Owned(tool_config)),
            ..Default::default()
        };

        let (tools, tool_choice, parallel) =
            prepare_chat_completion_tools(&request, false).unwrap();
        assert!(tools.is_some());
        assert!(tool_choice.is_some());
        assert_eq!(parallel, Some(true));

        // Should NOT use AllowedTools constraint, use regular tool_choice
        match tool_choice.unwrap() {
            ChatCompletionToolChoice::String(ChatCompletionToolChoiceString::Required) => (),
            _ => panic!("Expected Required string variant"),
        }
    }

    #[test]
    fn test_prepare_tools_multiple_tools() {
        use crate::jsonschema_util::StaticJSONSchema;
        use crate::tool::StaticToolConfig;
        use std::sync::Arc;

        let tool1 = FunctionToolConfig::Static(Arc::new(StaticToolConfig {
            name: "tool1".to_string(),
            description: "First tool".to_string(),
            parameters: StaticJSONSchema::from_value(json!({"type": "object"})).unwrap(),
            strict: true,
        }));

        let tool2 = FunctionToolConfig::Static(Arc::new(StaticToolConfig {
            name: "tool2".to_string(),
            description: "Second tool".to_string(),
            parameters: StaticJSONSchema::from_value(json!({"type": "object"})).unwrap(),
            strict: false,
        }));

        let tool_config = ToolCallConfig {
            tool_choice: ToolChoice::Auto,
            allowed_tools: AllowedTools {
                tools: Vec::new(),
                choice: AllowedToolsChoice::FunctionDefault,
            },
            parallel_tool_calls: None,
            static_tools_available: vec![tool1, tool2],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
        };

        let request = ModelInferenceRequest {
            tool_config: Some(std::borrow::Cow::Owned(tool_config)),
            ..Default::default()
        };

        let (tools, tool_choice, _) = prepare_chat_completion_tools(&request, true).unwrap();
        let tools = tools.unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].function.name, "tool1");
        assert!(tools[0].strict);
        assert_eq!(tools[1].function.name, "tool2");
        assert!(!tools[1].strict);
        assert!(tool_choice.is_some());
    }

    #[test]
    fn test_prepare_tools_with_specific_tool_choice() {
        let tool_config = ToolCallConfig {
            tool_choice: ToolChoice::Specific("test_tool".to_string()),
            allowed_tools: AllowedTools {
                tools: Vec::new(),
                choice: AllowedToolsChoice::FunctionDefault,
            },
            parallel_tool_calls: None,
            static_tools_available: vec![create_static_tool_config()],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
        };

        let request = ModelInferenceRequest {
            tool_config: Some(std::borrow::Cow::Owned(tool_config)),
            ..Default::default()
        };

        let (tools, tool_choice, _) = prepare_chat_completion_tools(&request, true).unwrap();
        assert!(tools.is_some());
        assert!(tool_choice.is_some());

        match tool_choice.unwrap() {
            ChatCompletionToolChoice::Specific(specific) => {
                assert_eq!(specific.function.name, "test_tool");
            }
            _ => panic!("Expected Specific variant"),
        }
    }
}
