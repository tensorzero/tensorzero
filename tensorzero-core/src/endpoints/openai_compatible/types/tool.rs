//! Tool types and conversions for OpenAI-compatible API.
//!
//! This module provides types for tool calling functionality in the OpenAI-compatible API,
//! including tool definitions, tool calls, tool choice options, and conversion logic
//! between OpenAI's tool format and TensorZero's internal tool representations.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tool::{
    FunctionTool, InferenceResponseToolCall, OpenAICustomTool, Tool, ToolCallWrapper, ToolChoice,
};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OpenAICompatibleFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OpenAICompatibleToolCallDelta {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OpenAICompatibleToolCall {
    /// The ID of the tool call.
    pub id: String,
    /// The type of the tool. Currently, only `function` is supported.
    pub r#type: String,
    /// The function that the model called.
    pub function: OpenAICompatibleFunctionCall,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OpenAICompatibleToolCallChunk {
    /// The ID of the tool call.
    pub id: Option<String>,
    /// The index of the tool call.
    pub index: usize,
    /// The type of the tool. Currently, only `function` is supported.
    pub r#type: String,
    /// The function that the model called.
    pub function: OpenAICompatibleToolCallDelta,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct OpenAICompatibleToolMessage {
    pub content: Option<Value>,
    pub tool_call_id: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAICompatibleTool {
    Function {
        function: OpenAICompatibleFunctionTool,
    },
    Custom {
        custom: OpenAICustomTool,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct OpenAICompatibleFunctionTool {
    description: Option<String>,
    name: String,
    parameters: Value,
    #[serde(default)]
    strict: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct FunctionName {
    pub name: String,
}

/// Specifies a tool the model should use. Use to force the model to call a specific function.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct OpenAICompatibleNamedToolChoice {
    /// The type of the tool. Currently, only `function` is supported.
    pub r#type: String,
    pub function: FunctionName,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum OpenAICompatibleAllowedToolsMode {
    Auto,
    Required,
}

impl From<OpenAICompatibleAllowedToolsMode> for ToolChoice {
    fn from(mode: OpenAICompatibleAllowedToolsMode) -> Self {
        match mode {
            OpenAICompatibleAllowedToolsMode::Auto => ToolChoice::Auto,
            OpenAICompatibleAllowedToolsMode::Required => ToolChoice::Required,
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct OpenAICompatibleAllowedTools {
    pub tools: Vec<OpenAICompatibleNamedToolChoice>,
    pub mode: OpenAICompatibleAllowedToolsMode,
}

/// Controls which (if any) tool is called by the model.
/// `none` means the model will not call any tool and instead generates a message.
/// `auto` means the model can pick between generating a message or calling one or more tools.
/// `required` means the model must call one or more tools.
/// Specifying a particular tool via `{"type": "function", "function": {"name": "my_function"}}` forces the model to call that tool.
///
/// `none` is the default when no tools are present. `auto` is the default if tools are present.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum ChatCompletionToolChoiceOption {
    #[default]
    None,
    Auto,
    Required,
    AllowedTools(OpenAICompatibleAllowedTools),
    Named(OpenAICompatibleNamedToolChoice),
}

impl<'de> Deserialize<'de> for ChatCompletionToolChoiceOption {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;
        use serde_json::Value;

        let value = Value::deserialize(deserializer)?;

        match &value {
            Value::String(s) => match s.as_str() {
                "none" => Ok(ChatCompletionToolChoiceOption::None),
                "auto" => Ok(ChatCompletionToolChoiceOption::Auto),
                "required" => Ok(ChatCompletionToolChoiceOption::Required),
                _ => Err(D::Error::custom(format!("Invalid tool choice string: {s}"))),
            },
            Value::Object(obj) => {
                if let Some(type_value) = obj.get("type") {
                    if let Some(type_str) = type_value.as_str() {
                        match type_str {
                            "function" => {
                                // This is a named tool choice
                                let named: OpenAICompatibleNamedToolChoice =
                                    serde_json::from_value(value).map_err(D::Error::custom)?;
                                Ok(ChatCompletionToolChoiceOption::Named(named))
                            }
                            "allowed_tools" => {
                                // This is an allowed tools choice - extract the allowed_tools field
                                if let Some(allowed_tools_value) = obj.get("allowed_tools") {
                                    let allowed_tools: OpenAICompatibleAllowedTools =
                                        serde_json::from_value(allowed_tools_value.clone())
                                            .map_err(D::Error::custom)?;
                                    Ok(ChatCompletionToolChoiceOption::AllowedTools(allowed_tools))
                                } else {
                                    Err(D::Error::custom(
                                        "Missing 'allowed_tools' field in allowed_tools type",
                                    ))
                                }
                            }
                            _ => Err(D::Error::custom(format!(
                                "Invalid tool choice type: {type_str}",
                            ))),
                        }
                    } else {
                        Err(D::Error::custom(
                            "Tool choice 'type' field must be a string",
                        ))
                    }
                } else {
                    Err(D::Error::custom(
                        "Tool choice field must have a 'type' field if it is an object",
                    ))
                }
            }
            _ => Err(D::Error::custom("Tool choice must be a string or object")),
        }
    }
}

impl ChatCompletionToolChoiceOption {
    pub(crate) fn into_tool_params(self) -> OpenAICompatibleToolChoiceParams {
        match self {
            ChatCompletionToolChoiceOption::None => OpenAICompatibleToolChoiceParams {
                allowed_tools: None,
                tool_choice: Some(ToolChoice::None),
            },
            ChatCompletionToolChoiceOption::Auto => OpenAICompatibleToolChoiceParams {
                allowed_tools: None,
                tool_choice: Some(ToolChoice::Auto),
            },
            ChatCompletionToolChoiceOption::Required => OpenAICompatibleToolChoiceParams {
                allowed_tools: None,
                tool_choice: Some(ToolChoice::Required),
            },
            ChatCompletionToolChoiceOption::AllowedTools(allowed_tool_info) => {
                OpenAICompatibleToolChoiceParams {
                    allowed_tools: Some(
                        allowed_tool_info
                            .tools
                            .into_iter()
                            .map(|tool| tool.function.name)
                            .collect(),
                    ),
                    tool_choice: Some(allowed_tool_info.mode.into()),
                }
            }
            ChatCompletionToolChoiceOption::Named(named_tool) => OpenAICompatibleToolChoiceParams {
                allowed_tools: None,
                tool_choice: Some(ToolChoice::Specific(named_tool.function.name)),
            },
        }
    }
}

#[derive(Default)]
pub(crate) struct OpenAICompatibleToolChoiceParams {
    pub allowed_tools: Option<Vec<String>>,
    pub tool_choice: Option<ToolChoice>,
}

impl From<OpenAICompatibleTool> for Tool {
    fn from(tool: OpenAICompatibleTool) -> Self {
        match tool {
            OpenAICompatibleTool::Function { function } => {
                Tool::Function(FunctionTool::from(function))
            }
            OpenAICompatibleTool::Custom { custom } => Tool::OpenAICustom(custom),
        }
    }
}

impl From<OpenAICompatibleFunctionTool> for FunctionTool {
    fn from(tool: OpenAICompatibleFunctionTool) -> Self {
        FunctionTool {
            description: tool.description.unwrap_or_default(),
            parameters: tool.parameters,
            name: tool.name,
            strict: tool.strict,
        }
    }
}

impl From<OpenAICompatibleToolCall> for ToolCallWrapper {
    fn from(tool_call: OpenAICompatibleToolCall) -> Self {
        ToolCallWrapper::InferenceResponseToolCall(InferenceResponseToolCall {
            id: tool_call.id,
            raw_name: tool_call.function.name,
            raw_arguments: tool_call.function.arguments,
            name: None,
            arguments: None,
        })
    }
}

impl From<InferenceResponseToolCall> for OpenAICompatibleToolCall {
    fn from(tool_call: InferenceResponseToolCall) -> Self {
        OpenAICompatibleToolCall {
            id: tool_call.id,
            r#type: "function".to_string(),
            function: OpenAICompatibleFunctionCall {
                name: tool_call.raw_name,
                arguments: tool_call.raw_arguments,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_chat_completion_tool_choice_option_deserialization_and_conversion() {
        // Test deserialization from JSON and conversion to OpenAICompatibleToolChoiceParams

        // Test None variant
        let json_none = json!("none");
        let tool_choice: ChatCompletionToolChoiceOption =
            serde_json::from_value(json_none).unwrap();
        assert_eq!(tool_choice, ChatCompletionToolChoiceOption::None);
        let params = tool_choice.into_tool_params();
        assert_eq!(params.allowed_tools, None);
        assert_eq!(params.tool_choice, Some(ToolChoice::None));

        // Test Auto variant
        let json_auto = json!("auto");
        let tool_choice: ChatCompletionToolChoiceOption =
            serde_json::from_value(json_auto).unwrap();
        assert_eq!(tool_choice, ChatCompletionToolChoiceOption::Auto);
        let params = tool_choice.into_tool_params();
        assert_eq!(params.allowed_tools, None);
        assert_eq!(params.tool_choice, Some(ToolChoice::Auto));

        // Test Required variant
        let json_required = json!("required");
        let tool_choice: ChatCompletionToolChoiceOption =
            serde_json::from_value(json_required).unwrap();
        assert_eq!(tool_choice, ChatCompletionToolChoiceOption::Required);
        let params = tool_choice.into_tool_params();
        assert_eq!(params.allowed_tools, None);
        assert_eq!(params.tool_choice, Some(ToolChoice::Required));

        // Test Named variant (specific tool)
        let json_named = json!({
            "type": "function",
            "function": {
                "name": "get_weather"
            }
        });
        let tool_choice: ChatCompletionToolChoiceOption =
            serde_json::from_value(json_named).unwrap();
        assert_eq!(
            tool_choice,
            ChatCompletionToolChoiceOption::Named(OpenAICompatibleNamedToolChoice {
                r#type: "function".to_string(),
                function: FunctionName {
                    name: "get_weather".to_string()
                }
            })
        );
        let params = tool_choice.into_tool_params();
        assert_eq!(params.allowed_tools, None);
        assert_eq!(
            params.tool_choice,
            Some(ToolChoice::Specific("get_weather".to_string()))
        );

        // Test AllowedTools variant with auto mode
        let json_allowed_auto = json!({
            "type": "allowed_tools",
            "allowed_tools": {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather"
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "send_email"
                    }
                }
            ],
            "mode": "auto"
        }});
        let tool_choice: ChatCompletionToolChoiceOption =
            serde_json::from_value(json_allowed_auto).unwrap();
        assert_eq!(
            tool_choice,
            ChatCompletionToolChoiceOption::AllowedTools(OpenAICompatibleAllowedTools {
                tools: vec![
                    OpenAICompatibleNamedToolChoice {
                        r#type: "function".to_string(),
                        function: FunctionName {
                            name: "get_weather".to_string()
                        }
                    },
                    OpenAICompatibleNamedToolChoice {
                        r#type: "function".to_string(),
                        function: FunctionName {
                            name: "send_email".to_string()
                        }
                    }
                ],
                mode: OpenAICompatibleAllowedToolsMode::Auto
            })
        );
        let params = tool_choice.into_tool_params();
        assert_eq!(
            params.allowed_tools,
            Some(vec!["get_weather".to_string(), "send_email".to_string()])
        );
        assert_eq!(params.tool_choice, Some(ToolChoice::Auto));

        // Test AllowedTools variant with required mode
        let json_allowed_required = json!({
            "type": "allowed_tools",
            "allowed_tools": {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather"
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "send_email"
                    }
                }
            ],
            "mode": "required"
        }});
        let tool_choice: ChatCompletionToolChoiceOption =
            serde_json::from_value(json_allowed_required).unwrap();
        assert_eq!(
            tool_choice,
            ChatCompletionToolChoiceOption::AllowedTools(OpenAICompatibleAllowedTools {
                tools: vec![
                    OpenAICompatibleNamedToolChoice {
                        r#type: "function".to_string(),
                        function: FunctionName {
                            name: "get_weather".to_string()
                        }
                    },
                    OpenAICompatibleNamedToolChoice {
                        r#type: "function".to_string(),
                        function: FunctionName {
                            name: "send_email".to_string()
                        }
                    }
                ],
                mode: OpenAICompatibleAllowedToolsMode::Required
            })
        );
        let params = tool_choice.into_tool_params();
        assert_eq!(
            params.allowed_tools,
            Some(vec!["get_weather".to_string(), "send_email".to_string()])
        );
        assert_eq!(params.tool_choice, Some(ToolChoice::Required));

        // Test default value (should be None)
        let tool_choice_default = ChatCompletionToolChoiceOption::default();
        assert_eq!(tool_choice_default, ChatCompletionToolChoiceOption::None);
        let params_default = tool_choice_default.into_tool_params();
        assert_eq!(params_default.allowed_tools, None);
        assert_eq!(params_default.tool_choice, Some(ToolChoice::None));
    }

    #[test]
    fn test_chat_completion_tool_choice_option_invalid_deserialization() {
        // Test invalid JSON values that should fail to deserialize

        // Invalid string value
        let json_invalid = json!("invalid_choice");
        let result: Result<ChatCompletionToolChoiceOption, _> =
            serde_json::from_value(json_invalid);
        assert!(result.is_err());

        // Invalid object structure for named tool choice
        let json_invalid_named = json!({
            "type": "invalid_type",
            "function": {
                "name": "test"
            }
        });
        let result: Result<ChatCompletionToolChoiceOption, _> =
            serde_json::from_value(json_invalid_named);
        assert!(result.is_err());

        // Missing function name in named tool choice
        let json_missing_name = json!({
            "type": "function",
            "function": {}
        });
        let result: Result<ChatCompletionToolChoiceOption, _> =
            serde_json::from_value(json_missing_name);
        assert!(result.is_err());

        // Invalid mode in allowed tools
        let json_invalid_mode = json!({
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "test"
                    }
                }
            ],
            "mode": "invalid_mode"
        });
        let result: Result<ChatCompletionToolChoiceOption, _> =
            serde_json::from_value(json_invalid_mode);
        assert!(result.is_err());

        // Test AllowedTools variant with no type
        let json_allowed_required = json!({
            "allowed_tools": {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather"
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "send_email"
                    }
                }
            ],
            "mode": "required"
        }});
        let err = serde_json::from_value::<ChatCompletionToolChoiceOption>(json_allowed_required)
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "Tool choice field must have a 'type' field if it is an object"
        );
    }

    #[test]
    fn test_openai_compatible_allowed_tools_mode_conversion() {
        // Test conversion from OpenAICompatibleAllowedToolsMode to ToolChoice
        let auto_mode = OpenAICompatibleAllowedToolsMode::Auto;
        let tool_choice: ToolChoice = auto_mode.into();
        assert_eq!(tool_choice, ToolChoice::Auto);

        let required_mode = OpenAICompatibleAllowedToolsMode::Required;
        let tool_choice: ToolChoice = required_mode.into();
        assert_eq!(tool_choice, ToolChoice::Required);
    }

    #[test]
    fn test_openai_compatible_tool_deserialization() {
        // Test Function variant
        let function_json = json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                },
                "strict": true
            }
        });

        let tool: OpenAICompatibleTool = serde_json::from_value(function_json).unwrap();
        match tool {
            OpenAICompatibleTool::Function { function } => {
                assert_eq!(function.name, "get_weather");
                assert_eq!(
                    function.description,
                    Some("Get the current weather".to_string())
                );
                assert!(function.strict);
            }
            OpenAICompatibleTool::Custom { .. } => panic!("Expected Function variant"),
        }

        // Test Custom variant with text format
        let custom_text_json = json!({
            "type": "custom",
            "custom": {
                "name": "custom_tool",
                "description": "A custom tool",
                "format": {
                    "type": "text"
                }
            }
        });

        let tool: OpenAICompatibleTool = serde_json::from_value(custom_text_json).unwrap();
        match tool {
            OpenAICompatibleTool::Custom { custom } => {
                assert_eq!(custom.name, "custom_tool");
                assert_eq!(custom.description, Some("A custom tool".to_string()));
                assert!(custom.format.is_some());
            }
            OpenAICompatibleTool::Function { .. } => panic!("Expected Custom variant"),
        }

        // Test Custom variant with grammar format
        let custom_grammar_json = json!({
            "type": "custom",
            "custom": {
                "name": "grammar_tool",
                "format": {
                    "type": "grammar",
                    "grammar": {
                        "syntax": "lark",
                        "definition": "start: /[a-z]+/"
                    }
                }
            }
        });

        let tool: OpenAICompatibleTool = serde_json::from_value(custom_grammar_json).unwrap();
        match tool {
            OpenAICompatibleTool::Custom { custom } => {
                assert_eq!(custom.name, "grammar_tool");
                assert!(custom.format.is_some());
            }
            OpenAICompatibleTool::Function { .. } => panic!("Expected Custom variant"),
        }

        // Test conversion to Tool
        let function_json = json!({
            "type": "function",
            "function": {
                "name": "test_function",
                "parameters": {}
            }
        });

        let openai_tool: OpenAICompatibleTool = serde_json::from_value(function_json).unwrap();
        let tool: Tool = openai_tool.into();
        match tool {
            Tool::Function(func) => {
                assert_eq!(func.name, "test_function");
            }
            Tool::OpenAICustom(_) => panic!("Expected Function variant"),
        }

        // Test conversion of custom tool to Tool
        let custom_json = json!({
            "type": "custom",
            "custom": {
                "name": "custom_test"
            }
        });

        let openai_tool: OpenAICompatibleTool = serde_json::from_value(custom_json).unwrap();
        let tool: Tool = openai_tool.into();
        match tool {
            Tool::OpenAICustom(custom) => {
                assert_eq!(custom.name, "custom_test");
            }
            Tool::Function(_) => panic!("Expected OpenAICustom variant"),
        }
    }
}
