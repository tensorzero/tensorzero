//! Tool types for Anthropic-compatible API.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::tool::{ProviderTool, Tool};
use tensorzero_types::ToolChoice;

/// Tool definition for Anthropic-compatible requests
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct AnthropicTool {
    pub name: String,
    pub description: String,
    pub input_schema: AnthropicInputSchema,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicToolChoice {
    Auto,
    Any,
    Tool { name: String },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
pub struct AnthropicInputSchema {
    #[serde(rename = "type")]
    pub schema_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub additional_properties: Option<bool>,
}

/// Parameters extracted from tool choice
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AnthropicToolChoiceParams {
    pub allowed_tools: Option<Vec<String>>,
    pub tool_choice: Option<ToolChoice>,
}

impl AnthropicToolChoice {
    pub fn into_tool_params(self) -> AnthropicToolChoiceParams {
        match self {
            AnthropicToolChoice::Auto => AnthropicToolChoiceParams {
                allowed_tools: None,
                tool_choice: Some(ToolChoice::Auto),
            },
            AnthropicToolChoice::Any => AnthropicToolChoiceParams {
                allowed_tools: None,
                tool_choice: Some(ToolChoice::Required),
            },
            AnthropicToolChoice::Tool { name } => AnthropicToolChoiceParams {
                allowed_tools: Some(vec![name.clone()]),
                tool_choice: Some(ToolChoice::Specific(name)),
            },
        }
    }
}

impl From<AnthropicTool> for Tool {
    fn from(tool: AnthropicTool) -> Self {
        Tool::Function(crate::tool::FunctionTool {
            name: tool.name,
            description: tool.description,
            parameters: serde_json::json!({
                "type": tool.input_schema.schema_type,
                "properties": tool.input_schema.properties.unwrap_or_default(),
                "required": tool.input_schema.required.unwrap_or_default(),
                "additionalProperties": tool.input_schema.additional_properties,
            }),
            strict: false,
        })
    }
}

impl From<AnthropicTool> for ProviderTool {
    fn from(tool: AnthropicTool) -> Self {
        ProviderTool {
            scope: Default::default(),
            tool: serde_json::to_value(tool).unwrap_or_default(),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_tool_choice_auto_conversion() {
        let tool_choice = AnthropicToolChoice::Auto;
        let params = tool_choice.into_tool_params();

        assert!(params.allowed_tools.is_none());
        assert_eq!(params.tool_choice, Some(tensorzero_types::ToolChoice::Auto));
    }

    #[test]
    fn test_tool_choice_any_conversion() {
        let tool_choice = AnthropicToolChoice::Any;
        let params = tool_choice.into_tool_params();

        assert!(params.allowed_tools.is_none());
        assert_eq!(
            params.tool_choice,
            Some(tensorzero_types::ToolChoice::Required)
        );
    }

    #[test]
    fn test_tool_choice_specific_conversion() {
        let tool_choice = AnthropicToolChoice::Tool {
            name: "my_tool".to_string(),
        };
        let params = tool_choice.into_tool_params();

        assert_eq!(params.allowed_tools, Some(vec!["my_tool".to_string()]));
        assert_eq!(
            params.tool_choice,
            Some(tensorzero_types::ToolChoice::Specific(
                "my_tool".to_string()
            ))
        );
    }

    #[test]
    fn test_anthropic_tool_serialization() {
        let tool = AnthropicTool {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            input_schema: AnthropicInputSchema {
                schema_type: "object".to_string(),
                properties: Some(HashMap::from_iter([(
                    "param1".to_string(),
                    json!({"type": "string"}),
                )])),
                required: Some(vec!["param1".to_string()]),
                additional_properties: Some(false),
            },
        };

        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["name"], "test_tool");
        assert_eq!(json["description"], "A test tool");
        assert_eq!(json["input_schema"]["type"], "object");
    }

    #[test]
    fn test_anthropic_tool_deserialization() {
        let json = json!({
            "name": "test_tool",
            "description": "A test tool",
            "input_schema": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string"}
                },
                "required": ["param1"]
            }
        });

        let tool: AnthropicTool = serde_json::from_value(json).unwrap();
        assert_eq!(tool.name, "test_tool");
        assert_eq!(tool.description, "A test tool");
        assert_eq!(tool.input_schema.schema_type, "object");
        assert_eq!(tool.input_schema.required, Some(vec!["param1".to_string()]));
    }

    #[test]
    fn test_anthropic_tool_to_tool_conversion() {
        let anthropic_tool = AnthropicTool {
            name: "test_tool".to_string(),
            description: "Test".to_string(),
            input_schema: AnthropicInputSchema {
                schema_type: "object".to_string(),
                properties: None,
                required: None,
                additional_properties: None,
            },
        };

        let tool: Tool = anthropic_tool.into();
        assert!(matches!(tool, Tool::Function(_)));
    }

    #[test]
    fn test_anthropic_tool_to_provider_tool_conversion() {
        let anthropic_tool = AnthropicTool {
            name: "test_tool".to_string(),
            description: "Test".to_string(),
            input_schema: AnthropicInputSchema {
                schema_type: "object".to_string(),
                properties: None,
                required: None,
                additional_properties: None,
            },
        };

        let provider_tool: ProviderTool = anthropic_tool.into();
        assert_eq!(
            provider_tool.scope,
            crate::tool::ProviderToolScope::Unscoped
        );
    }
}
