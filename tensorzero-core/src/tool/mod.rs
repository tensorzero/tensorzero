//! Tool types and configuration for TensorZero.
//!
//! This module provides the core types for working with tools in TensorZero:
//!
//! - **Wire types**: [`ToolCall`], [`ToolResult`], [`ToolChoice`] - API request/response formats
//! - **Tool definitions**: [`Tool`], [`FunctionTool`], [`OpenAICustomTool`] - Tool type definitions
//! - **Configuration**: [`ToolCallConfig`], [`DynamicToolParams`] - Runtime tool configuration
//! - **Storage**: [`ToolCallConfigDatabaseInsert`] - Database persistence format

mod call;
pub mod config;
pub mod params;
pub mod storage;
pub mod types;
pub mod wire;

// Re-export core types for convenience
pub use call::{InferenceResponseToolCallExt, ToolCallChunk};
// Re-export InferenceResponseToolCall from tensorzero-types
pub use config::{
    AllowedTools, AllowedToolsChoice, DynamicImplicitToolConfig, DynamicToolConfig,
    FunctionToolConfig, ImplicitToolConfig, StaticToolConfig, ToolCallConfig,
    ToolCallConfigConstructorArgs, ToolConfig, ToolConfigRef,
};
pub use params::{BatchDynamicToolParams, BatchDynamicToolParamsWithSize, DynamicToolParams};
pub use storage::{
    LegacyToolCallConfigDatabaseInsert, ToolCallConfigDatabaseInsert,
    apply_dynamic_tool_params_update_to_tool_call_config, deserialize_optional_tool_info,
    deserialize_tool_info,
};
pub use tensorzero_types::InferenceResponseToolCall;
pub use types::{
    FunctionTool, OpenAICustomTool, OpenAICustomToolFormat, OpenAIGrammarDefinition,
    OpenAIGrammarSyntax, ProviderTool, ProviderToolScope, ProviderToolScopeModelProvider, Tool,
};
// Re-export tool wire types from tensorzero-types
pub use tensorzero_types::{ToolCall, ToolCallWrapper, ToolChoice, ToolResult};

// Extension traits for tool types
pub use wire::{ToolCallExt, ToolResultExt};

use serde_json::Value;

use crate::jsonschema_util::JSONSchema;

/*  Key tool types in TensorZero
 * - DynamicToolParams: the wire format for tool configuration info (flattened into struct body)
 *       contains a disjoint set of information from that specified in FunctionConfig and config.tools
 * - ToolCallConfig: the representation at inference time of what tool calls are possible
 * - ToolCallConfigDatabaseInsert: the storage format for tool call configuration info
 *     In a close-following PR @viraj will refactor this type.
 * All of these types are convertible given access to the current Config. The conversion from ToolCallConfig
 * to ToolCallConfigDatabaseInsert is temporarily lossy because we don't yet stored dynamic provider tools.
 *
 * Tool: represents a single Tool that could be called by an LLM. This will be generalized soon to an enum.
 * ToolCall: represents a request by an LLM to call a tool.
 * ToolResult: the response from a tool call.
 */

/* A Tool is a function that can be called by an LLM
 * We represent them in various ways depending on how they are configured by the user.
 * The primary difficulty is that tools require an input signature that we represent as a JSONSchema.
 * JSONSchema compilation takes time so we want to do it at startup if the tool is in the config.
 * We also don't want to clone compiled JSON schemas.
 * If the tool is dynamic we want to run compilation while LLM inference is happening so that we can validate the tool call arguments.
 *
 * If we are doing an implicit tool call for JSON schema enforcement, we can use the compiled schema from the output signature.
 */

pub const IMPLICIT_TOOL_NAME: &str = "respond";
pub const IMPLICIT_TOOL_DESCRIPTION: &str = "Respond to the user using the output schema provided.";

pub fn create_dynamic_implicit_tool_config(schema: Value) -> ToolCallConfig {
    let tool_schema = JSONSchema::compile_background(schema);
    let implicit_tool = FunctionToolConfig::DynamicImplicit(DynamicImplicitToolConfig {
        parameters: tool_schema,
    });
    ToolCallConfig {
        static_tools_available: vec![],
        dynamic_tools_available: vec![implicit_tool],
        openai_custom_tools: vec![],
        tool_choice: ToolChoice::Specific(IMPLICIT_TOOL_NAME.to_string()),
        parallel_tool_calls: None,
        provider_tools: vec![],
        allowed_tools: AllowedTools::default(),
    }
}

/// For use in initializing JSON functions
/// Creates a ToolCallConfig with a single implicit tool that takes the schema as arguments
pub fn create_json_mode_tool_call_config(schema: JSONSchema) -> ToolCallConfig {
    create_json_mode_tool_call_config_with_allowed_tools(schema, AllowedTools::default())
}

pub fn create_json_mode_tool_call_config_with_allowed_tools(
    schema: JSONSchema,
    allowed_tools: AllowedTools,
) -> ToolCallConfig {
    let implicit_tool = FunctionToolConfig::Implicit(ImplicitToolConfig { parameters: schema });
    ToolCallConfig {
        static_tools_available: vec![implicit_tool],
        dynamic_tools_available: vec![],
        tool_choice: ToolChoice::Specific(IMPLICIT_TOOL_NAME.to_string()),
        openai_custom_tools: vec![],
        parallel_tool_calls: None,
        provider_tools: vec![],
        allowed_tools,
    }
}

#[cfg(test)]
impl ToolCallConfig {
    #[expect(clippy::missing_panics_doc)]
    pub fn implicit_from_value(value: &Value) -> Self {
        let parameters = JSONSchema::from_value(value.clone()).unwrap();
        let implicit_tool_config = FunctionToolConfig::Implicit(ImplicitToolConfig { parameters });
        Self {
            static_tools_available: vec![implicit_tool_config],
            dynamic_tools_available: vec![],
            openai_custom_tools: vec![],
            tool_choice: ToolChoice::Specific(IMPLICIT_TOOL_NAME.to_string()),
            parallel_tool_calls: None,
            provider_tools: vec![],
            allowed_tools: AllowedTools::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorDetails;
    use lazy_static::lazy_static;
    use serde::Deserialize;
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::Arc;

    lazy_static! {
        static ref TOOLS: HashMap<String, Arc<StaticToolConfig>> = {
            let mut map = HashMap::new();
            map.insert(
                "get_temperature".to_string(),
                Arc::new(StaticToolConfig {
                    name: "get_temperature".to_string(),
                    key: "get_temperature".to_string(),
                    description: "Get the current temperature in a given location".to_string(),
                    parameters: JSONSchema::from_value(json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                        "required": ["location"]
                    }))
                    .expect("Failed to create schema for get_temperature"),
                    strict: true,
                }),
            );
            map.insert(
                "query_articles".to_string(),
                Arc::new(StaticToolConfig {
                    name: "query_articles".to_string(),
                    key: "query_articles".to_string(),
                    description: "Query articles from a database based on given criteria"
                        .to_string(),
                    parameters: JSONSchema::from_value(json!({
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string"},
                            "category": {"type": "string"},
                            "limit": {"type": "integer", "minimum": 1, "maximum": 100}
                        },
                        "required": ["keyword"]
                    }))
                    .expect("Failed to create schema for query_articles"),
                    strict: false,
                }),
            );
            map
        };
        static ref EMPTY_TOOLS: HashMap<String, Arc<StaticToolConfig>> = HashMap::new();
        static ref EMPTY_FUNCTION_TOOLS: Vec<String> = vec![];
        static ref ALL_FUNCTION_TOOLS: Vec<String> =
            vec!["get_temperature".to_string(), "query_articles".to_string()];
        static ref AUTO_TOOL_CHOICE: ToolChoice = ToolChoice::Auto;
        static ref WEATHER_TOOL_CHOICE: ToolChoice =
            ToolChoice::Specific("get_temperature".to_string());
    }

    #[tokio::test]
    async fn test_tool_call_config_new() {
        // Empty tools in function, no dynamic tools, tools are configured in the config
        // This should return no tools because the function does not specify any tools
        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &EMPTY_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            DynamicToolParams::default(),
        ))
        .unwrap();
        assert!(tool_call_config.is_none());

        // All tools available, no dynamic tools, tools are configured in the config
        // This should return all tools because the function specifies all tools
        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            DynamicToolParams::default(),
        ))
        .unwrap()
        .unwrap();
        assert_eq!(tool_call_config.tools_available().unwrap().count(), 2);

        // strict_tools_available should return all tools (FunctionDefault mode)
        assert_eq!(
            tool_call_config.strict_tools_available().unwrap().count(),
            2
        );
        assert!(matches!(
            tool_call_config.allowed_tools.choice,
            AllowedToolsChoice::FunctionDefault
        ));
        assert_eq!(tool_call_config.tool_choice, ToolChoice::Auto);
        assert_eq!(tool_call_config.parallel_tool_calls, Some(true));
        let tools: Vec<_> = tool_call_config.tools_available().unwrap().collect();
        assert!(tools[0].strict());
        assert!(!tools[1].strict());

        // Empty tools in function and config but we specify an allowed tool (should fail)
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec!["get_temperature".to_string()]),
            ..Default::default()
        };
        let err = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &EMPTY_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &EMPTY_TOOLS,
            dynamic_tool_params,
        ))
        .unwrap_err();
        assert_eq!(
            err,
            ErrorDetails::ToolNotFound {
                name: "get_temperature".to_string()
            }
            .into()
        );

        // Dynamic tool config specifies a particular tool to call and it's in the function tools list
        let dynamic_tool_params = DynamicToolParams {
            tool_choice: Some(ToolChoice::Specific("get_temperature".to_string())),
            ..Default::default()
        };
        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        ))
        .unwrap()
        .unwrap();
        assert_eq!(tool_call_config.tools_available().unwrap().count(), 2);
        assert_eq!(
            tool_call_config.tool_choice,
            ToolChoice::Specific("get_temperature".to_string())
        );
        assert_eq!(tool_call_config.parallel_tool_calls, Some(true));

        // Dynamic tool config specifies a particular tool to call and it's not in the function tools list
        let dynamic_tool_params = DynamicToolParams {
            tool_choice: Some(ToolChoice::Specific("establish_campground".to_string())),
            ..Default::default()
        };
        let err = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        ))
        .unwrap_err();
        assert_eq!(
            err,
            ErrorDetails::ToolNotFound {
                name: "establish_campground".to_string()
            }
            .into()
        );

        // We pass an empty list of allowed tools and then configure a new tool
        // All function tools are still included, plus the dynamic tool
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec![]),
            additional_tools: Some(vec![Tool::Function(FunctionTool {
                name: "establish_campground".to_string(),
                description: "Establish a campground".to_string(),
                parameters: json!({}),
                strict: false,
            })]),
            ..Default::default()
        };
        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        ))
        .unwrap()
        .unwrap();
        // Should have all function tools (get_temperature, query_articles) + dynamic tool (establish_campground)
        assert_eq!(tool_call_config.tools_available().unwrap().count(), 3);
        assert!(
            tool_call_config
                .tools_available()
                .unwrap()
                .any(|t| t.name() == "get_temperature")
        );
        assert!(
            tool_call_config
                .tools_available()
                .unwrap()
                .any(|t| t.name() == "query_articles")
        );
        assert!(
            tool_call_config
                .tools_available()
                .unwrap()
                .any(|t| t.name() == "establish_campground")
        );

        // We pass a list of a single allowed tool and then configure a new tool
        // All function tools are still included, plus the dynamic tool
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec!["get_temperature".to_string()]),
            additional_tools: Some(vec![Tool::Function(FunctionTool {
                name: "establish_campground".to_string(),
                description: "Establish a campground".to_string(),
                parameters: json!({}),
                strict: false,
            })]),
            parallel_tool_calls: Some(false),
            ..Default::default()
        };
        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        ))
        .unwrap()
        .unwrap();
        // Should have all function tools + dynamic tool
        assert_eq!(tool_call_config.tools_available().unwrap().count(), 3);
        assert!(
            tool_call_config
                .tools_available()
                .unwrap()
                .any(|t| t.name() == "get_temperature")
        );
        assert!(
            tool_call_config
                .tools_available()
                .unwrap()
                .any(|t| t.name() == "query_articles")
        );
        assert!(
            tool_call_config
                .tools_available()
                .unwrap()
                .any(|t| t.name() == "establish_campground")
        );
        assert_eq!(tool_call_config.parallel_tool_calls, Some(false));

        // We pass a list of no allowed tools and then configure a new tool
        // All function tools are still included, plus the dynamic tool
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec![]),
            additional_tools: Some(vec![Tool::Function(FunctionTool {
                name: "establish_campground".to_string(),
                description: "Establish a campground".to_string(),
                parameters: json!({}),
                strict: false,
            })]),
            tool_choice: Some(ToolChoice::Specific("establish_campground".to_string())),
            ..Default::default()
        };
        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        ))
        .unwrap()
        .unwrap();
        // Should have all function tools + dynamic tool
        assert_eq!(tool_call_config.tools_available().unwrap().count(), 3);
        assert!(
            tool_call_config
                .tools_available()
                .unwrap()
                .any(|t| t.name() == "establish_campground")
        );
        assert_eq!(tool_call_config.parallel_tool_calls, Some(true));
        assert_eq!(
            tool_call_config.tool_choice,
            ToolChoice::Specific("establish_campground".to_string())
        );
    }

    #[tokio::test]
    async fn test_inference_response_tool_call_new() {
        let tool_call = ToolCall {
            name: "get_temperature".to_string(),
            arguments: "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}".to_string(),
            id: "123".to_string(),
        };
        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            DynamicToolParams::default(),
        ))
        .unwrap()
        .unwrap();
        // Tool call is valid, so we should get a valid InferenceResponseToolCall
        let inference_response_tool_call =
            InferenceResponseToolCall::new_from_tool_call(tool_call, Some(&tool_call_config)).await;
        assert_eq!(inference_response_tool_call.raw_name, "get_temperature");
        assert_eq!(
            inference_response_tool_call.raw_arguments,
            "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}"
        );
        assert_eq!(inference_response_tool_call.id, "123");
        assert_eq!(
            inference_response_tool_call.name,
            Some("get_temperature".to_string())
        );
        assert_eq!(
            inference_response_tool_call.arguments,
            Some(json!({
                "location": "San Francisco",
                "unit": "celsius"
            }))
        );

        // Bad arguments, but valid name (parsed_name is set but parsed_arguments is not)
        let tool_call = ToolCall {
            name: "get_temperature".to_string(),
            arguments: "{\"location\": \"San Francisco\", \"unit\": \"kelvin\"}".to_string(),
            id: "321".to_string(),
        };
        let inference_response_tool_call =
            InferenceResponseToolCall::new_from_tool_call(tool_call, Some(&tool_call_config)).await;
        assert_eq!(
            inference_response_tool_call.name,
            Some("get_temperature".to_string())
        );
        assert_eq!(inference_response_tool_call.arguments, None);
        assert_eq!(inference_response_tool_call.id, "321");
        assert_eq!(inference_response_tool_call.raw_name, "get_temperature");
        assert_eq!(
            inference_response_tool_call.raw_arguments,
            "{\"location\": \"San Francisco\", \"unit\": \"kelvin\"}"
        );

        // Bad name, good arguments (both not set since the name is invalid and we can't be sure what tool this goes to)
        let tool_call = ToolCall {
            name: "not_get_weather".to_string(),
            arguments: "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}".to_string(),
            id: "321".to_string(),
        };
        let inference_response_tool_call =
            InferenceResponseToolCall::new_from_tool_call(tool_call, Some(&tool_call_config)).await;
        assert_eq!(inference_response_tool_call.name, None);
        assert_eq!(inference_response_tool_call.arguments, None);
        assert_eq!(inference_response_tool_call.id, "321");
        assert_eq!(inference_response_tool_call.raw_name, "not_get_weather");
        assert_eq!(
            inference_response_tool_call.raw_arguments,
            "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}"
        );

        // Make sure validation works with dynamic tools
        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            DynamicToolParams {
                additional_tools: Some(vec![Tool::Function(FunctionTool {
                    name: "establish_campground".to_string(),
                    description: "Establish a campground".to_string(),
                    parameters: json!({"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}),
                    strict: false,
                })]),
                ..Default::default()
            },
        ))
        .unwrap()
        .unwrap();
        let tool_call = ToolCall {
            name: "establish_campground".to_string(),
            arguments: "{\"location\": \"Lucky Dog\"}".to_string(),
            id: "321".to_string(),
        };
        let inference_response_tool_call =
            InferenceResponseToolCall::new_from_tool_call(tool_call, Some(&tool_call_config)).await;
        assert_eq!(
            inference_response_tool_call.raw_name,
            "establish_campground"
        );
        assert_eq!(
            inference_response_tool_call.raw_arguments,
            "{\"location\": \"Lucky Dog\"}"
        );
        assert_eq!(inference_response_tool_call.id, "321");
        assert_eq!(
            inference_response_tool_call.name,
            Some("establish_campground".to_string())
        );
        assert_eq!(
            inference_response_tool_call.arguments,
            Some(json!({"location": "Lucky Dog"}))
        );
    }

    #[tokio::test]
    async fn test_inference_response_tool_call_with_custom_tools() {
        // Create a ToolCallConfig with a custom tool
        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &EMPTY_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &EMPTY_TOOLS,
            DynamicToolParams {
                additional_tools: Some(vec![Tool::OpenAICustom(OpenAICustomTool {
                    name: "code_generator".to_string(),
                    description: Some("Generates code snippets".to_string()),
                    format: Some(OpenAICustomToolFormat::Text),
                })]),
                ..Default::default()
            },
        ))
        .unwrap()
        .unwrap();

        // Valid custom tool call - name should be validated
        let tool_call = ToolCall {
            name: "code_generator".to_string(),
            arguments: "{\"description\": \"Print hello world\"}".to_string(),
            id: "ctc_123".to_string(),
        };
        let inference_response_tool_call =
            InferenceResponseToolCall::new_from_tool_call(tool_call, Some(&tool_call_config)).await;

        // The parsed_name should be set since this is a valid custom tool
        assert_eq!(
            inference_response_tool_call.name,
            Some("code_generator".to_string())
        );
        assert_eq!(inference_response_tool_call.raw_name, "code_generator");
        assert_eq!(
            inference_response_tool_call.raw_arguments,
            "{\"description\": \"Print hello world\"}"
        );
        assert_eq!(inference_response_tool_call.id, "ctc_123");
        // Custom tools don't validate arguments against JSON schemas, so parsed_arguments should be None
        assert_eq!(inference_response_tool_call.arguments, None);

        // Invalid custom tool name - name should not be validated
        let tool_call = ToolCall {
            name: "not_a_custom_tool".to_string(),
            arguments: "{\"description\": \"Test\"}".to_string(),
            id: "ctc_456".to_string(),
        };
        let inference_response_tool_call =
            InferenceResponseToolCall::new_from_tool_call(tool_call, Some(&tool_call_config)).await;

        // The parsed_name should be None since this tool doesn't exist
        assert_eq!(inference_response_tool_call.name, None);
        assert_eq!(inference_response_tool_call.raw_name, "not_a_custom_tool");
        assert_eq!(
            inference_response_tool_call.raw_arguments,
            "{\"description\": \"Test\"}"
        );
        assert_eq!(inference_response_tool_call.id, "ctc_456");
        assert_eq!(inference_response_tool_call.arguments, None);

        // Test with both function tools and custom tools
        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            DynamicToolParams {
                additional_tools: Some(vec![Tool::OpenAICustom(OpenAICustomTool {
                    name: "calculator".to_string(),
                    description: Some("Performs calculations".to_string()),
                    format: Some(OpenAICustomToolFormat::Grammar {
                        grammar: OpenAIGrammarDefinition {
                            syntax: OpenAIGrammarSyntax::Lark,
                            definition: "start: NUMBER".to_string(),
                        },
                    }),
                })]),
                ..Default::default()
            },
        ))
        .unwrap()
        .unwrap();

        // Valid function tool should still work
        let tool_call = ToolCall {
            name: "get_temperature".to_string(),
            arguments: "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}".to_string(),
            id: "123".to_string(),
        };
        let inference_response_tool_call =
            InferenceResponseToolCall::new_from_tool_call(tool_call, Some(&tool_call_config)).await;
        assert_eq!(
            inference_response_tool_call.name,
            Some("get_temperature".to_string())
        );
        assert_eq!(
            inference_response_tool_call.arguments,
            Some(json!({
                "location": "San Francisco",
                "unit": "celsius"
            }))
        );

        // Valid custom tool should also work
        let tool_call = ToolCall {
            name: "calculator".to_string(),
            arguments: "42".to_string(),
            id: "ctc_789".to_string(),
        };
        let inference_response_tool_call =
            InferenceResponseToolCall::new_from_tool_call(tool_call, Some(&tool_call_config)).await;
        assert_eq!(
            inference_response_tool_call.name,
            Some("calculator".to_string())
        );
        assert_eq!(inference_response_tool_call.raw_name, "calculator");
        assert_eq!(inference_response_tool_call.raw_arguments, "42");
        assert_eq!(inference_response_tool_call.id, "ctc_789");
        // Custom tools don't validate arguments, so parsed_arguments is None
        assert_eq!(inference_response_tool_call.arguments, None);
    }

    #[test]
    fn test_tool_call_deserialize_plain_raw() {
        let tool_call = serde_json::json!({
            "name": "get_temperature",
            "raw_name": "should have ignored raw name",
            "arguments": "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}",
            "raw_arguments": "should have ignored raw arguments",
            "id": "123"
        });
        let tool_call: ToolCall = serde_json::from_value(tool_call).unwrap();
        assert_eq!(tool_call.name, "get_temperature");
        assert_eq!(
            tool_call.arguments,
            "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}"
        );
        assert_eq!(tool_call.id, "123");
    }

    #[test]
    fn test_tool_call_deserialize_raw_only() {
        let tool_call = serde_json::json!({
            "raw_name": "get_temperature",
            "raw_arguments": "my raw arguments",
            "id": "123"
        });
        let tool_call_wrapper: ToolCallWrapper = serde_json::from_value(tool_call).unwrap();
        let tool_call: ToolCall = tool_call_wrapper.into_tool_call();
        assert_eq!(tool_call.name, "get_temperature");
        assert_eq!(tool_call.arguments, "my raw arguments");
        assert_eq!(tool_call.id, "123");
    }

    #[test]
    fn test_tool_call_deserialize_arguments_object() {
        let tool_call = serde_json::json!({
            "name": "get_temperature",
            "arguments": {"my": "arguments"},
            "id": "123"
        });
        let tool_call_wrapper = serde_json::from_value::<ToolCallWrapper>(tool_call).unwrap();
        let tool_call = tool_call_wrapper.into_tool_call();
        assert_eq!(tool_call.name, "get_temperature");
        assert_eq!(tool_call.arguments, "{\"my\":\"arguments\"}");
        assert_eq!(tool_call.id, "123");
    }

    #[test]
    fn test_tool_call_deserialize_arguments_string() {
        let tool_call = serde_json::json!({
            "name": "get_temperature",
            "arguments": "{\"my\": \"arguments\"}",
            "id": "123"
        });
        let tool_call: ToolCall = serde_json::from_value(tool_call).unwrap();
        assert_eq!(tool_call.name, "get_temperature");
        assert_eq!(tool_call.arguments, "{\"my\": \"arguments\"}");
        assert_eq!(tool_call.id, "123");
    }

    #[test]
    fn test_tool_call_deserialize_missing_name() {
        let tool_call = serde_json::json!({
            "arguments": "{\"my\": \"arguments\"}",
            "id": "123"
        });
        // Now we get an ugly error because of the untagged enum, but that's ok for now...
        // https://github.com/tensorzero/tensorzero/discussions/4258
        serde_json::from_value::<ToolCallWrapper>(tool_call).unwrap_err();
    }

    #[test]
    fn test_tool_call_deserialize_missing_arguments() {
        let tool_call = serde_json::json!({
            "name": "get_temperature",
            "id": "123"
        });
        let err_msg = serde_json::from_value::<ToolCall>(tool_call)
            .unwrap_err()
            .to_string();
        assert_eq!(err_msg, "missing field `arguments`");
    }

    #[test]
    fn test_tool_call_deserialize_object_arguments() {
        let tool_call = serde_json::json!({
            "name": "get_temperature",
            "id": "123",
            "arguments": {
                "role": "intern"
            }
        });
        let tool_call_wrapper = serde_json::from_value::<ToolCallWrapper>(tool_call).unwrap();
        let tool_call = tool_call_wrapper.into_tool_call();
        assert_eq!(tool_call.arguments, "{\"role\":\"intern\"}");
        assert_eq!(tool_call.name, "get_temperature");
        assert_eq!(tool_call.id, "123");
    }

    #[tokio::test]
    async fn test_duplicate_tool_names_error() {
        // Test case where dynamic tool params add a tool with the same name as a static tool
        let dynamic_tool_params = DynamicToolParams {
            additional_tools: Some(vec![Tool::Function(FunctionTool {
                name: "get_temperature".to_string(), // Same name as static tool
                description: "Another temperature tool".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }),
                strict: false,
            })]),
            ..Default::default()
        };

        let err = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        ))
        .unwrap_err();

        assert_eq!(
            err,
            ErrorDetails::DuplicateTool {
                name: "get_temperature".to_string()
            }
            .into()
        );
    }

    #[tokio::test]
    async fn test_duplicate_custom_tool_names_error() {
        // Test case where two custom tools have the same name
        let dynamic_tool_params = DynamicToolParams {
            additional_tools: Some(vec![
                Tool::OpenAICustom(OpenAICustomTool {
                    name: "custom_tool".to_string(),
                    description: Some("First custom tool".to_string()),
                    format: None,
                }),
                Tool::OpenAICustom(OpenAICustomTool {
                    name: "custom_tool".to_string(), // Duplicate name
                    description: Some("Second custom tool".to_string()),
                    format: Some(OpenAICustomToolFormat::Text),
                }),
            ]),
            ..Default::default()
        };

        let err = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &EMPTY_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        ))
        .unwrap_err();

        assert_eq!(
            err,
            ErrorDetails::DuplicateTool {
                name: "custom_tool".to_string()
            }
            .into()
        );
    }

    #[tokio::test]
    async fn test_custom_tool_conflicts_with_function_tool() {
        // Test case where a custom tool has the same name as a function tool
        let dynamic_tool_params = DynamicToolParams {
            additional_tools: Some(vec![Tool::OpenAICustom(OpenAICustomTool {
                name: "get_temperature".to_string(), // Same name as static function tool
                description: Some("Custom temperature tool".to_string()),
                format: None,
            })]),
            ..Default::default()
        };

        let err = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        ))
        .unwrap_err();

        assert_eq!(
            err,
            ErrorDetails::DuplicateTool {
                name: "get_temperature".to_string()
            }
            .into()
        );
    }

    #[test]
    fn test_get_scoped_provider_tools() {
        // Set up provider tools with different scopes
        let provider_tools = vec![
            ProviderTool {
                scope: ProviderToolScope::Unscoped,
                tool: json!({"type": "unscoped_tool"}),
            },
            ProviderTool {
                scope: ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
                    model_name: "gpt-4".to_string(),
                    provider_name: Some("openai".to_string()),
                }),
                tool: json!({"type": "gpt4_tool"}),
            },
            ProviderTool {
                scope: ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
                    model_name: "claude-4".to_string(),
                    provider_name: Some("anthropic".to_string()),
                }),
                tool: json!({"type": "claude_tool"}),
            },
        ];

        let config = ToolCallConfig {
            provider_tools,
            ..Default::default()
        };

        // Test matching gpt-4/openai: should return unscoped + gpt4_tool
        let result = config.get_scoped_provider_tools("gpt-4", "openai");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].tool, json!({"type": "unscoped_tool"}));
        assert_eq!(result[1].tool, json!({"type": "gpt4_tool"}));

        // Test matching claude-4/anthropic: should return unscoped + claude_tool
        let result = config.get_scoped_provider_tools("claude-4", "anthropic");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].tool, json!({"type": "unscoped_tool"}));
        assert_eq!(result[1].tool, json!({"type": "claude_tool"}));

        // Test non-matching model: should return only unscoped
        let result = config.get_scoped_provider_tools("llama-2", "meta");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].tool, json!({"type": "unscoped_tool"}));

        // Test partial match (correct model, wrong provider): should return only unscoped
        let result = config.get_scoped_provider_tools("gpt-4", "azure");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].tool, json!({"type": "unscoped_tool"}));

        // Test with None provider_tools
        let config_no_tools = ToolCallConfig::with_tools_available(vec![], vec![]);
        let result = config_no_tools.get_scoped_provider_tools("gpt-4", "openai");
        assert_eq!(result.len(), 0);
    }

    #[tokio::test]
    async fn test_dynamic_tool_in_allowed_tools() {
        // Test that a dynamic tool name in allowed_tools is recognized and doesn't error
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec![
                "get_temperature".to_string(),
                "establish_campground".to_string(),
            ]),
            additional_tools: Some(vec![Tool::Function(FunctionTool {
                name: "establish_campground".to_string(),
                description: "Establish a campground".to_string(),
                parameters: json!({"type": "object", "properties": {"location": {"type": "string"}}}),
                strict: false,
            })]),
            ..Default::default()
        };

        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        ))
        .unwrap()
        .unwrap();

        // Should have all function tools plus dynamic tools
        // function_tools: get_temperature, query_articles
        // dynamic tools: establish_campground
        assert_eq!(tool_call_config.tools_available().unwrap().count(), 3);

        // Verify the static tools are included
        assert!(
            tool_call_config
                .tools_available()
                .unwrap()
                .any(|t| t.name() == "get_temperature")
        );
        assert!(
            tool_call_config
                .tools_available()
                .unwrap()
                .any(|t| t.name() == "query_articles")
        );

        // Verify the dynamic tool is included
        assert!(
            tool_call_config
                .tools_available()
                .unwrap()
                .any(|t| t.name() == "establish_campground")
        );

        // strict_tools_available should filter to only allowed_tools (AllAllowedTools mode)
        assert_eq!(
            tool_call_config.strict_tools_available().unwrap().count(),
            2
        );
        assert!(
            tool_call_config
                .strict_tools_available()
                .unwrap()
                .any(|t| t.name() == "get_temperature")
        );
        assert!(
            tool_call_config
                .strict_tools_available()
                .unwrap()
                .any(|t| t.name() == "establish_campground")
        );
    }

    #[tokio::test]
    async fn test_allowed_tool_not_found_in_static_or_dynamic() {
        // Test that a tool name in allowed_tools that's not in static_tools or additional_tools throws error
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec![
                "get_temperature".to_string(),
                "nonexistent_tool".to_string(),
            ]),
            additional_tools: Some(vec![Tool::Function(FunctionTool {
                name: "establish_campground".to_string(),
                description: "Establish a campground".to_string(),
                parameters: json!({"type": "object"}),
                strict: false,
            })]),
            ..Default::default()
        };

        let err = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        ))
        .unwrap_err();

        assert_eq!(
            err,
            ErrorDetails::ToolNotFound {
                name: "nonexistent_tool".to_string()
            }
            .into()
        );
    }

    #[tokio::test]
    async fn test_dynamic_tool_not_auto_added_to_allowed_tools() {
        // Test that dynamic tools are sent as definitions but not added to allowed_tools
        // when allowed_tools is explicitly set (AllAllowedTools mode)
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec!["get_temperature".to_string()]),
            additional_tools: Some(vec![Tool::Function(FunctionTool {
                name: "establish_campground".to_string(),
                description: "Establish a campground".to_string(),
                parameters: json!({"type": "object", "properties": {"location": {"type": "string"}}}),
                strict: false,
            })]),
            ..Default::default()
        };

        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        ))
        .unwrap()
        .unwrap();

        // All tool definitions should be available (sent to provider)
        // function_tools: get_temperature, query_articles
        // dynamic tools: establish_campground
        assert_eq!(tool_call_config.tools_available().unwrap().count(), 3);
        assert!(
            tool_call_config
                .tools_available()
                .unwrap()
                .any(|t| t.name() == "get_temperature")
        );
        assert!(
            tool_call_config
                .tools_available()
                .unwrap()
                .any(|t| t.name() == "query_articles")
        );
        assert!(
            tool_call_config
                .tools_available()
                .unwrap()
                .any(|t| t.name() == "establish_campground")
        );

        // But only get_temperature should be in allowed_tools
        assert_eq!(tool_call_config.allowed_tools.tools.len(), 1);
        assert!(
            tool_call_config
                .allowed_tools
                .tools
                .contains(&"get_temperature".to_string())
        );
        assert!(matches!(
            tool_call_config.allowed_tools.choice,
            AllowedToolsChoice::Explicit
        ));

        // strict_tools_available should filter to only allowed_tools (AllAllowedTools mode)
        assert_eq!(
            tool_call_config.strict_tools_available().unwrap().count(),
            1
        );
        assert!(
            tool_call_config
                .strict_tools_available()
                .unwrap()
                .any(|t| t.name() == "get_temperature")
        );
    }

    // Helper struct to test deserialization with flattening
    #[derive(Debug, Deserialize, PartialEq)]
    struct ToolCallConfigDeserializeTestHelper {
        baz: String,
        #[serde(flatten)]
        #[serde(deserialize_with = "deserialize_optional_tool_info")]
        tool_info: Option<ToolCallConfigDatabaseInsert>,
    }

    // Helper function to assert that deserialization results in None for tool_info
    fn assert_deserialize_to_none(json: serde_json::Value, expected_baz: &str) {
        let result: ToolCallConfigDeserializeTestHelper =
            serde_json::from_value(json).expect("Deserialization should succeed");
        assert_eq!(result.baz, expected_baz);
        assert_eq!(result.tool_info, None, "tool_info should be None");
    }

    #[test]
    fn test_tool_call_config_database_insert_deserialize_ragged_with_flatten() {
        // Test with a flattened struct (ragged case)
        // Note: dynamic_tools and dynamic_provider_tools are arrays of JSON strings
        // allowed_tools is a JSON string, tool_choice is a bare string/object
        let json = json!({
            "baz": "test_value",
            "dynamic_tools": [
                r#"{"type":"function","name":"ragged_tool","description":"A ragged tool","parameters":{"type":"string"},"strict":true}"#
            ],
            "dynamic_provider_tools": [],
            "allowed_tools": r#"{"tools":["ragged_tool"],"choice":"function_default"}"#,
            "tool_choice": {"specific": "ragged_tool"},
            "parallel_tool_calls": null,
            "tool_params": {
                "tools_available": [],
                "tool_choice": {"specific": "ragged_tool"},
                "parallel_tool_calls": null
            }
        });

        let result: ToolCallConfigDeserializeTestHelper = serde_json::from_value(json).unwrap();

        assert_eq!(result.baz, "test_value");
        let tool_info = result.tool_info.unwrap();
        assert_eq!(tool_info.dynamic_tools.len(), 1);
        assert_eq!(tool_info.dynamic_tools[0].name(), "ragged_tool");
        assert_eq!(tool_info.dynamic_provider_tools.len(), 0);
        assert_eq!(tool_info.allowed_tools.tools, vec!["ragged_tool"]);
        assert_eq!(
            tool_info.tool_choice,
            ToolChoice::Specific("ragged_tool".to_string())
        );
        assert_eq!(tool_info.parallel_tool_calls, None);
    }

    #[test]
    fn test_tool_call_config_database_insert_deserialize_ragged_legacy() {
        // Test legacy format with flattening
        let json = json!({
            "baz": "legacy_value",
            "tool_params": {
                "tools_available": [
                    {
                        "name": "legacy_ragged_tool",
                        "description": "A legacy ragged tool",
                        "parameters": {"type": "number"},
                        "strict": false
                    }
                ],
                "tool_choice": "none",
                "parallel_tool_calls": true
            }
        });

        let result: ToolCallConfigDeserializeTestHelper = serde_json::from_value(json).unwrap();

        assert_eq!(result.baz, "legacy_value");
        let tool_info = result.tool_info.unwrap();
        assert_eq!(tool_info.dynamic_tools.len(), 0);
        assert_eq!(tool_info.dynamic_provider_tools.len(), 0);
        assert_eq!(tool_info.tool_choice, ToolChoice::None);
        assert_eq!(tool_info.parallel_tool_calls, Some(true));
    }

    #[test]
    fn test_tool_call_config_database_insert_deserialize_ragged_empty() {
        // Test empty format with flattening
        let json = json!({
            "baz": "empty_value"
        });
        assert_deserialize_to_none(json, "empty_value");
    }

    #[test]
    fn test_tool_call_config_database_insert_deserialize_legacy_null_tool_params() {
        // Test legacy format with explicit null tool_params
        // Should return None, same as missing tool_params
        let json = json!({
            "baz": "test_value",
            "tool_params": null
        });
        assert_deserialize_to_none(json, "test_value");
    }

    #[test]
    fn test_tool_call_config_database_insert_deserialize_legacy_empty_tool_params() {
        // Test legacy format with empty string tool_params
        // Should return None
        let json = json!({
            "baz": "test_value",
            "tool_params": ""
        });
        assert_deserialize_to_none(json, "test_value");
    }

    #[test]
    fn test_strict_tools_available_with_function_default() {
        // Test that FunctionDefault returns all available tools
        let config = ToolCallConfig {
            static_tools_available: vec![
                FunctionToolConfig::Static(TOOLS.get("get_temperature").unwrap().clone()),
                FunctionToolConfig::Static(TOOLS.get("query_articles").unwrap().clone()),
            ],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
            allowed_tools: AllowedTools::default(), // FunctionDefault
        };

        let tools: Vec<_> = config.strict_tools_available().unwrap().collect();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name(), "get_temperature");
        assert_eq!(tools[1].name(), "query_articles");
    }

    #[test]
    fn test_strict_tools_available_with_all_allowed_tools() {
        // Test that AllAllowedTools filters to the specified subset
        let config = ToolCallConfig {
            static_tools_available: vec![
                FunctionToolConfig::Static(TOOLS.get("get_temperature").unwrap().clone()),
                FunctionToolConfig::Static(TOOLS.get("query_articles").unwrap().clone()),
            ],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
            allowed_tools: AllowedTools {
                tools: vec!["get_temperature".to_string()].into_iter().collect(),
                choice: AllowedToolsChoice::Explicit,
            },
        };

        let tools: Vec<_> = config.strict_tools_available().unwrap().collect();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "get_temperature");
    }

    /// Test that allowed_tools filtering uses the tool's key (HashMap key), not the display name.
    /// This is important when a tool has a custom `name` field that differs from its config key.
    #[test]
    fn test_strict_tools_available_filters_by_key_not_name() {
        // Create a tool where key != name (simulating `[tools.my_tool_key]\nname = "display_name"`)
        let tool_with_custom_name = Arc::new(StaticToolConfig {
            key: "my_tool_key".to_string(),
            name: "display_name_for_llm".to_string(),
            description: "A tool with different key and name".to_string(),
            parameters: JSONSchema::from_value(json!({
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                }
            }))
            .expect("Failed to create schema"),
            strict: false,
        });

        let config = ToolCallConfig {
            static_tools_available: vec![FunctionToolConfig::Static(tool_with_custom_name)],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
            // allowed_tools should use the KEY, not the display name
            allowed_tools: AllowedTools {
                tools: vec!["my_tool_key".to_string()],
                choice: AllowedToolsChoice::Explicit,
            },
        };

        // strict_tools_available should find the tool by its key
        let tools: Vec<_> = config.strict_tools_available().unwrap().collect();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].key(), "my_tool_key");
        assert_eq!(tools[0].name(), "display_name_for_llm");

        // Now test that filtering by the display name does NOT work
        // (this would have been a bug before the fix)
        let config_with_wrong_filter = ToolCallConfig {
            static_tools_available: vec![FunctionToolConfig::Static(Arc::new(StaticToolConfig {
                key: "my_tool_key".to_string(),
                name: "display_name_for_llm".to_string(),
                description: "A tool with different key and name".to_string(),
                parameters: JSONSchema::from_value(json!({
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"}
                    }
                }))
                .expect("Failed to create schema"),
                strict: false,
            }))],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
            // Try to filter by display name - this should NOT match
            allowed_tools: AllowedTools {
                tools: vec!["display_name_for_llm".to_string()],
                choice: AllowedToolsChoice::Explicit,
            },
        };

        // Should return 0 tools because "display_name_for_llm" is not a valid key
        let tools: Vec<_> = config_with_wrong_filter
            .strict_tools_available()
            .unwrap()
            .collect();
        assert_eq!(tools.len(), 0);
    }

    /// Test DynamicTool deserialization with untagged (legacy) format for backward compatibility
    #[test]
    fn test_dynamic_tool_deserialize_untagged_function() {
        let json = json!({
            "name": "legacy_tool",
            "description": "A tool in legacy format",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                }
            },
            "strict": false
        });

        let result: Tool = serde_json::from_value(json).unwrap();
        assert!(matches!(result, Tool::Function(_)));
        if let Tool::Function(func) = result {
            assert_eq!(func.name, "legacy_tool");
            assert_eq!(func.description, "A tool in legacy format");
            assert!(!func.strict);
        }
    }

    /// Test DynamicTool deserialization with new tagged format
    #[test]
    fn test_dynamic_tool_deserialize_tagged_formats() {
        // Test tagged function format
        let function_json = json!({
            "type": "function",
            "name": "tagged_function",
            "description": "A function tool with tag",
            "parameters": {
                "type": "object",
                "properties": {
                    "arg": {"type": "string"}
                }
            },
            "strict": true
        });

        let result: Tool = serde_json::from_value(function_json).unwrap();
        assert!(matches!(result, Tool::Function(_)));
        assert!(result.is_function());
        assert!(!result.is_custom());

        // Test tagged custom format
        let custom_json = json!({
            "type": "openai_custom",
            "name": "custom_tool",
            "description": "A custom tool",
            "format": {
                "type": "text"
            }
        });

        let result: Tool = serde_json::from_value(custom_json).unwrap();
        assert!(matches!(result, Tool::OpenAICustom(_)));
        assert!(result.is_custom());
        assert!(!result.is_function());
    }

    /// Test Tool enum serialization round-trip and helper methods
    #[test]
    fn test_tool_serialization_and_methods() {
        // Function tool round-trip
        let function_tool = Tool::Function(FunctionTool {
            name: "test_func".to_string(),
            description: "Test function".to_string(),
            parameters: json!({"type": "object"}),
            strict: true,
        });

        let json = serde_json::to_value(&function_tool).unwrap();
        let deserialized: Tool = serde_json::from_value(json).unwrap();
        assert_eq!(function_tool, deserialized);
        assert_eq!(deserialized.name(), "test_func");
        assert!(!deserialized.is_custom());
        assert!(deserialized.is_function());

        // Custom tool round-trip
        let custom_tool = Tool::OpenAICustom(OpenAICustomTool {
            name: "custom_func".to_string(),
            description: Some("Custom function".to_string()),
            format: Some(OpenAICustomToolFormat::Text),
        });

        let json = serde_json::to_value(&custom_tool).unwrap();
        let deserialized: Tool = serde_json::from_value(json).unwrap();
        assert_eq!(custom_tool, deserialized);
        assert_eq!(deserialized.name(), "custom_func");
        assert!(deserialized.is_custom());
        assert!(!deserialized.is_function());
    }

    /// Test that ToolCallConfig is created (not None) when ONLY custom tools are provided
    #[tokio::test]
    async fn test_tool_call_config_with_only_custom_tools() {
        // Create params with ONLY custom tools - no function tools, no provider tools
        let dynamic_params = DynamicToolParams {
            allowed_tools: None,
            additional_tools: Some(vec![
                Tool::OpenAICustom(OpenAICustomTool {
                    name: "only_custom_1".to_string(),
                    description: Some("First custom tool".to_string()),
                    format: Some(OpenAICustomToolFormat::Text),
                }),
                Tool::OpenAICustom(OpenAICustomTool {
                    name: "only_custom_2".to_string(),
                    description: Some("Second custom tool".to_string()),
                    format: Some(OpenAICustomToolFormat::Grammar {
                        grammar: OpenAIGrammarDefinition {
                            syntax: OpenAIGrammarSyntax::Lark,
                            definition: "start: WORD+".to_string(),
                        },
                    }),
                }),
            ]),
            tool_choice: None,
            parallel_tool_calls: None,
            provider_tools: vec![],
        };

        // Create ToolCallConfig with NO static function tools
        let tool_call_config_result =
            ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
                &EMPTY_FUNCTION_TOOLS, // No static function tools
                &AUTO_TOOL_CHOICE,
                Some(true),
                &TOOLS,
                dynamic_params,
            ))
            .unwrap();

        // CRITICAL: This should be Some, not None, even though there are no function tools
        assert!(
            tool_call_config_result.is_some(),
            "ToolCallConfig should be Some when only custom tools are provided"
        );

        let tool_call_config = tool_call_config_result.unwrap();

        // Verify custom tools are present
        assert_eq!(tool_call_config.openai_custom_tools.len(), 2);
        assert_eq!(
            tool_call_config.openai_custom_tools[0].name,
            "only_custom_1"
        );
        assert_eq!(
            tool_call_config.openai_custom_tools[1].name,
            "only_custom_2"
        );

        // Verify no function tools are present
        // tools_available() should error when custom tools are present
        assert!(tool_call_config.tools_available().is_err());
    }

    /// Test that tools_available() returns an error when custom tools are present
    #[tokio::test]
    async fn test_tools_available_errors_with_custom_tools() {
        let dynamic_params = DynamicToolParams {
            allowed_tools: Some(vec!["get_temperature".to_string()]),
            additional_tools: Some(vec![Tool::OpenAICustom(OpenAICustomTool {
                name: "custom_tool".to_string(),
                description: Some("A custom tool".to_string()),
                format: Some(OpenAICustomToolFormat::Text),
            })]),
            tool_choice: None,
            parallel_tool_calls: None,
            provider_tools: vec![],
        };

        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &["get_temperature".to_string()],
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_params,
        ))
        .unwrap()
        .unwrap();

        // tools_available() should error
        let result = tool_call_config.tools_available();
        assert!(result.is_err());

        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("Expected error, got Ok"),
        };
        assert!(matches!(
            err.get_details(),
            ErrorDetails::IncompatibleTool { .. }
        ));
    }

    /// Test that tools_available_with_openai_custom() works correctly with mixed tools
    #[tokio::test]
    async fn test_tools_available_with_openai_custom_mixed_tools() {
        let dynamic_params = DynamicToolParams {
            allowed_tools: Some(vec![
                "get_temperature".to_string(),
                "query_articles".to_string(),
            ]),
            additional_tools: Some(vec![Tool::OpenAICustom(OpenAICustomTool {
                name: "custom_1".to_string(),
                description: Some("Custom tool".to_string()),
                format: Some(OpenAICustomToolFormat::Text),
            })]),
            tool_choice: None,
            parallel_tool_calls: None,
            provider_tools: vec![],
        };

        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &["get_temperature".to_string()],
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_params,
        ))
        .unwrap()
        .unwrap();

        // Should have 3 tools total (2 function + 1 custom)
        let tools: Vec<_> = tool_call_config
            .tools_available_with_openai_custom()
            .collect();
        assert_eq!(tools.len(), 3);

        // Count function vs custom tools
        let mut function_count = 0;
        let mut custom_count = 0;

        for tool_ref in tools {
            match tool_ref {
                ToolConfigRef::Function(_) => function_count += 1,
                ToolConfigRef::OpenAICustom(_) => custom_count += 1,
            }
        }

        assert_eq!(function_count, 2);
        assert_eq!(custom_count, 1);
    }

    #[test]
    fn test_provider_tool_scope_deserialize_new_format_with_provider() {
        let json = r#"{"model_name": "gpt-4", "provider_name": "openai"}"#;
        let scope: ProviderToolScope = serde_json::from_str(json).unwrap();
        assert_eq!(
            scope,
            ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
                model_name: "gpt-4".to_string(),
                provider_name: Some("openai".to_string()),
            })
        );
    }

    #[test]
    fn test_provider_tool_scope_deserialize_new_format_without_provider() {
        let json = r#"{"model_name": "gpt-4"}"#;
        let scope: ProviderToolScope = serde_json::from_str(json).unwrap();
        assert_eq!(
            scope,
            ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
                model_name: "gpt-4".to_string(),
                provider_name: None,
            })
        );
    }

    #[test]
    fn test_provider_tool_scope_deserialize_old_format_backward_compat() {
        // Old format with model_provider_name should still work
        let json = r#"{"model_name": "gpt-4", "model_provider_name": "openai"}"#;
        let scope: ProviderToolScope = serde_json::from_str(json).unwrap();
        assert_eq!(
            scope,
            ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
                model_name: "gpt-4".to_string(),
                provider_name: Some("openai".to_string()),
            })
        );
    }

    #[test]
    fn test_provider_tool_scope_deserialize_null() {
        let json = "null";
        let scope: ProviderToolScope = serde_json::from_str(json).unwrap();
        assert_eq!(scope, ProviderToolScope::Unscoped);
    }

    #[test]
    fn test_provider_tool_scope_serialize_with_provider() {
        let scope = ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
            model_name: "gpt-4".to_string(),
            provider_name: Some("openai".to_string()),
        });
        let json = serde_json::to_string(&scope).unwrap();
        // Should serialize with provider_name (new format)
        assert_eq!(json, r#"{"model_name":"gpt-4","provider_name":"openai"}"#);
    }

    #[test]
    fn test_provider_tool_scope_serialize_without_provider() {
        let scope = ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
            model_name: "gpt-4".to_string(),
            provider_name: None,
        });
        let json = serde_json::to_string(&scope).unwrap();
        // Should serialize without provider_name field when None
        assert_eq!(json, r#"{"model_name":"gpt-4"}"#);
    }

    #[test]
    fn test_provider_tool_scope_serialize_unscoped() {
        let scope = ProviderToolScope::Unscoped;
        let json = serde_json::to_string(&scope).unwrap();
        assert_eq!(json, "null");
    }

    #[test]
    fn test_provider_tool_scope_matches_with_provider() {
        let scope = ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
            model_name: "gpt-4".to_string(),
            provider_name: Some("openai".to_string()),
        });
        assert!(scope.matches("gpt-4", "openai"));
        assert!(!scope.matches("gpt-4", "azure"));
        assert!(!scope.matches("claude-4", "openai"));
    }

    #[test]
    fn test_provider_tool_scope_matches_without_provider() {
        // When provider_name is None, should match any provider for the model
        let scope = ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
            model_name: "gpt-4".to_string(),
            provider_name: None,
        });
        assert!(scope.matches("gpt-4", "openai"));
        assert!(scope.matches("gpt-4", "azure"));
        assert!(scope.matches("gpt-4", "any-provider"));
        assert!(!scope.matches("claude-4", "anthropic"));
    }
}
