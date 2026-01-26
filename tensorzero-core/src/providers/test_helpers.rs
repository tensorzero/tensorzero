use std::sync::Arc;

use crate::jsonschema_util::JSONSchema;
use crate::tool::types::{ProviderTool, ProviderToolScope};
use crate::tool::{FunctionToolConfig, StaticToolConfig, ToolCallConfig, ToolChoice};
use lazy_static::lazy_static;
use serde_json::json;

lazy_static! {
    /// These are useful for tests which don't need mutable tools.
    pub static ref WEATHER_TOOL_CONFIG_STATIC: Arc<StaticToolConfig> = Arc::new(StaticToolConfig {
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
        })).unwrap(),
        strict: false,
    });
    pub static ref WEATHER_TOOL: FunctionToolConfig = FunctionToolConfig::Static(WEATHER_TOOL_CONFIG_STATIC.clone());
    pub static ref WEATHER_TOOL_CHOICE: ToolChoice = ToolChoice::Specific("get_temperature".to_string());
    pub static ref WEATHER_TOOL_CONFIG: ToolCallConfig = ToolCallConfig {
        tool_choice: ToolChoice::Specific("get_temperature".to_string()),
        ..ToolCallConfig::with_tools_available(
            vec![FunctionToolConfig::Static(WEATHER_TOOL_CONFIG_STATIC.clone())],
            vec![],
        )
    };
    pub static ref QUERY_TOOL_CONFIG_STATIC: Arc<StaticToolConfig> = Arc::new(StaticToolConfig {
        name: "query_articles".to_string(),
        key: "query_articles".to_string(),
        description: "Query articles from Wikipedia".to_string(),
        parameters: JSONSchema::from_value(json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "year": {"type": "integer"}
            },
            "required": ["query", "year"]
        })).unwrap(),
        strict: true,
    });
    pub static ref QUERY_TOOL: FunctionToolConfig = FunctionToolConfig::Static(QUERY_TOOL_CONFIG_STATIC.clone());
    pub static ref ANY_TOOL_CHOICE: ToolChoice = ToolChoice::Required;
    pub static ref MULTI_TOOL_CONFIG: ToolCallConfig = ToolCallConfig {
        tool_choice: ToolChoice::Required,
        parallel_tool_calls: Some(true),
        ..ToolCallConfig::with_tools_available(
            vec![
                FunctionToolConfig::Static(WEATHER_TOOL_CONFIG_STATIC.clone()),
                FunctionToolConfig::Static(QUERY_TOOL_CONFIG_STATIC.clone())
            ],
            vec![],
        )
    };
}

// For use in tests which need a mutable tool config.
pub fn get_temperature_tool_config() -> ToolCallConfig {
    let weather_tool = FunctionToolConfig::Static(WEATHER_TOOL_CONFIG_STATIC.clone());
    ToolCallConfig {
        tool_choice: ToolChoice::Specific("get_temperature".to_string()),
        parallel_tool_calls: Some(false),
        ..ToolCallConfig::with_tools_available(vec![weather_tool], vec![])
    }
}

/// Creates a tool config with only provider tools (no function tools).
/// This is useful for testing that provider-only configurations work correctly.
/// The model_name and provider_name parameters are accepted for API consistency but not used
/// since this creates an unscoped provider tool.
pub fn provider_only_tool_config(_model_name: &str, _provider_name: &str) -> ToolCallConfig {
    ToolCallConfig {
        tool_choice: ToolChoice::Auto,
        parallel_tool_calls: None,
        provider_tools: vec![ProviderTool {
            scope: ProviderToolScope::Unscoped,
            tool: json!({"type": "web_search"}),
        }],
        ..ToolCallConfig::with_tools_available(vec![], vec![])
    }
}

/// Creates a tool config with only scoped provider tools (no function tools).
/// The provider tools are scoped to the given model and provider.
pub fn scoped_provider_only_tool_config(model_name: &str, provider_name: &str) -> ToolCallConfig {
    use crate::tool::types::ProviderToolScopeModelProvider;

    ToolCallConfig {
        tool_choice: ToolChoice::Auto,
        parallel_tool_calls: None,
        provider_tools: vec![ProviderTool {
            scope: ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
                model_name: model_name.to_string(),
                provider_name: Some(provider_name.to_string()),
            }),
            tool: json!({"type": "google_search"}),
        }],
        ..ToolCallConfig::with_tools_available(vec![], vec![])
    }
}
