use std::sync::Arc;

use crate::jsonschema_util::StaticJSONSchema;
use crate::tool::{FunctionToolConfig, StaticToolConfig, ToolCallConfig, ToolChoice};
use lazy_static::lazy_static;
use serde_json::json;

lazy_static! {
    /// These are useful for tests which don't need mutable tools.
    pub static ref WEATHER_TOOL_CONFIG_STATIC: Arc<StaticToolConfig> = Arc::new(StaticToolConfig {
        name: "get_temperature".to_string(),
        description: "Get the current temperature in a given location".to_string(),
        parameters: StaticJSONSchema::from_value(json!({
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
        description: "Query articles from Wikipedia".to_string(),
        parameters: StaticJSONSchema::from_value(json!({
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
