use crate::jsonschema_util::JSONSchemaFromPath;
use crate::tool::{StaticToolConfig, ToolCallConfig, ToolChoice, ToolConfig};
use lazy_static::lazy_static;
use serde_json::json;

lazy_static! {
    /// These are useful for tests which don't need mutable tools.
    static ref WEATHER_TOOL_CONFIG_STATIC: StaticToolConfig = StaticToolConfig {
        name: "get_weather".to_string(),
        description: "Get the current weather in a given location".to_string(),
        parameters: JSONSchemaFromPath::from_value(&json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        })),
        strict: false,
    };
    pub static ref WEATHER_TOOL: ToolConfig = ToolConfig::Static(&WEATHER_TOOL_CONFIG_STATIC);
    pub static ref WEATHER_TOOL_CHOICE: ToolChoice = ToolChoice::Tool("get_weather".to_string());
    pub static ref WEATHER_TOOL_CONFIG: ToolCallConfig = ToolCallConfig {
        tools_available: vec![ToolConfig::Static(&WEATHER_TOOL_CONFIG_STATIC)],
        tool_choice: ToolChoice::Tool("get_weather".to_string()),
        parallel_tool_calls: false,
    };
    static ref QUERY_TOOL_CONFIG: StaticToolConfig = StaticToolConfig {
        name: "query_articles".to_string(),
        description: "Query articles from Wikipedia".to_string(),
        parameters: JSONSchemaFromPath::from_value(&json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "year": {"type": "integer"}
            },
            "required": ["query", "year"]
        })),
        strict: true,
    };
    pub static ref QUERY_TOOL: ToolConfig = ToolConfig::Static(&QUERY_TOOL_CONFIG);
    pub static ref ANY_TOOL_CHOICE: ToolChoice = ToolChoice::Required;
    pub static ref MULTI_TOOL_CONFIG: ToolCallConfig = ToolCallConfig {
        tools_available: vec![
            ToolConfig::Static(&WEATHER_TOOL_CONFIG_STATIC),
            ToolConfig::Static(&QUERY_TOOL_CONFIG)
        ],
        tool_choice: ToolChoice::Required,
        parallel_tool_calls: true,
    };
}

// For use in tests which need a mutable tool config.
pub fn get_weather_tool_config() -> ToolCallConfig {
    let weather_tool = ToolConfig::Static(&WEATHER_TOOL_CONFIG_STATIC);
    ToolCallConfig {
        tools_available: vec![weather_tool],
        tool_choice: ToolChoice::Tool("get_weather".to_string()),
        parallel_tool_calls: false,
    }
}
