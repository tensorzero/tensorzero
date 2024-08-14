use crate::jsonschema_util::JSONSchemaFromPath;
use crate::tool::{ToolCallConfig, ToolChoice, ToolConfig};
use lazy_static::lazy_static;
use serde_json::json;

lazy_static! {
    pub static ref WEATHER_TOOL: ToolConfig = ToolConfig {
        name: "get_weather".to_string(),
        description: "Get the current weather in a given location".to_string(),
        parameters: JSONSchemaFromPath::from_value(&json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }))
    };
    pub static ref WEATHER_TOOL_CHOICE: ToolChoice = ToolChoice::Tool("get_weather".to_string());
    pub static ref WEATHER_TOOL_CONFIG: ToolCallConfig = ToolCallConfig {
        tools_available: vec![&*WEATHER_TOOL],
        tool_choice: &WEATHER_TOOL_CHOICE,
        parallel_tool_calls: false,
    };
    pub static ref QUERY_TOOL: ToolConfig = ToolConfig {
        name: "query_articles".to_string(),
        description: "Query articles from Wikipedia".to_string(),
        parameters: JSONSchemaFromPath::from_value(&json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "year": {"type": "integer"},
            },
            "required": ["query", "date"]
        }))
    };
    pub static ref ANY_TOOL_CHOICE: ToolChoice = ToolChoice::Required;
    pub static ref MULTI_TOOL_CONFIG: ToolCallConfig = ToolCallConfig {
        tools_available: vec![&*WEATHER_TOOL, &*QUERY_TOOL],
        tool_choice: &ANY_TOOL_CHOICE,
        parallel_tool_calls: true,
    };
}
