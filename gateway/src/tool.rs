use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    error::Error,
    jsonschema_util::{DynamicJSONSchema, JSONSchemaFromPath},
};

/// A Tool is a function that can be called by an LLM
/// We represent them in various ways depending on how they are configured by the user.
/// The primary difficulty is that tools require an input signature that we represent as a JSONSchema.
/// JSONSchema compilation takes time so we want to do it at startup if the tool is in the config.
/// We also don't want to clone compiled JSON schemas.
/// If the tool is dynamic we want to run compilation while LLM inference is happening so that we can validate the tool call arguments.
///
/// If we are doing an implicit tool call for JSON schema enforcement, we can use the compiled schema from the output signature.

/// A Tool object describes how a tool can be dynamically configured by the user.
#[derive(Debug, Deserialize, PartialEq, Serialize)]
pub struct Tool {
    pub description: String,
    pub parameters: Value,
    pub name: String,
}

#[derive(Debug, PartialEq)]
pub enum ToolConfig {
    Static(&'static StaticToolConfig),
    Dynamic(DynamicToolConfig),
}

/// Contains the configuration information for a specific tool
#[derive(Debug, PartialEq)]
pub struct StaticToolConfig {
    pub description: String,
    pub parameters: JSONSchemaFromPath,
    pub name: String,
}

#[derive(Debug, PartialEq)]
pub struct DynamicToolConfig {
    pub description: String,
    pub parameters: DynamicJSONSchema,
    pub name: String,
}

/// Contains all information required to tell an LLM what tools it can call
/// and what sorts of tool calls (parallel, none, etc) it is allowed to respond with.
/// Most inference providers can convert this into their desired tool format.
#[derive(Debug)]
pub struct ToolCallConfig {
    pub tools_available: Vec<ToolConfig>,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: bool,
}

/// ToolCallConfigDatabaseInsert is a lightweight version of ToolCallConfig that can be serialized and cloned.
/// It is used to insert the ToolCallConfig into the database.
#[derive(Debug, PartialEq, Serialize)]
pub struct ToolCallConfigDatabaseInsert {
    pub tools_available: Vec<Tool>,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: bool,
}

impl ToolCallConfig {
    pub fn new(
        function_tools: &'static [String],
        function_tool_choice: &'static ToolChoice,
        function_parallel_tool_calls: bool,
        static_tools: &'static HashMap<String, StaticToolConfig>,
        dynamic_tool_params: DynamicToolParams,
    ) -> Result<Option<Self>, Error> {
        // If `allowed_tools` is not provided, use the function's configured tools.
        // This means we allow all tools for the function.
        let allowed_tools = dynamic_tool_params
            .allowed_tools
            .as_deref()
            .unwrap_or(function_tools);

        // Get each tool from the static tool config.
        let tools_available: Result<Vec<ToolConfig>, Error> = allowed_tools
            .iter()
            .map(|tool_name| {
                static_tools
                    .get(tool_name)
                    .map(ToolConfig::Static)
                    .ok_or(Error::ToolNotFound {
                        name: tool_name.clone(),
                    })
            })
            .collect();

        // Throw an error if any tool was not found in the previous step.
        let mut tools_available = tools_available?;

        // Adds the additional tools to the list of available tools
        // (this kicks off async compilation in another thread for each)
        tools_available.extend(dynamic_tool_params.additional_tools.into_iter().flat_map(
            |tools| {
                tools.into_iter().map(|tool| {
                    ToolConfig::Dynamic(DynamicToolConfig {
                        description: tool.description,
                        parameters: DynamicJSONSchema::new(tool.parameters),
                        name: tool.name,
                    })
                })
            },
        ));
        let tool_choice = dynamic_tool_params
            .tool_choice
            .unwrap_or(function_tool_choice.clone());
        // If the tool choice is a specific tool, make sure it's in the list of available tools
        if let ToolChoice::Tool(tool_name) = &tool_choice {
            if !tools_available.iter().any(|tool| match tool {
                ToolConfig::Static(config) => config.name == *tool_name,
                ToolConfig::Dynamic(config) => config.name == *tool_name,
            }) {
                return Err(Error::ToolNotFound {
                    name: tool_name.clone(),
                });
            }
        }
        let parallel_tool_calls = dynamic_tool_params
            .parallel_tool_calls
            .unwrap_or(function_parallel_tool_calls);
        let tool_call_config_option = match tools_available.len() {
            0 => None,
            _ => Some(Self {
                tools_available,
                tool_choice,
                parallel_tool_calls,
            }),
        };
        Ok(tool_call_config_option)
    }

    pub fn get_tool(&self, name: &str) -> Option<&ToolConfig> {
        self.tools_available.iter().find(|tool_cfg| match tool_cfg {
            ToolConfig::Static(config) => config.name == name,
            ToolConfig::Dynamic(config) => config.name == name,
        })
    }
}

/// A struct to hold the dynamic tool parameters passed at inference time.
/// These should override the function-level tool parameters.
/// `allowed_tools` should be a subset of the configured tools for the function.
/// if `allowed_tools` is not provided, all tools are allowed.
/// `additional_tools` are the tools that are provided at runtime, which we compile on the fly.
/// `tool_choice` and `parallel_tool_calls` are optional and will override the function-level values.
#[derive(Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(test, derive(Default))]
pub struct DynamicToolParams {
    pub allowed_tools: Option<Vec<String>>,
    pub additional_tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    pub parallel_tool_calls: Option<bool>,
}

/// A ToolCall is a request by a model to call a Tool
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: String,
    pub id: String,
}

/// A ToolCallOutput is a request by a model to call a Tool
/// in the form that we return to the client / ClickHouse
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCallOutput {
    pub name: String,
    pub arguments: String,
    pub id: String,
    pub parsed_name: Option<String>,
    pub parsed_arguments: Option<Value>,
}

impl ToolCallOutput {
    /// Validates that a ToolCall is compliant with the ToolCallConfig
    /// First, it finds the ToolConfig for the ToolCall
    /// Then, it validates the ToolCall arguments against the ToolConfig
    pub async fn new(tool_call: ToolCall, tool_cfg: Option<&ToolCallConfig>) -> Self {
        let tool = tool_cfg.and_then(|t| t.get_tool(&tool_call.name));
        let parsed_name = match tool {
            Some(_) => Some(tool_call.name.clone()),
            None => None,
        };
        let parsed_arguments = match &tool {
            Some(tool) => {
                if let Ok(arguments) = serde_json::from_str(&tool_call.arguments) {
                    if tool.validate_arguments(&arguments).await.is_ok() {
                        Some(arguments)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            None => None,
        };
        Self {
            name: tool_call.name.clone(),
            arguments: tool_call.arguments.clone(),
            id: tool_call.id,
            parsed_name,
            parsed_arguments,
        }
    }
}

/// A ToolResult is the outcome of a ToolCall, which we may want to present back to the model
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub name: String,
    pub result: String,
    pub id: String,
}

/// Most inference providers allow the user to force a tool to be used
/// and even specify which tool to be used.
///
/// This enum is used to denote this tool choice.
#[derive(Clone, Debug, PartialEq, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    None,
    #[default]
    Auto,
    Required,
    Tool(String), // Forces the LLM to call a particular tool, the String is the name of the Tool
    Implicit, // It is occasionally helpful to make an "implicit" tool call to enforce that a JSON schema is followed
              // In this case, the tool call is not exposed to the client, but the output is still validated against the schema
              // Implicit means that the tool will always be called "respond" and that we should convert it back to chat-style output
              // before response.
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToolCallChunk {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

impl ToolConfig {
    pub async fn validate_arguments(&self, arguments: &Value) -> Result<(), Error> {
        match self {
            ToolConfig::Static(config) => config.parameters.validate(arguments),
            ToolConfig::Dynamic(config) => config.parameters.validate(arguments).await,
        }
    }

    pub fn description(&self) -> &str {
        match self {
            ToolConfig::Static(config) => &config.description,
            ToolConfig::Dynamic(config) => &config.description,
        }
    }

    pub fn parameters(&self) -> &Value {
        match self {
            ToolConfig::Static(config) => config.parameters.value,
            ToolConfig::Dynamic(config) => &config.parameters.value,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            ToolConfig::Static(config) => &config.name,
            ToolConfig::Dynamic(config) => &config.name,
        }
    }
}

impl From<ToolCallConfig> for ToolCallConfigDatabaseInsert {
    fn from(tool_call_config: ToolCallConfig) -> Self {
        Self {
            tools_available: tool_call_config
                .tools_available
                .into_iter()
                .map(|tool| tool.into())
                .collect(),
            tool_choice: tool_call_config.tool_choice,
            parallel_tool_calls: tool_call_config.parallel_tool_calls,
        }
    }
}

impl From<ToolConfig> for Tool {
    fn from(tool_config: ToolConfig) -> Self {
        Self {
            description: tool_config.description().to_string(),
            parameters: tool_config.parameters().clone(),
            name: tool_config.name().to_string(),
        }
    }
}

mod tests {
    use super::*;
    use lazy_static::lazy_static;
    use serde_json::json;

    lazy_static! {
        static ref TOOLS: HashMap<String, StaticToolConfig> = {
            let mut map = HashMap::new();
            map.insert(
                "get_weather".to_string(),
                StaticToolConfig {
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
                },
            );
            map.insert(
                "query_articles".to_string(),
                StaticToolConfig {
                    name: "query_articles".to_string(),
                    description: "Query articles from a database based on given criteria"
                        .to_string(),
                    parameters: JSONSchemaFromPath::from_value(&json!({
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string"},
                            "category": {"type": "string"},
                            "limit": {"type": "integer", "minimum": 1, "maximum": 100}
                        },
                        "required": ["keyword"]
                    })),
                },
            );
            map
        };
        static ref EMPTY_TOOLS: HashMap<String, StaticToolConfig> = HashMap::new();
        static ref EMPTY_FUNCTION_TOOLS: Vec<String> = vec![];
        static ref ALL_FUNCTION_TOOLS: Vec<String> =
            vec!["get_weather".to_string(), "query_articles".to_string()];
        static ref AUTO_TOOL_CHOICE: ToolChoice = ToolChoice::Auto;
        static ref WEATHER_TOOL_CHOICE: ToolChoice = ToolChoice::Tool("get_weather".to_string());
    }

    #[tokio::test]
    async fn test_tool_call_config_new() {
        // Empty tools in function, no dynamic tools, tools are configured in the config
        // This should return no tools because the function does not specify any tools
        let tool_call_config = ToolCallConfig::new(
            &EMPTY_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            true,
            &TOOLS,
            DynamicToolParams::default(),
        )
        .unwrap();
        assert!(tool_call_config.is_none());

        // All tools available, no dynamic tools, tools are configured in the config
        // This should return all tools because the function specifies all tools
        let tool_call_config = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            true,
            &TOOLS,
            DynamicToolParams::default(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(tool_call_config.tools_available.len(), 2);
        assert_eq!(tool_call_config.tool_choice, ToolChoice::Auto);
        assert!(tool_call_config.parallel_tool_calls);

        // Empty tools in function and config but we specify an allowed tool (should fail)
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec!["get_weather".to_string()]),
            ..Default::default()
        };
        let err = ToolCallConfig::new(
            &EMPTY_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            true,
            &EMPTY_TOOLS,
            dynamic_tool_params,
        )
        .unwrap_err();
        assert_eq!(
            err,
            Error::ToolNotFound {
                name: "get_weather".to_string()
            }
        );

        // Dynamic tool config specifies a particular tool to call and it's in the function tools list
        let dynamic_tool_params = DynamicToolParams {
            tool_choice: Some(ToolChoice::Tool("get_weather".to_string())),
            ..Default::default()
        };
        let tool_call_config = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            true,
            &TOOLS,
            dynamic_tool_params,
        )
        .unwrap()
        .unwrap();
        assert_eq!(tool_call_config.tools_available.len(), 2);
        assert_eq!(
            tool_call_config.tool_choice,
            ToolChoice::Tool("get_weather".to_string())
        );
        assert!(tool_call_config.parallel_tool_calls);

        // Dynamic tool config specifies a particular tool to call and it's not in the function tools list
        let dynamic_tool_params = DynamicToolParams {
            tool_choice: Some(ToolChoice::Tool("establish_campground".to_string())),
            ..Default::default()
        };
        let err = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            true,
            &TOOLS,
            dynamic_tool_params,
        )
        .unwrap_err();
        assert_eq!(
            err,
            Error::ToolNotFound {
                name: "establish_campground".to_string()
            }
        );

        // We pass an empty list of allowed tools and then configure a new tool
        // This should remove all configured tools and add the new tool
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec![]),
            additional_tools: Some(vec![Tool {
                name: "establish_campground".to_string(),
                description: "Establish a campground".to_string(),
                parameters: json!({}),
            }]),
            ..Default::default()
        };
        let tool_call_config = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            true,
            &TOOLS,
            dynamic_tool_params,
        )
        .unwrap()
        .unwrap();
        assert_eq!(tool_call_config.tools_available.len(), 1);
        let first_tool = tool_call_config.tools_available.first().unwrap();
        assert_eq!(first_tool.name(), "establish_campground");

        // We pass a list of a single allowed tool and then configure a new tool
        // This should remove the other configured tools and add the new tool
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec!["get_weather".to_string()]),
            additional_tools: Some(vec![Tool {
                name: "establish_campground".to_string(),
                description: "Establish a campground".to_string(),
                parameters: json!({}),
            }]),
            parallel_tool_calls: Some(false),
            ..Default::default()
        };
        let tool_call_config = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            true,
            &TOOLS,
            dynamic_tool_params,
        )
        .unwrap()
        .unwrap();
        assert_eq!(tool_call_config.tools_available.len(), 2);
        // The following code depends on an implementation detail for this ordering,
        // might break if we change the order
        assert_eq!(tool_call_config.tools_available[0].name(), "get_weather");
        assert_eq!(
            tool_call_config.tools_available[1].name(),
            "establish_campground"
        );
        assert!(!tool_call_config.parallel_tool_calls);

        // We pass a list of no allowed tools and then configure a new tool
        // This should remove all configured tools and add the new tool
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec![]),
            additional_tools: Some(vec![Tool {
                name: "establish_campground".to_string(),
                description: "Establish a campground".to_string(),
                parameters: json!({}),
            }]),
            tool_choice: Some(ToolChoice::Tool("establish_campground".to_string())),
            ..Default::default()
        };
        let tool_call_config = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            true,
            &TOOLS,
            dynamic_tool_params,
        )
        .unwrap()
        .unwrap();
        assert_eq!(tool_call_config.tools_available.len(), 1);
        assert_eq!(
            tool_call_config.tools_available[0].name(),
            "establish_campground"
        );
        assert!(tool_call_config.parallel_tool_calls);
        assert_eq!(
            tool_call_config.tool_choice,
            ToolChoice::Tool("establish_campground".to_string())
        );
    }

    #[tokio::test]
    async fn test_tool_call_output_new() {
        let tool_call = ToolCall {
            name: "get_weather".to_string(),
            arguments: "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}".to_string(),
            id: "123".to_string(),
        };
        let tool_call_config = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            true,
            &TOOLS,
            DynamicToolParams::default(),
        )
        .unwrap()
        .unwrap();
        // Tool call is valid, so we should get a valid ToolCallOutput
        let tool_call_output = ToolCallOutput::new(tool_call, Some(&tool_call_config)).await;
        assert_eq!(tool_call_output.name, "get_weather");
        assert_eq!(
            tool_call_output.arguments,
            "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}"
        );
        assert_eq!(tool_call_output.id, "123");
        assert_eq!(
            tool_call_output.parsed_name,
            Some("get_weather".to_string())
        );
        assert_eq!(
            tool_call_output.parsed_arguments,
            Some(json!({
                "location": "San Francisco",
                "unit": "celsius"
            }))
        );

        // Bad arguments, but valid name (parsed_name is set but parsed_arguments is not)
        let tool_call = ToolCall {
            name: "get_weather".to_string(),
            arguments: "{\"location\": \"San Francisco\", \"unit\": \"kelvin\"}".to_string(),
            id: "321".to_string(),
        };
        let tool_call_output = ToolCallOutput::new(tool_call, Some(&tool_call_config)).await;
        assert_eq!(
            tool_call_output.parsed_name,
            Some("get_weather".to_string())
        );
        assert_eq!(tool_call_output.parsed_arguments, None);
        assert_eq!(tool_call_output.id, "321");
        assert_eq!(tool_call_output.name, "get_weather");
        assert_eq!(
            tool_call_output.arguments,
            "{\"location\": \"San Francisco\", \"unit\": \"kelvin\"}"
        );

        // Bad name, good arguments (both not set since the name is invalid and we can't be sure what tool this goes to)
        let tool_call = ToolCall {
            name: "get_wether".to_string(),
            arguments: "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}".to_string(),
            id: "321".to_string(),
        };
        let tool_call_output = ToolCallOutput::new(tool_call, Some(&tool_call_config)).await;
        assert_eq!(tool_call_output.parsed_name, None);
        assert_eq!(tool_call_output.parsed_arguments, None);
        assert_eq!(tool_call_output.id, "321");
        assert_eq!(tool_call_output.name, "get_wether");
        assert_eq!(
            tool_call_output.arguments,
            "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}"
        );

        // Make sure validation works with dynamic tools
        let tool_call_config = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            true,
            &TOOLS,
            DynamicToolParams {
                additional_tools: Some(vec![Tool {
                    name: "establish_campground".to_string(),
                    description: "Establish a campground".to_string(),
                    parameters: json!({"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}),
                }]),
                ..Default::default()
            },
        )
        .unwrap()
        .unwrap();
        let tool_call = ToolCall {
            name: "establish_campground".to_string(),
            arguments: "{\"location\": \"Lucky Dog\"}".to_string(),
            id: "321".to_string(),
        };
        let tool_call_output = ToolCallOutput::new(tool_call, Some(&tool_call_config)).await;
        assert_eq!(tool_call_output.name, "establish_campground");
        assert_eq!(tool_call_output.arguments, "{\"location\": \"Lucky Dog\"}");
        assert_eq!(tool_call_output.id, "321");
        assert_eq!(
            tool_call_output.parsed_name,
            Some("establish_campground".to_string())
        );
        assert_eq!(
            tool_call_output.parsed_arguments,
            Some(json!({"location": "Lucky Dog"}))
        );
    }
}
