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
    Static(&'static OwnedToolConfig),
    Dynamic(DynamicToolConfig),
}

/// Contains the configuration information for a specific tool
#[derive(Debug, PartialEq)]
pub struct OwnedToolConfig {
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
        static_tools: &'static HashMap<String, OwnedToolConfig>,
        dynamic_tool_params: DynamicToolParams,
    ) -> Result<Self, Error> {
        let tool_names = dynamic_tool_params
            .allowed_tools
            .as_deref()
            .unwrap_or(function_tools);

        let tools_available: Result<Vec<ToolConfig>, Error> = tool_names
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

        let tool_choice = match dynamic_tool_params.tool_choice {
            Some(tool_choice) => tool_choice,
            None => function_tool_choice.clone(),
        };
        let parallel_tool_calls = dynamic_tool_params
            .parallel_tool_calls
            .unwrap_or(function_parallel_tool_calls);

        Ok(Self {
            tools_available,
            tool_choice,
            parallel_tool_calls,
        })
    }

    pub fn get_tool(&mut self, name: &str) -> Option<&mut ToolConfig> {
        self.tools_available
            .iter_mut()
            .find(|tool_cfg| match tool_cfg {
                ToolConfig::Static(config) => config.name == name,
                ToolConfig::Dynamic(config) => config.name == name,
            })
    }
}

/// Anticipates #126
#[derive(Debug, PartialEq, Serialize)]
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
    pub async fn new(tool_call: ToolCall, tool_cfg: Option<&mut ToolCallConfig>) -> Self {
        let mut tool = tool_cfg.and_then(|t| t.get_tool(&tool_call.name));
        let parsed_name = match tool {
            Some(_) => Some(tool_call.name.clone()),
            None => None,
        };
        let parsed_arguments = match &mut tool {
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
    pub async fn validate_arguments(&mut self, arguments: &Value) -> Result<(), Error> {
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
