use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{error::Error, jsonschema_util::JSONSchemaFromPath};

/// Contains the configuration information for a specific tool
#[derive(Debug, PartialEq)]
pub struct ToolConfig {
    pub description: String,
    pub parameters: JSONSchemaFromPath,
    // This should get set at `load` time
    pub tool: Tool,
}

/// Contains all information required to tell an LLM what tools it can call
/// and what sorts of tool calls (parallel, none, etc) it is allowed to respond with.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCallConfig {
    pub tools_available: Vec<&'static ToolConfig>,
    pub tool_choice: &'static ToolChoice,
    pub parallel_tool_calls: bool,
}

impl ToolCallConfig {
    pub fn new(
        function_tools: &'static Vec<String>,
        function_tool_choice: &'static ToolChoice,
        function_parallel_tool_calls: bool,
        static_tools: &'static HashMap<String, ToolConfig>,
        // _dynamic_tool_config: &'a DynamicToolConfig,
    ) -> Result<Self, Error> {
        // TODO (add issue): support dynamic tool calling properly
        // let allowed_tool_names = match dynamic_tool_config.allowed_tools {
        //     Some(allowed_tools) => allowed_tools,
        //     None => function_tools,
        // };
        let allowed_tool_names = function_tools;
        let tools_available: Result<Vec<&ToolConfig>, Error> = allowed_tool_names
            .iter()
            .map(|tool_name| {
                static_tools.get(tool_name).ok_or(Error::ToolNotFound {
                    name: tool_name.clone(),
                })
            })
            .collect();
        let mut tools_available = tools_available?;
        // if let Some(additional_tools) = dynamic_tool_config.additional_tools {
        //     tools_available.extend(additional_tools);
        // }
        // let tool_choice = dynamic_tool_config
        //     .tool_choice
        //     .unwrap_or(function_tool_choice);
        let tool_choice = function_tool_choice;
        // let parallel_tool_calls = dynamic_tool_config
        //     .parallel_tool_calls
        //     .unwrap_or(function_parallel_tool_calls);
        let parallel_tool_calls = function_parallel_tool_calls;

        Ok(Self {
            tools_available,
            tool_choice,
            parallel_tool_calls,
        })
    }

    pub fn get_tool(&self, name: &str) -> Option<&ToolConfig> {
        self.tools_available
            .iter()
            .find(|tool_cfg| matches!(tool_cfg.tool, Tool::Function { name: n, .. } if n == name))
            .copied()
    }
}

// TODO(viraj): store this in ClickHouse
#[derive(Debug, PartialEq, Serialize)]
pub struct DynamicToolConfig<'a> {
    pub allowed_tools: Option<&'a Vec<String>>,
    pub additional_tools: Option<&'a Vec<Tool>>,
    pub tool_choice: Option<&'a ToolChoice>,
    pub parallel_tool_calls: Option<bool>,
}

/// The Tool type is used to represent a tool that is available to the model.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub enum Tool {
    Function {
        // TODO(maybe, otherwise remove this): make these references somehow
        // The difficulty here is that Tools are sometimes deserialized from input (need to be owned)
        // or they are constructed from ToolConfigs in schema validation (need to be references)
        description: Option<String>,
        name: String,
        parameters: Value,
    },
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
    pub fn new(tool_call: ToolCall, tool_cfg: Option<&ToolConfig>) -> Self {
        let mut tool_call_output = Self {
            name: tool_call.name,
            arguments: tool_call.arguments,
            id: tool_call.id,
            parsed_name: None,
            parsed_arguments: None,
        };
        if let Some(tool_cfg) = tool_cfg {
            tool_call_output.parsed_name = Some(tool_call.name.clone());
            if let Ok(arguments) = serde_json::from_str(&tool_call.arguments) {
                // validate parameters
                if tool_cfg.parameters.validate(&arguments).is_ok() {
                    tool_call_output.parsed_arguments = Some(arguments);
                }
            }
        }
        tool_call_output
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
