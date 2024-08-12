use std::{collections::HashMap, path::Path};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{error::Error, jsonschema_util::JSONSchemaFromPath};

/// Contains the configuration information for a specific tool
#[derive(Clone, Debug, Deserialize)]
pub struct ToolConfig {
    pub description: String,
    pub parameters: JSONSchemaFromPath,
    // This should get set at `load` time
    #[serde(default)]
    pub tool: Option<Tool>,
}

/// Contains all information required to tell an LLM what tools it can call
/// and what sorts of tool calls (parallel, none, etc) it is allowed to respond with.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCallConfig<'a> {
    pub tools_available: Vec<&'a Tool>,
    pub tool_choice: &'a ToolChoice,
    pub parallel_tool_calls: bool,
}

impl<'a> ToolCallConfig<'a> {
    pub fn new(
        function_tools: &'a Vec<String>,
        function_tool_choice: &'a ToolChoice,
        function_parallel_tool_calls: bool,
        static_tools: &'a HashMap<String, ToolConfig>,
        dynamic_tool_config: &'a DynamicToolConfig,
    ) -> Result<Self, Error> {
        let allowed_tool_names = match dynamic_tool_config.allowed_tools {
            Some(allowed_tools) => allowed_tools,
            None => function_tools,
        };
        let tools_available: Result<Vec<&Tool>, Error> = allowed_tool_names
            .iter()
            .map(|tool_name| {
                static_tools
                    .get(tool_name)
                    .ok_or(Error::ToolNotFound {
                        name: tool_name.clone(),
                    })?
                    .tool
                    .as_ref()
                    .ok_or(Error::ToolNotLoaded {
                        name: tool_name.clone(),
                    })
            })
            .collect();
        let mut tools_available = tools_available?;
        if let Some(additional_tools) = dynamic_tool_config.additional_tools {
            tools_available.extend(additional_tools);
        }
        let tool_choice = dynamic_tool_config
            .tool_choice
            .unwrap_or(function_tool_choice);
        let parallel_tool_calls = dynamic_tool_config
            .parallel_tool_calls
            .unwrap_or(function_parallel_tool_calls);

        Ok(Self {
            tools_available,
            tool_choice,
            parallel_tool_calls,
        })
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

impl ToolConfig {
    pub fn load<P: AsRef<Path>>(&mut self, name: String, base_path: P) -> Result<(), Error> {
        self.parameters.load(base_path)?;
        let parameters = self.parameters.value()?.clone();
        self.tool = Some(Tool::Function {
            name,
            parameters,
            description: Some(self.description.clone()),
        });
        Ok(())
    }
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
