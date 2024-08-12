use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{error::Error, jsonschema_util::JSONSchemaFromPath};

#[derive(Clone, Debug, Deserialize)]
pub struct ToolConfig {
    pub description: String,
    pub parameters: JSONSchemaFromPath,
}

impl ToolConfig {
    pub fn load<P: AsRef<Path>>(&mut self, base_path: P) -> Result<(), Error> {
        self.parameters.load(base_path)
    }
}

/// The Tool type is used to represent a tool that is available to the model.
#[derive(Clone, Debug, PartialEq)]
pub enum Tool {
    Function {
        description: Option<String>,
        name: String,
        parameters: Value,
    },
}

/// A ToolCall is a request by a model to call a Tool
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: String,
    pub id: String,
}

/// A ToolResult is the outcome of a ToolCall, which we may want to present back to the model
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToolResult {
    pub name: String,
    pub result: String,
    pub id: String,
}

/// Most inference providers allow the user to force a tool to be used
/// and even specify which tool to be used.
///
/// This enum is used to denote this tool choice.
#[derive(Clone, Debug, PartialEq, Default, Deserialize)]
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
