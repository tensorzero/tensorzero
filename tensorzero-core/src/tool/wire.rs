//! Wire format types for tool calls and results.
//!
//! This module contains the types used in API requests and responses for tool interactions:
//! - `ToolCall` - A request by an LLM to call a tool
//! - `ToolResult` - The response from a tool call
//! - `ToolChoice` - Strategy for how the LLM should choose tools

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tensorzero_derive::export_schema;

use crate::error::Error;
use crate::rate_limiting::{RateLimitedInputContent, get_estimated_tokens};

use super::call::InferenceResponseToolCall;

/// In most cases, tool call arguments are a string.
/// However, when looping back from an inference response, they will be an object.
fn deserialize_tool_call_arguments<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    let value = Value::deserialize(deserializer)?;
    match value {
        Value::String(s) => Ok(s),
        Value::Object(_) => Ok(value.to_string()),
        _ => Err(D::Error::custom(
            "`arguments` must be a string or an object",
        )),
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS, JsonSchema)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(get_all, str))]
#[export_schema]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    #[serde(deserialize_with = "deserialize_tool_call_arguments")] // String or Object --> String
    pub arguments: String,
}

impl std::fmt::Display for ToolCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl RateLimitedInputContent for ToolCall {
    fn estimated_input_token_usage(&self) -> u64 {
        let ToolCall {
            name,
            arguments,
            #[expect(unused_variables)]
            id,
        } = self;
        get_estimated_tokens(name) + get_estimated_tokens(arguments)
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl ToolCall {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}

/// `ToolCallWrapper` helps us disambiguate between `ToolCall` (no `raw_*`) and `InferenceResponseToolCall` (has `raw_*`).
/// Typically tool calls come from previous inferences and are therefore outputs of TensorZero (`InferenceResponseToolCall`)
/// but they may also be constructed client side or through the OpenAI endpoint `ToolCall` so we support both via this wrapper.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS, JsonSchema)]
#[ts(export)]
#[schemars(description = "")]
#[serde(untagged)]
pub enum ToolCallWrapper {
    // The format we store in the database
    #[schemars(title = "ContentBlockInputToolCall")]
    ToolCall(ToolCall),

    // The format we send on an inference response, with parsed name and arguments
    #[schemars(title = "ContentBlockValidatedToolCall")]
    InferenceResponseToolCall(InferenceResponseToolCall),
}

#[derive(JsonSchema)]
pub(crate) struct ToolCallWrapperJsonSchema {
    #[serde(flatten)]
    _tool_call_wrapper: ToolCallWrapper,
}

/// - ToolCallWrapper::ToolCall: passthrough
/// - ToolCallWrapper::InferenceResponseToolCall: this is an inference loopback --> use raw values, ignore parsed values
impl TryFrom<ToolCallWrapper> for ToolCall {
    type Error = Error;
    fn try_from(wrapper: ToolCallWrapper) -> Result<Self, Self::Error> {
        match wrapper {
            ToolCallWrapper::ToolCall(tc) => Ok(tc),
            ToolCallWrapper::InferenceResponseToolCall(tc) => Ok(ToolCall {
                id: tc.id,
                name: tc.raw_name,
                arguments: tc.raw_arguments,
            }),
        }
    }
}

/// A ToolResult is the outcome of a ToolCall, which we may want to present back to the model
#[cfg_attr(feature = "pyo3", pyclass(get_all, str))]
#[derive(ts_rs::TS, Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
#[ts(export)]
#[serde(deny_unknown_fields)]
#[export_schema]
pub struct ToolResult {
    pub name: String,
    pub result: String,
    pub id: String,
}

impl std::fmt::Display for ToolResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl RateLimitedInputContent for ToolResult {
    fn estimated_input_token_usage(&self) -> u64 {
        let ToolResult {
            name,
            result,
            #[expect(unused_variables)]
            id,
        } = self;
        get_estimated_tokens(name) + get_estimated_tokens(result)
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl ToolResult {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}

/// Most inference providers allow the user to force a tool to be used
/// and even specify which tool to be used.
///
/// This enum is used to denote this tool choice.
#[derive(ts_rs::TS, Clone, Debug, Default, Deserialize, PartialEq, Serialize, JsonSchema)]
#[ts(export)]
#[serde(rename_all = "lowercase")]
#[serde(deny_unknown_fields)]
#[export_schema]
pub enum ToolChoice {
    None,
    #[default]
    Auto,
    Required,
    /// Forces the LLM to call a specific tool. The String is the name of the tool.
    #[schemars(title = "ToolChoiceSpecific")]
    Specific(String),
}
