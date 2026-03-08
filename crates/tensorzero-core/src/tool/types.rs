//! Core tool type definitions.
//!
//! This module contains the fundamental types that represent tools in TensorZero:
//! - `Tool` - The main enum representing all tool types
//! - `FunctionTool` - Client-side function tools with JSON schema parameters
//! - `OpenAICustomTool` - OpenAI's custom tool format (text/grammar)
//! - `ProviderTool` - Provider-specific tool configurations

use std::fmt;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use strum::AsRefStr;
use tensorzero_derive::export_schema;

#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::serialize_to_dict;
use crate::jsonschema_util::JSONSchema;

use super::config::DynamicToolConfig;

/// `Tool` is the generic form for all tools that TensorZero itself manages.
/// This includes function tools (the original kind) and OpenAI's custom tools
/// (which support text and grammar formats). Future additions may include MCP and other standards.
///
/// We store this type (serialized) in the Array(String) in the `dynamic_tools` column
/// in the ChatInference, ChatInferenceDatapoint, and BatchModelInference tables.
///
/// For the wire format, we use `DynamicTool` which wraps this enum with a custom deserializer
/// that allows function tools to be specified without type tags for backward compatibility,
/// while other tool types require explicit tagging.
///
/// Notably, provider tools (like OpenAI websearch) are not part of this enum
/// as there's not really anything we can do besides experiment with them.
/// They are a separate type `ProviderTool`.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(AsRefStr, Clone, Debug, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[strum(serialize_all = "snake_case")]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub enum Tool {
    #[schemars(title = "FunctionTool")]
    Function(FunctionTool), // Custom deserializer below accepts no type or type="client_side_function" (legacy)
    #[schemars(title = "OpenAICustomTool")]
    #[serde(rename = "openai_custom")]
    OpenAICustom(OpenAICustomTool),
}

impl std::fmt::Display for Tool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

/// Custom deserializer for Tool that provides backward compatibility.
/// If the type tag is present, deserialize normally. If missing, assume Function.
///
/// Additionally, accept `"client_side_function"` as an alias for `"function"`.
/// We've stored the former in the database, so we can't remove this alias.
impl<'de> Deserialize<'de> for Tool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(tag = "type", rename_all = "snake_case")]
        enum TaggedTool {
            #[serde(rename = "function", alias = "client_side_function")]
            Function(FunctionTool),
            #[serde(rename = "openai_custom")]
            OpenAICustom(OpenAICustomTool),
        }

        let value = serde_json::Value::deserialize(deserializer)?;

        // First, try to deserialize as a tagged Tool (new format)
        if let Ok(tagged) = serde_json::from_value::<TaggedTool>(value.clone()) {
            return Ok(match tagged {
                TaggedTool::Function(tool) => Tool::Function(tool),
                TaggedTool::OpenAICustom(tool) => Tool::OpenAICustom(tool),
            });
        }

        // Fall back to untagged FunctionTool format (legacy backward compatibility)
        match serde_json::from_value::<FunctionTool>(value) {
            Ok(function_tool) => Ok(Tool::Function(function_tool)),
            Err(e) => Err(serde::de::Error::custom(format!(
                "Failed to parse as `Tool` (tagged) or `FunctionTool` (untagged): {e}"
            ))),
        }
    }
}

impl Tool {
    pub(crate) fn name(&self) -> &str {
        match self {
            Tool::Function(tool) => &tool.name,
            Tool::OpenAICustom(tool) => &tool.name,
        }
    }

    /// Returns true if this is a custom tool (not a function tool)
    pub fn is_custom(&self) -> bool {
        matches!(self, Tool::OpenAICustom(_))
    }

    /// Returns true if this is a function tool
    pub fn is_function(&self) -> bool {
        matches!(self, Tool::Function(_))
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl Tool {
    /*
     * Note: as we add more tool types, we can throw AttributeError on fields that they don't have
     * and ask the caller to check the type field.
     * This avoids a breaking change to the Python interface as we go from a single tool type to potentially more in the future.
     * most notably, MCP
     */
    #[getter]
    pub fn get_type(&self) -> &str {
        self.as_ref()
    }

    #[getter]
    pub fn get_parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Tool::Function(tool) => {
                serialize_to_dict(py, tool.parameters.clone()).map(|x| x.into_bound(py))
            }
            Tool::OpenAICustom(_) => Err(pyo3::exceptions::PyAttributeError::new_err(
                "Custom tools do not have parameters. Check type field first.",
            )),
        }
    }

    #[getter]
    pub fn get_description(&self) -> PyResult<String> {
        match self {
            Tool::Function(tool) => Ok(tool.description.clone()),
            Tool::OpenAICustom(tool) => tool.description.clone().ok_or_else(|| {
                pyo3::exceptions::PyAttributeError::new_err("This custom tool has no description")
            }),
        }
    }

    #[getter]
    pub fn get_name(&self) -> &str {
        match self {
            Tool::Function(tool) => &tool.name,
            Tool::OpenAICustom(tool) => &tool.name,
        }
    }

    #[getter]
    pub fn get_strict(&self) -> PyResult<bool> {
        match self {
            Tool::Function(tool) => Ok(tool.strict),
            Tool::OpenAICustom(_) => Err(pyo3::exceptions::PyAttributeError::new_err(
                "Custom tools do not have strict mode. Check type field first.",
            )),
        }
    }

    #[getter]
    pub fn get_format<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Tool::OpenAICustom(tool) => match &tool.format {
                Some(format) => serialize_to_dict(py, format.clone()).map(|x| x.into_bound(py)),
                None => Ok(py.None().into_bound(py)),
            },
            Tool::Function(_) => Err(pyo3::exceptions::PyAttributeError::new_err(
                "Function tools do not have format. Check type field first.",
            )),
        }
    }

    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}

/// `FunctionTool` is a particular kind of tool that relies
/// on the client to execute a function on their side (a ToolCall content block)
/// and return the result on the next turn (a ToolCallResult).
/// Notably, we assume there is a JSON schema `parameters` that specifies the
/// set of arguments that the tool will accept.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct FunctionTool {
    pub description: String,
    pub parameters: Value,
    pub name: String,
    /// `strict` here specifies that TensorZero should attempt to use any facilities
    /// available from the model provider to force the model to generate an accurate tool call,
    /// notably OpenAI's strict tool call mode (https://platform.openai.com/docs/guides/function-calling#strict-mode).
    /// This imposes additional restrictions on the JSON schema that may vary across providers
    /// so we allow it to be configurable.
    #[serde(default)]
    pub strict: bool,
}

impl fmt::Display for FunctionTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl FunctionTool {
    #[getter]
    pub fn get_parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        serialize_to_dict(py, self.parameters.clone()).map(|x| x.into_bound(py))
    }

    #[getter]
    pub fn get_description(&self) -> &str {
        &self.description
    }

    #[getter]
    pub fn get_name(&self) -> &str {
        &self.name
    }

    #[getter]
    pub fn get_strict(&self) -> bool {
        self.strict
    }

    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}

impl FunctionTool {
    pub(crate) fn into_dynamic_tool_config(self) -> DynamicToolConfig {
        DynamicToolConfig {
            description: self.description,
            parameters: JSONSchema::compile_background(self.parameters),
            name: self.name,
            strict: self.strict,
        }
    }
}

/// `OpenAICustomTool` represents OpenAI's custom tool format, which allows
/// for text or grammar-based tool definitions beyond standard function calling.
/// Currently, this type is a wire + outbound + storage type so it forces a consistent format.
/// This only applies to the Chat Completions API. The Responses API has a slightly different request
/// shape so we implement a conversion in `responses.rs`.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct OpenAICustomTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<OpenAICustomToolFormat>,
}

impl fmt::Display for OpenAICustomTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAICustomToolFormat {
    #[schemars(title = "OpenAICustomToolFormatText")]
    Text,
    #[schemars(title = "OpenAICustomToolFormatGrammar")]
    Grammar { grammar: OpenAIGrammarDefinition },
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct OpenAIGrammarDefinition {
    pub syntax: OpenAIGrammarSyntax,
    pub definition: String,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(rename_all = "snake_case")]
pub enum OpenAIGrammarSyntax {
    Lark,
    Regex,
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl OpenAICustomTool {
    #[getter]
    pub fn get_name(&self) -> &str {
        &self.name
    }

    #[getter]
    pub fn get_description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    #[getter]
    pub fn get_format<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match &self.format {
            Some(format) => serialize_to_dict(py, format.clone()).map(|x| Some(x.into_bound(py))),
            None => Ok(None),
        }
    }

    pub fn __repr__(&self) -> String {
        format!("OpenAICustomTool(name='{}')", self.name)
    }
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
#[schemars(title = "ProviderToolScopeModelProvider")]
#[cfg_attr(feature = "ts-bindings", ts(optional_fields))]
pub struct ProviderToolScopeModelProvider {
    pub model_name: String,
    #[serde(alias = "model_provider_name", skip_serializing_if = "Option::is_none")] // legacy
    pub provider_name: Option<String>,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize, JsonSchema)]
#[serde(untagged)]
#[export_schema]
#[cfg_attr(feature = "ts-bindings", ts(optional_fields))]
pub enum ProviderToolScope {
    #[default]
    Unscoped,
    ModelProvider(ProviderToolScopeModelProvider),
}

impl ProviderToolScope {
    pub(crate) fn matches(&self, scope_model_name: &str, scope_provider_name: &str) -> bool {
        match self {
            ProviderToolScope::Unscoped => true,
            ProviderToolScope::ModelProvider(mp) => {
                if scope_model_name != mp.model_name {
                    return false;
                }
                match &mp.provider_name {
                    Some(pn) => scope_provider_name == pn,
                    None => true, // If provider_name is None, match any provider for this model
                }
            }
        }
    }
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct ProviderTool {
    #[serde(default)]
    pub scope: ProviderToolScope,
    pub tool: Value,
}

impl fmt::Display for ProviderTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| fmt::Error)?;
        write!(f, "{json}")
    }
}
