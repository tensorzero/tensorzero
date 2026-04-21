//! Tool types for inference API requests.
//!
//! These were originally in `tensorzero-core::tool`, but live here so that
//! leaf crates (e.g. `ts-executor-pool`) can consume them without pulling in
//! all of `tensorzero-core`.

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use std::fmt;
use strum::AsRefStr;
use tensorzero_derive::export_schema;
use tensorzero_types::ToolChoice;

use crate::{OpenAICustomTool, ProviderTool};

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
    pub fn name(&self) -> &str {
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
            Tool::Function(tool) => value_to_py_dict(py, &tool.parameters),
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
                Some(format) => serialize_to_py(py, format),
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
        value_to_py_dict(py, &self.parameters)
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

/// Wire/API representation of dynamic tool parameters for inference requests.
///
/// This type is the **wire format** for tool configurations used in API requests and responses.
/// It distinguishes between static tools (configured in the function) and dynamic tools
/// (provided at runtime), allowing clients to reference pre-configured tools by name or
/// provide new tools on-the-fly.
///
/// # Purpose
/// - Accept tool parameters in inference API requests (e.g., `/inference/{function_name}`)
/// - Expose tool configurations in API responses for stored inferences
/// - Support Python and TypeScript client bindings
/// - Allow runtime customization of tool behavior
///
/// # Fields
/// - `allowed_tools`: Names of static tools from function config to use (subset selection)
/// - `additional_tools`: New tools defined at runtime (not in static config)
/// - `tool_choice`: Override the function's default tool choice strategy
/// - `parallel_tool_calls`: Override whether parallel tool calls are enabled
/// - `provider_tools`: Provider-specific tool configurations (not persisted to database)
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "ts-bindings", ts(optional_fields, export))]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[export_schema]
pub struct DynamicToolParams {
    /// A subset of static tools configured for the function that the inference is allowed to use. Optional.
    /// If not provided, all static tools are allowed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<Vec<String>>,

    /// Tools that the user provided at inference time (not in function config), in addition to the function-configured
    /// tools, that are also allowed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additional_tools: Option<Vec<Tool>>,
    /// User-specified tool choice strategy. If provided during inference, it will override the function-configured tool choice.
    /// Optional.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Whether to use parallel tool calls in the inference. Optional.
    /// If provided during inference, it will override the function-configured parallel tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    /// Provider-specific tool configurations
    #[serde(default)]
    pub provider_tools: Vec<ProviderTool>,
}

impl std::fmt::Display for DynamicToolParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl DynamicToolParams {
    #[getter]
    pub fn allowed_tools(&self) -> Option<Vec<String>> {
        self.allowed_tools.clone()
    }

    #[getter]
    pub fn additional_tools(&self) -> Option<Vec<Tool>> {
        self.additional_tools.clone()
    }

    // TODO: Add tool_choice getter when we decide how to handle it.
    // Mixed enums (with unit and tuple variants) aren't well supported in PyO3,
    // and we need to decide on the proper Python representation.

    #[getter]
    pub fn parallel_tool_calls(&self) -> Option<bool> {
        self.parallel_tool_calls
    }

    #[getter]
    pub fn provider_tools<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        serialize_to_py(py, &self.provider_tools)
    }

    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}

#[cfg(feature = "pyo3")]
fn serialize_to_py<'py, T: Serialize>(py: Python<'py>, val: &T) -> PyResult<Bound<'py, PyAny>> {
    let json_str = serde_json::to_string(val).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Failed to serialize to JSON: {e:?}"))
    })?;
    let json_module = py.import("json")?;
    Ok(json_module.call_method1("loads", (json_str,))?.into_any())
}

#[cfg(feature = "pyo3")]
fn value_to_py_dict<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    serialize_to_py(py, value)
}
