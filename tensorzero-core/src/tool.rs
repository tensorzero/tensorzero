use std::fmt;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use schemars::JsonSchema;
use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use tensorzero_derive::export_schema;

use crate::endpoints::datasets::v1::types::UpdateDynamicToolParamsRequest;
#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::serialize_to_dict;
use crate::{
    config::Config,
    error::{Error, ErrorDetails},
    function::FunctionConfig,
    jsonschema_util::{DynamicJSONSchema, StaticJSONSchema},
    rate_limiting::{get_estimated_tokens, RateLimitedInputContent},
};
use strum::AsRefStr;

/*  Key tool types in TensorZero
 * - DynamicToolParams: the wire format for tool configuration info (flattened into struct body)
 *       contains a disjoint set of information from that specified in FunctionConfig and config.tools
 * - ToolCallConfig: the representation at inference time of what tool calls are possible
 * - ToolCallConfigDatabaseInsert: the storage format for tool call configuration info
 *     In a close-following PR @viraj will refactor this type.
 * All of these types are convertible given access to the current Config. The conversion from ToolCallConfig
 * to ToolCallConfigDatabaseInsert is temporarily lossy because we don't yet stored dynamic provider tools.
 *
 * Tool: represents a single Tool that could be called by an LLM. This will be generalized soon to an enum.
 * ToolCall: represents a request by an LLM to call a tool.
 * ToolResult: the response from a tool call.
 */

/* A Tool is a function that can be called by an LLM
 * We represent them in various ways depending on how they are configured by the user.
 * The primary difficulty is that tools require an input signature that we represent as a JSONSchema.
 * JSONSchema compilation takes time so we want to do it at startup if the tool is in the config.
 * We also don't want to clone compiled JSON schemas.
 * If the tool is dynamic we want to run compilation while LLM inference is happening so that we can validate the tool call arguments.
 *
 * If we are doing an implicit tool call for JSON schema enforcement, we can use the compiled schema from the output signature.
 */

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
#[derive(AsRefStr, Clone, Debug, JsonSchema, PartialEq, Serialize, ts_rs::TS)]
#[serde(tag = "type", rename_all = "snake_case")]
#[ts(export)]
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
    fn name(&self) -> &str {
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
#[derive(ts_rs::TS, Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[ts(export)]
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

impl std::fmt::Display for FunctionTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
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
            parameters: DynamicJSONSchema::new(self.parameters),
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
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct OpenAICustomTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<OpenAICustomToolFormat>,
}

impl std::fmt::Display for OpenAICustomTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[derive(ts_rs::TS, Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[ts(export)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAICustomToolFormat {
    #[schemars(title = "OpenAICustomToolFormatText")]
    Text,
    #[schemars(title = "OpenAICustomToolFormatGrammar")]
    Grammar { grammar: OpenAIGrammarDefinition },
}

#[derive(ts_rs::TS, Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[ts(export)]
pub struct OpenAIGrammarDefinition {
    pub syntax: OpenAIGrammarSyntax,
    pub definition: String,
}

#[derive(ts_rs::TS, Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[ts(export)]
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

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS, JsonSchema)]
#[schemars(title = "ProviderToolScopeModelProvider")]
#[ts(optional_fields)]
pub struct ProviderToolScopeModelProvider {
    pub model_name: String,
    #[serde(alias = "model_provider_name", skip_serializing_if = "Option::is_none")] // legacy
    pub provider_name: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize, ts_rs::TS, JsonSchema)]
#[serde(untagged)]
#[ts(optional_fields)]
#[export_schema]
pub enum ProviderToolScope {
    #[default]
    Unscoped,
    ModelProvider(ProviderToolScopeModelProvider),
}

impl ProviderToolScope {
    fn matches(&self, scope_model_name: &str, scope_provider_name: &str) -> bool {
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

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS, JsonSchema)]
#[ts(export)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct ProviderTool {
    #[serde(default)]
    pub scope: ProviderToolScope,
    pub tool: Value,
}

impl std::fmt::Display for ProviderTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[derive(ts_rs::TS)]
#[ts(export)]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum ToolConfig {
    Function(FunctionToolConfig),
    OpenAICustom(OpenAICustomTool),
}

#[derive(ts_rs::TS)]
#[ts(export)]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum FunctionToolConfig {
    Static(Arc<StaticToolConfig>),
    Dynamic(DynamicToolConfig),
    Implicit(ImplicitToolConfig),
    DynamicImplicit(DynamicImplicitToolConfig),
}

/// Contains the configuration information for a specific tool
#[derive(ts_rs::TS)]
#[ts(export)]
#[derive(Debug, PartialEq, Serialize)]
pub struct StaticToolConfig {
    pub description: String,
    pub parameters: StaticJSONSchema,
    pub name: String,
    pub strict: bool,
}

/// Contains the configuration information for a tool defined at runtime
#[derive(ts_rs::TS)]
#[ts(export)]
#[derive(Debug, PartialEq, Clone, Serialize)]
pub struct DynamicToolConfig {
    pub description: String,
    pub parameters: DynamicJSONSchema,
    pub name: String,
    pub strict: bool,
}

/// Contains the configuration information for a tool used in implicit tool calling for
/// JSON schema enforcement
#[derive(ts_rs::TS)]
#[ts(export)]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ImplicitToolConfig {
    pub parameters: StaticJSONSchema,
}

/// Contains the configuration information for a tool used in implicit tool calling for
/// JSON schema enforcement for a JSON schema that is dynamically passed at inference time
#[derive(ts_rs::TS)]
#[ts(export)]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DynamicImplicitToolConfig {
    pub parameters: DynamicJSONSchema,
}

/// Records / lists the tools that were allowed in the request
/// Also lists how they were set (default, dynamically set)
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, ts_rs::TS)]
#[serde(deny_unknown_fields)]
#[ts(export)]
pub struct AllowedTools {
    pub tools: Vec<String>,
    pub choice: AllowedToolsChoice,
}

impl AllowedTools {
    pub fn into_dynamic_allowed_tools(self) -> Option<Vec<String>> {
        #[expect(deprecated)]
        match self.choice {
            AllowedToolsChoice::FunctionDefault => None,
            AllowedToolsChoice::DynamicAllowedTools | AllowedToolsChoice::Explicit => {
                Some(self.tools.into_iter().collect())
            }
        }
    }

    pub fn as_dynamic_allowed_tools(&self) -> Option<Vec<&str>> {
        #[expect(deprecated)]
        match self.choice {
            AllowedToolsChoice::FunctionDefault => None,
            AllowedToolsChoice::DynamicAllowedTools | AllowedToolsChoice::Explicit => {
                Some(self.tools.iter().map(|s| s.as_str()).collect())
            }
        }
    }
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, ts_rs::TS)]
#[ts(export)]
#[serde(rename_all = "snake_case")]
pub enum AllowedToolsChoice {
    /// If `allowed_tools` is not explicitly passed, we set the function tools
    /// by default and add any dynamic tools
    #[default]
    FunctionDefault,
    /// If `allowed_tools` was explicitly passed we use that list only and then automatically add dynamically set tools
    /// This is deprecated but we keep it around as it may still be in the database.
    /// We have never allowed users to specify AllowedToolsChoice so this is more about the semantics of the data than anything else.
    #[deprecated]
    DynamicAllowedTools,
    /// Currently, we match OpenAI in that if allowed tools is set we only allow the tools that are in it.
    Explicit,
}

/// Reference to either a function tool config or an OpenAI custom tool.
/// Used by the OpenAI provider to iterate over all available tools.
pub enum ToolConfigRef<'a> {
    Function(&'a FunctionToolConfig),
    OpenAICustom(&'a OpenAICustomTool),
}

/// Contains all information required to tell an LLM what tools it can call
/// and what sorts of tool calls (parallel, none, etc) it is allowed to respond with.
/// Most inference providers can convert this into their desired tool format.
#[derive(Clone, Debug, Default, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct ToolCallConfig {
    pub(crate) static_tools_available: Vec<FunctionToolConfig>,
    pub(crate) dynamic_tools_available: Vec<FunctionToolConfig>,
    pub provider_tools: Vec<ProviderTool>,
    pub openai_custom_tools: Vec<OpenAICustomTool>,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: Option<bool>,
    pub allowed_tools: AllowedTools,
}

pub struct ToolCallConfigConstructorArgs<'a> {
    pub function_tools: &'a [String],
    pub function_tool_choice: &'a ToolChoice,
    pub function_parallel_tool_calls: Option<bool>,
    pub static_tools: &'a HashMap<String, Arc<StaticToolConfig>>,
    pub dynamic_allowed_tools: Option<Vec<String>>,
    pub dynamic_additional_tools: Option<Vec<Tool>>,
    pub dynamic_tool_choice: Option<ToolChoice>,
    pub dynamic_parallel_tool_calls: Option<bool>,
    pub dynamic_provider_tools: Vec<ProviderTool>,
}

impl<'a> ToolCallConfigConstructorArgs<'a> {
    /// Returns a ToolCallConfigConstructorArgs with dynamic tool param fields set to defaults.
    /// Use this with struct update syntax to avoid specifying all dynamic fields at callsites.
    pub fn with_dynamic_tool_params(
        function_tools: &'a [String],
        function_tool_choice: &'a ToolChoice,
        function_parallel_tool_calls: Option<bool>,
        static_tools: &'a HashMap<String, Arc<StaticToolConfig>>,
    ) -> Self {
        Self {
            function_tools,
            function_tool_choice,
            function_parallel_tool_calls,
            static_tools,
            dynamic_allowed_tools: None,
            dynamic_additional_tools: None,
            dynamic_tool_choice: None,
            dynamic_parallel_tool_calls: None,
            dynamic_provider_tools: Vec::new(),
        }
    }

    // Helper to construct ToolCallConfigConstructorArgs with defaults
    #[cfg(test)]
    pub fn new_for_test(
        function_tools: &'a [String],
        function_tool_choice: &'a ToolChoice,
        function_parallel_tool_calls: Option<bool>,
        static_tools: &'a HashMap<String, Arc<StaticToolConfig>>,
        dynamic_tool_params: DynamicToolParams,
    ) -> ToolCallConfigConstructorArgs<'a> {
        ToolCallConfigConstructorArgs {
            function_tools,
            function_tool_choice,
            function_parallel_tool_calls,
            static_tools,
            dynamic_allowed_tools: dynamic_tool_params.allowed_tools,
            dynamic_additional_tools: dynamic_tool_params.additional_tools,
            dynamic_tool_choice: dynamic_tool_params.tool_choice,
            dynamic_parallel_tool_calls: dynamic_tool_params.parallel_tool_calls,
            dynamic_provider_tools: dynamic_tool_params.provider_tools,
        }
    }
}

impl ToolCallConfig {
    /// Creates a new `ToolCallConfig` from the provided arguments.
    ///
    /// This method validates and categorizes tools into three groups:
    /// 1. **Function tools**: Tools explicitly configured in the function's tool list
    /// 2. **Dynamic tools**: Tools provided at inference time via `dynamic_additional_tools`
    /// 3. **Config-only allowed tools**: Tools from the TensorZero config that are in `allowed_tools` but not in the function's tool list or the dynamic tool
    ///
    /// We store function tools + config-only allowed tools in `static_tools_available`.
    /// We store dynamic tools in `dynamic_tools_available`.
    /// We check here that there are no tools with duplicate display names.
    /// We also validate tool choice arguments.
    /// If there are no tools we return None.
    pub fn new(args: ToolCallConfigConstructorArgs<'_>) -> Result<Option<Self>, Error> {
        let ToolCallConfigConstructorArgs {
            function_tools,
            function_tool_choice,
            function_parallel_tool_calls,
            static_tools,
            dynamic_allowed_tools,
            dynamic_additional_tools,
            dynamic_tool_choice,
            dynamic_parallel_tool_calls,
            dynamic_provider_tools,
        } = args;
        let allowed_tools = match dynamic_allowed_tools {
            Some(allowed_tools) => AllowedTools {
                tools: allowed_tools,
                choice: AllowedToolsChoice::Explicit,
            },
            // If `allowed_tools` is not provided, use the function's configured tools plus any dynamic tools.
            // This means we allow all tools for the function.
            None => {
                // Collect function tools
                let mut tools: Vec<String> = function_tools.to_vec();

                // Add dynamic tool names in FunctionDefault mode
                if let Some(additional_tools) = &dynamic_additional_tools {
                    tools.extend(additional_tools.iter().map(|t| t.name().to_string()));
                }

                AllowedTools {
                    tools,
                    choice: AllowedToolsChoice::FunctionDefault,
                }
            }
        };

        // Build set of all available tool names (static + dynamic)
        let additional_tool_names: HashSet<&str> = dynamic_additional_tools
            .as_ref()
            .map(|tools| tools.iter().map(|dt| dt.name()).collect())
            .unwrap_or_default();

        let all_available_tool_names: HashSet<String> = static_tools
            .keys()
            .cloned()
            .chain(additional_tool_names.iter().map(|&s| s.to_string()))
            .collect();

        // Validate that all tools in allowed_tools exist in the union of static + dynamic tools
        for tool_name in &allowed_tools.tools {
            if !all_available_tool_names.contains(tool_name) {
                return Err(Error::new(ErrorDetails::ToolNotFound {
                    name: tool_name.clone(),
                }));
            }
        }

        // Get all static tools from function_tools and allowed_tools
        // First, collect tools from function_tools (preserving order)
        let mut static_tool_names: Vec<&str> = function_tools.iter().map(|s| s.as_str()).collect();

        // Then, add any tools from allowed_tools that exist in static_tools but not in function_tools
        // This ensures that all allowed tools from the config are actually available
        for tool_name in &allowed_tools.tools {
            if static_tools.contains_key(tool_name)
                && !static_tool_names.contains(&tool_name.as_str())
                && !additional_tool_names.contains(&tool_name.as_str())
            {
                static_tool_names.push(tool_name);
            }
        }

        let static_tools_available: Vec<FunctionToolConfig> = static_tool_names
            .iter()
            .filter_map(|tool_name| {
                static_tools
                    .get(*tool_name)
                    .map(|static_tool| FunctionToolConfig::Static(static_tool.clone()))
            })
            .collect();

        // Get all dynamic tools
        let mut dynamic_tools_available: Vec<FunctionToolConfig> = Vec::new();
        let mut openai_custom_tools: Vec<OpenAICustomTool> = Vec::new();
        if let Some(dynamic_additional_tools) = dynamic_additional_tools {
            for tool in dynamic_additional_tools {
                match tool {
                    Tool::Function(func) => dynamic_tools_available
                        .push(FunctionToolConfig::Dynamic(func.into_dynamic_tool_config())),
                    Tool::OpenAICustom(custom_tool) => {
                        openai_custom_tools.push(custom_tool);
                    }
                }
            }
        }

        let mut tool_display_names = HashSet::new();

        // Check for duplicate tool names among function tools
        for tool in static_tools_available
            .iter()
            .chain(dynamic_tools_available.iter())
        {
            let duplicate = !tool_display_names.insert(tool.name());
            if duplicate {
                return Err(Error::new(ErrorDetails::DuplicateTool {
                    name: tool.name().to_string(),
                }));
            }
        }

        // Check for duplicate tool names in OpenAI custom tools
        // and ensure they don't conflict with function tool names
        for custom_tool in &openai_custom_tools {
            let duplicate = !tool_display_names.insert(custom_tool.name.as_str());
            if duplicate {
                return Err(Error::new(ErrorDetails::DuplicateTool {
                    name: custom_tool.name.clone(),
                }));
            }
        }

        let tool_choice = dynamic_tool_choice.unwrap_or_else(|| function_tool_choice.clone());

        // If the tool choice is a specific tool, make sure it's in the list of available tools
        if let ToolChoice::Specific(tool_name) = &tool_choice {
            let tool_found = static_tools_available
                .iter()
                .chain(dynamic_tools_available.iter())
                .any(|tool| match tool {
                    FunctionToolConfig::Static(config) => config.name == *tool_name,
                    FunctionToolConfig::Dynamic(config) => config.name == *tool_name,
                    FunctionToolConfig::Implicit(_) => false,
                    FunctionToolConfig::DynamicImplicit(_) => false,
                })
                || openai_custom_tools
                    .iter()
                    .any(|tool| tool.name == *tool_name);

            if !tool_found {
                return Err(ErrorDetails::ToolNotFound {
                    name: tool_name.clone(),
                }
                .into());
            }
        }

        let parallel_tool_calls = dynamic_parallel_tool_calls.or(function_parallel_tool_calls);

        let tool_call_config_option = if static_tools_available.is_empty()
            && dynamic_tools_available.is_empty()
            && dynamic_provider_tools.is_empty()
            && openai_custom_tools.is_empty()
        {
            None
        } else {
            Some(Self {
                static_tools_available,
                dynamic_tools_available,
                openai_custom_tools,
                tool_choice,
                provider_tools: dynamic_provider_tools,
                parallel_tool_calls,
                allowed_tools,
            })
        };

        Ok(tool_call_config_option)
    }

    /// Returns an iterator over all available tools (function tools only).
    /// Returns an error if OpenAI custom tools are present, as they are not compatible
    /// with standard function tool iteration.
    pub fn tools_available(
        &self,
    ) -> Result<Box<dyn Iterator<Item = &FunctionToolConfig> + '_>, Error> {
        if !self.openai_custom_tools.is_empty() {
            return Err(Error::new(ErrorDetails::IncompatibleTool {
                message: "OpenAI custom tools are not supported by this provider".to_string(),
            }));
        }
        Ok(Box::new(
            self.static_tools_available
                .iter()
                .chain(self.dynamic_tools_available.iter()),
        ))
    }

    /// Returns an iterator over all available tools including OpenAI custom tools.
    /// This method is intended for the OpenAI provider only.
    pub fn tools_available_with_openai_custom(
        &self,
    ) -> Box<dyn Iterator<Item = ToolConfigRef<'_>> + '_> {
        Box::new(
            self.static_tools_available
                .iter()
                .map(ToolConfigRef::Function)
                .chain(
                    self.dynamic_tools_available
                        .iter()
                        .map(ToolConfigRef::Function),
                )
                .chain(
                    self.openai_custom_tools
                        .iter()
                        .map(ToolConfigRef::OpenAICustom),
                ),
        )
    }

    /// Returns tools filtered by allowed_tools list and tool type filter.
    /// Returns an error if OpenAI custom tools are present, as they are not compatible
    /// with standard function tool filtering.
    ///
    /// # Behavior
    /// - For FunctionDefault and DynamicAllowedTools modes: returns tools based on type filter
    /// - For Explicit mode: applies allowed_tools filtering first, then tool type filtering
    pub fn strict_tools_available(
        &self,
    ) -> Result<Box<dyn Iterator<Item = &FunctionToolConfig> + '_>, Error> {
        if !self.openai_custom_tools.is_empty() {
            return Err(Error::new(ErrorDetails::IncompatibleTool {
                message: "OpenAI custom tools are not supported by this provider".to_string(),
            }));
        }
        match self.allowed_tools.choice {
            #[expect(deprecated)] // DynamicAllowedTools
            AllowedToolsChoice::FunctionDefault | AllowedToolsChoice::DynamicAllowedTools => {
                // Return all tools based on type filter (lenient mode)
                self.tools_available()
            }
            AllowedToolsChoice::Explicit => {
                // Filter by allowed_tools list, then apply type filter
                Ok(Box::new(
                    self.static_tools_available
                        .iter()
                        .chain(self.dynamic_tools_available.iter())
                        .filter(|tool| self.allowed_tools.tools.iter().any(|t| t == tool.name())),
                ))
            }
        }
    }

    pub fn any_tools_available(&self) -> bool {
        !(self.static_tools_available.is_empty() && self.dynamic_tools_available.is_empty())
    }

    /// This only gets Function tools and skips OpenAI tools
    pub fn get_function_tool(&self, name: &str) -> Option<&FunctionToolConfig> {
        self.static_tools_available
            .iter()
            .chain(self.dynamic_tools_available.iter())
            .find(|tool_cfg| match tool_cfg {
                FunctionToolConfig::Static(config) => config.name == name,
                FunctionToolConfig::Dynamic(config) => config.name == name,
                FunctionToolConfig::Implicit(_config) => false,
                FunctionToolConfig::DynamicImplicit(_config) => false,
            })
    }

    #[cfg(test)]
    pub fn with_tools_available(
        static_tools_available: Vec<FunctionToolConfig>,
        dynamic_tools_available: Vec<FunctionToolConfig>,
    ) -> Self {
        Self {
            static_tools_available,
            dynamic_tools_available,
            ..Default::default()
        }
    }

    pub fn get_scoped_provider_tools(
        &self,
        model_name: &str,
        model_provider_name: &str,
    ) -> Vec<&ProviderTool> {
        self.provider_tools
            .iter()
            .filter(|t| t.scope.matches(model_name, model_provider_name))
            .collect()
    }
}

/// Storage representation of tool call configuration for database persistence.
///
/// This type is the **database/storage format** for tool configurations, designed to be stored
/// in ClickHouse and other persistence layers. It represents a simplified, flattened view of
/// tool configuration after all static and dynamic tools have been merged.
///
/// # Purpose
/// - Store tool configurations in the database alongside inference records
/// - Provide a serializable, cloneable format for persistence
/// - Simplify the tool configuration to a single merged list
///
/// # Key Differences from DynamicToolParams
/// - **Merged tools**: All tools (static from config + dynamic from runtime) are combined into a single `tools_available` list
/// - **No distinction**: Does not track which tools came from static config vs dynamic runtime parameters
/// - **No provider_tools**: This field is not persisted (lossy conversion)
/// - **No bindings**: Not exposed to Python/TypeScript clients (internal storage only)
///
/// # Conversion
/// - **From wire type**: Use `FunctionConfig::dynamic_tool_params_to_database_insert()` to convert `DynamicToolParams` → `ToolCallConfigDatabaseInsert`
/// - **To wire type**: Use `From<ToolCallConfigDatabaseInsert> for DynamicToolParams` trait to convert `ToolCallConfigDatabaseInsert` → `DynamicToolParams`
/// - **To ToolCallConfig**: Use the `into_tool_call_config()` method for a direct conversion to `ToolCallConfig`
///
/// See also: [`DynamicToolParams`] for the wire/API format
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ToolCallConfigDatabaseInsert {
    pub dynamic_tools: Vec<Tool>,
    pub dynamic_provider_tools: Vec<ProviderTool>,
    pub allowed_tools: AllowedTools,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: Option<bool>,
    // We write this in case any legacy code reads the database; it should not be read in new code
    #[serde(default)]
    tool_params: LegacyToolCallConfigDatabaseInsert,
}

/// Custom deserializer implementation for ToolCallConfigDatabaseInsert that handles three formats:
/// 1. Full format: Contains all fields (dynamic_tools, dynamic_provider_tools, allowed_tools, etc.)
/// 2. Legacy format: Contains only tool_config field (for backwards compatibility)
/// 3. Missing/Empty: Returns None
///
/// This deserializer is strict: if any tool-related fields are present, they must be valid and complete.
/// It supports flatten by only consuming tool-related fields and leaving others for the parent struct.
///
/// ## Why a custom deserializer?
///
/// This cannot be simplified using an untagged enum or standard serde derives because:
///
/// 1. **Flatten support requires selective field consumption**: When used with `#[serde(flatten)]`,
///    this deserializer must distinguish between tool-related fields (which it consumes) and
///    other fields (which it skips). An untagged enum would attempt to consume all fields or fail,
///    breaking the flatten behavior.
///
/// 2. **Overlapping field sets**: The `tool_params` field appears in both format variants:
///    - Full format: includes `tool_params` alongside other fields (optional, for legacy compatibility)
///    - Legacy format: contains only `tool_params`
///    This overlap makes it impossible for an untagged enum to reliably distinguish between variants.
///
/// 3. **Complex parsing requirements**: The deserializer performs custom transformations that can't
///    be expressed with derive macros:
///    - Parsing JSON strings into nested types (e.g., `Vec<String>` → `Vec<Tool>`)
///    - Handling multiple representations of the same field (e.g., `tool_choice` as plain string or JSON object)
///    - Backward compatibility fallbacks for different serialization formats
///
/// 4. **The None case**: Returning `None` when no tool fields are present is essential for optional
///    tool configurations. An untagged enum would fail to deserialize rather than gracefully returning None.
pub fn deserialize_optional_tool_info<'de, D>(
    deserializer: D,
) -> Result<Option<ToolCallConfigDatabaseInsert>, D::Error>
where
    D: Deserializer<'de>,
{
    struct ToolInfoVisitor;

    impl<'de> Visitor<'de> for ToolInfoVisitor {
        type Value = Option<ToolCallConfigDatabaseInsert>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("tool call configuration")
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: MapAccess<'de>,
        {
            // Collect only tool-related fields
            let tool_fields = [
                "dynamic_tools",
                "dynamic_provider_tools",
                "allowed_tools",
                "tool_choice",
                "parallel_tool_calls",
                "tool_params",
            ];

            let mut values: HashMap<String, Value> = HashMap::new();

            while let Some(key) = map.next_key::<String>()? {
                if tool_fields.contains(&key.as_str()) {
                    let value: Value = map.next_value()?;
                    if !value.is_null() {
                        values.insert(key, value);
                    }
                } else {
                    // Skip non-tool fields (for flatten support)
                    map.next_value::<serde::de::IgnoredAny>()?;
                }
            }

            // Determine format based on which fields are present
            // Since `dynamic_provider_tools` and `dynamic_tools` are going to return arrays
            // and `tool_params` will be a string regardles of format, the distinguishing factor for new data
            // is if `allowed_tools` is set (it always should be)
            let has_full_fields =
                values.contains_key("allowed_tools") || values.contains_key("tool_choice");

            // If we're NOT in full format mode, filter out empty arrays for dynamic_tools and dynamic_provider_tools
            // This handles the case where ClickHouse returns default values (empty arrays) for these columns
            // when they weren't explicitly set (i.e., legacy data or data without tools)
            if !has_full_fields {
                values.retain(|key, value| {
                    !(value.is_array()
                        && value.as_array().is_some_and(|arr| arr.is_empty())
                        && (key == "dynamic_tools" || key == "dynamic_provider_tools"))
                });
            }

            // If no tool fields present, return None
            if values.is_empty() {
                return Ok(None);
            }

            if has_full_fields {
                // Full format: require ALL full format fields
                let dynamic_tools_value = values
                    .get("dynamic_tools")
                    .ok_or_else(|| de::Error::missing_field("dynamic_tools"))?;

                // Parse as array of JSON strings (database storage format)
                let tool_strings: Vec<String> = serde_json::from_value(dynamic_tools_value.clone())
                    .map_err(|e| {
                        de::Error::custom(format!(
                            "dynamic_tools must be an array of JSON strings: {e}"
                        ))
                    })?;

                let dynamic_tools: Vec<Tool> = tool_strings
                    .iter()
                    .map(|s| {
                        serde_json::from_str(s).map_err(|e| {
                            de::Error::custom(format!("failed to parse tool from JSON string: {e}"))
                        })
                    })
                    .collect::<Result<Vec<Tool>, _>>()?;
                let dynamic_provider_tools_value = values
                    .get("dynamic_provider_tools")
                    .ok_or_else(|| de::Error::missing_field("dynamic_provider_tools"))?;

                // Parse as array of JSON strings (database storage format)
                let provider_tool_strings: Vec<String> =
                    serde_json::from_value(dynamic_provider_tools_value.clone()).map_err(|e| {
                        de::Error::custom(format!(
                            "dynamic_provider_tools must be an array of JSON strings: {e}"
                        ))
                    })?;

                let dynamic_provider_tools: Vec<ProviderTool> = provider_tool_strings
                    .iter()
                    .map(|s| {
                        serde_json::from_str(s).map_err(|e| {
                            de::Error::custom(format!(
                                "failed to parse provider tool from JSON string: {e}"
                            ))
                        })
                    })
                    .collect::<Result<Vec<ProviderTool>, _>>()?;

                let allowed_tools_value = values
                    .get("allowed_tools")
                    .ok_or_else(|| de::Error::missing_field("allowed_tools"))?;

                // Parse as JSON string (database storage format)
                let allowed_tools: AllowedTools =
                    if let Some(allowed_tools_str) = allowed_tools_value.as_str() {
                        serde_json::from_str(allowed_tools_str).map_err(|e| {
                            de::Error::custom(format!(
                                "failed to parse allowed_tools from JSON string: {e}"
                            ))
                        })?
                    } else {
                        // Fallback: try to deserialize as object (for backwards compatibility)
                        serde_json::from_value(allowed_tools_value.clone()).map_err(|e| {
                            de::Error::custom(format!(
                                "allowed_tools must be a JSON string or object: {e}"
                            ))
                        })?
                    };

                let tool_choice_value = values
                    .get("tool_choice")
                    .ok_or_else(|| de::Error::missing_field("tool_choice"))?;

                // Parse tool_choice - it's stored as a string in ClickHouse
                // Simple variants (auto, none, required) are stored as plain strings like "auto"
                // Complex variants (specific) are stored as JSON strings like "{\"specific\":\"tool_name\"}"
                let tool_choice: ToolChoice =
                    if let Some(tool_choice_str) = tool_choice_value.as_str() {
                        // Try parsing as a plain string first (for simple variants)
                        serde_json::from_value(Value::String(tool_choice_str.to_string()))
                            .or_else(|_| {
                                // If that fails, try parsing the string as JSON (for complex variants)
                                serde_json::from_str(tool_choice_str)
                            })
                            .map_err(|e| {
                                de::Error::custom(format!("failed to parse tool_choice: {e}"))
                            })?
                    } else {
                        // Fallback for non-string values (e.g., direct object for backwards compatibility)
                        serde_json::from_value(tool_choice_value.clone()).map_err(|e| {
                            de::Error::custom(format!("failed to parse tool_choice: {e}"))
                        })?
                    };

                let parallel_tool_calls: Option<bool> = values
                    .get("parallel_tool_calls")
                    .map(|v| {
                        if v.is_null() {
                            Ok(None)
                        } else {
                            serde_json::from_value::<bool>(v.clone())
                                .map(Some)
                                .map_err(|e| {
                                    de::Error::custom(format!("invalid parallel_tool_calls: {e}"))
                                })
                        }
                    })
                    .transpose()?
                    .flatten();
                // The tool params are serialized as a string in ClickHouse
                // but an Object in Python. In the full format, tool_params is optional
                // since the data is stored in the decomposed fields.
                let tool_config: LegacyToolCallConfigDatabaseInsert =
                    if let Some(tool_config_value) = values.get("tool_params") {
                        // Handle null case
                        if tool_config_value.is_null() {
                            LegacyToolCallConfigDatabaseInsert::default()
                        } else if let Some(tool_config_str) = tool_config_value.as_str() {
                            // Handle string case (ClickHouse serialization)
                            // ClickHouse empty string is None
                            if tool_config_str.is_empty() {
                                LegacyToolCallConfigDatabaseInsert::default()
                            } else {
                                serde_json::from_str(tool_config_str).map_err(|e| {
                                    de::Error::custom(format!("invalid tool_params string: {e}"))
                                })?
                            }
                        } else {
                            // Handle object case (Python serialization)
                            serde_json::from_value(tool_config_value.clone()).map_err(|e| {
                                de::Error::custom(format!("invalid tool_params object: {e}"))
                            })?
                        }
                    } else {
                        // tool_params not present - use default
                        LegacyToolCallConfigDatabaseInsert::default()
                    };
                Ok(Some(ToolCallConfigDatabaseInsert {
                    dynamic_tools,
                    dynamic_provider_tools,
                    allowed_tools,
                    tool_choice,
                    parallel_tool_calls,
                    tool_params: tool_config,
                }))
            } else if values.contains_key("tool_params") {
                // Legacy format: only tool_config should be present
                // The tool params are serialized as a string in ClickHouse
                let tool_config_value = values
                    .get("tool_params")
                    .ok_or_else(|| de::Error::missing_field("tool_params"))?;

                // Handle null case - return None if tool_params is explicitly null
                if tool_config_value.is_null() {
                    return Ok(None);
                }

                let tool_config: LegacyToolCallConfigDatabaseInsert =
                    if let Some(tool_config_str) = tool_config_value.as_str() {
                        // Handle string case (ClickHouse serialization)
                        // ClickHouse empty string is None
                        if tool_config_str.is_empty() {
                            return Ok(None);
                        }
                        serde_json::from_str(tool_config_str).map_err(|e| {
                            de::Error::custom(format!("invalid tool_params string: {e}"))
                        })?
                    } else {
                        // Handle object case (Python serialization)
                        serde_json::from_value(tool_config_value.clone()).map_err(|e| {
                            de::Error::custom(format!("invalid tool_params object: {e}"))
                        })?
                    };

                Ok(Some(ToolCallConfigDatabaseInsert {
                    dynamic_tools: vec![],
                    dynamic_provider_tools: vec![],
                    allowed_tools: AllowedTools::default(),
                    tool_choice: tool_config.tool_choice.clone(),
                    parallel_tool_calls: tool_config.parallel_tool_calls,
                    tool_params: tool_config,
                }))
            } else {
                // Unknown tool fields without proper structure
                Err(de::Error::custom(
                    "invalid tool configuration: unrecognized field combination",
                ))
            }
        }
    }

    deserializer.deserialize_map(ToolInfoVisitor)
}

/// Non-optional deserializer that uses the optional deserializer and defaults to empty if None
pub fn deserialize_tool_info<'de, D>(
    deserializer: D,
) -> Result<ToolCallConfigDatabaseInsert, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_optional_tool_info(deserializer).map(Option::unwrap_or_default)
}

impl std::fmt::Display for ToolCallConfigDatabaseInsert {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl ToolCallConfigDatabaseInsert {
    /// Creates a `ToolCallConfigDatabaseInsert` for testing purposes.
    ///
    /// # Understanding the Data Model
    ///
    /// `ToolCallConfigDatabaseInsert` stores tool configuration for database persistence.
    /// The key insight is that **static tools are NOT stored** - they come from the function
    /// config and are reconstructed when converting back to `ToolCallConfig`.
    ///
    /// ## Fields Explained
    ///
    /// - **`dynamic_tools`**: Tools provided at runtime (not in function config).
    ///   These are the *only* tool definitions we store in the database.
    ///
    /// - **`allowed_tools`**: Which tools (by name) are allowed to be used.
    ///   - `tools`: List of tool names (can be static, dynamic, or mixed)
    ///   - `choice`: How the allowed tools were determined:
    ///     - `FunctionDefault`: Use function's default tool list
    ///     - `DynamicAllowedTools`: Explicitly specified tool list (possibly different from function defaults)
    ///
    /// ## Conversion Back to DynamicToolParams
    ///
    /// When using `From<ToolCallConfigDatabaseInsert> for DynamicToolParams`:
    /// - If `choice == FunctionDefault` → `allowed_tools = None` (use function defaults)
    /// - If `choice == DynamicAllowedTools` → `allowed_tools = Some(tools)` (explicit override)
    /// - `additional_tools = Some(dynamic_tools)` if dynamic_tools is non-empty, else None
    ///
    /// ## Test Scenarios
    ///
    /// **Scenario 1: Only static tools (from function config)**
    /// ```
    /// // Function has: tools = ["tool1", "tool2"]
    /// // User doesn't provide additional tools, just uses the function's tools
    /// ToolCallConfigDatabaseInsert::new_for_test(
    ///     vec![],  // No dynamic tools
    ///     vec![],
    ///     AllowedTools {
    ///         tools: vec!["tool1".to_string(), "tool2".to_string()],
    ///         choice: AllowedToolsChoice::DynamicAllowedTools,  // Explicit list
    ///     },
    ///     ...
    /// )
    /// // Converts back to: allowed_tools=Some(["tool1", "tool2"]), additional_tools=None
    /// ```
    ///
    /// **Scenario 2: Only dynamic tools (not in function config)**
    /// ```
    /// // Function has: tools = ["static1"]
    /// // User provides new tools at runtime
    /// ToolCallConfigDatabaseInsert::new_for_test(
    ///     vec![Tool::Function(dynamic1), Tool::Function(dynamic2)],
    ///     vec![],
    ///     AllowedTools {
    ///         tools: vec![],  // Empty because these are dynamic, not in function config
    ///         choice: AllowedToolsChoice::DynamicAllowedTools,
    ///     },
    ///     ...
    /// )
    /// // Converts back to: allowed_tools=Some([]), additional_tools=Some([dynamic1, dynamic2])
    /// ```
    ///
    /// **Scenario 3: Mixed static and dynamic tools**
    /// ```
    /// // Function has: tools = ["a", "b"]
    /// // User also provides dynamic tools x, y
    /// ToolCallConfigDatabaseInsert::new_for_test(
    ///     vec![Tool::Function(x), Tool::Function(y)],
    ///     vec![],
    ///     AllowedTools {
    ///         tools: vec!["a".to_string(), "b".to_string()],
    ///         choice: AllowedToolsChoice::DynamicAllowedTools,
    ///     },
    ///     ...
    /// )
    /// // Converts back to: allowed_tools=Some(["a", "b"]), additional_tools=Some([x, y])
    /// ```
    #[cfg(any(test, feature = "e2e_tests"))]
    pub fn new_for_test(
        dynamic_tools: Vec<Tool>,
        dynamic_provider_tools: Vec<ProviderTool>,
        allowed_tools: AllowedTools,
        tool_choice: ToolChoice,
        parallel_tool_calls: Option<bool>,
    ) -> Self {
        // Compute the legacy tool_config field
        let tool_config = LegacyToolCallConfigDatabaseInsert {
            tools_available: dynamic_tools
                .iter()
                .filter_map(|t| match t {
                    Tool::Function(csf) => Some(csf.clone()),
                    Tool::OpenAICustom(_) => None, // Custom tools not supported in legacy format
                })
                .collect(),
            tool_choice: tool_choice.clone(),
            parallel_tool_calls,
        };

        Self {
            dynamic_tools,
            dynamic_provider_tools,
            allowed_tools,
            tool_choice,
            parallel_tool_calls,
            tool_params: tool_config,
        }
    }

    /// Converts back from a `ToolCallConfigDatabaseInsert` (storage type) to
    /// `ToolCallConfig` (internal type with nonserializable state).
    pub fn into_tool_call_config(
        self,
        function_config: &FunctionConfig,
        static_tools: &HashMap<String, Arc<StaticToolConfig>>,
    ) -> Result<Option<ToolCallConfig>, Error> {
        match function_config {
            FunctionConfig::Chat(params) => ToolCallConfig::new(ToolCallConfigConstructorArgs {
                dynamic_allowed_tools: self.allowed_tools.into_dynamic_allowed_tools(),
                dynamic_additional_tools: Some(self.dynamic_tools),
                dynamic_parallel_tool_calls: self.parallel_tool_calls,
                dynamic_provider_tools: self.dynamic_provider_tools,
                dynamic_tool_choice: Some(self.tool_choice),
                ..ToolCallConfigConstructorArgs::with_dynamic_tool_params(
                    &params.tools,
                    &params.tool_choice,
                    params.parallel_tool_calls,
                    static_tools,
                )
            }),
            FunctionConfig::Json(_) => Ok(None),
        }
    }

    pub fn tools_available(
        &self,
        function_name: &str,
        config: &Config,
    ) -> Result<impl Iterator<Item = Tool> + '_, Error> {
        let function_config = config.get_function(function_name)?;

        // Get the list of tool names from allowed_tools based on whether they were dynamically set
        #[expect(deprecated)]
        let tool_names = match self.allowed_tools.choice {
            AllowedToolsChoice::FunctionDefault => {
                // Use the function's configured tool names
                function_config
                    .tools()
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            }
            AllowedToolsChoice::DynamicAllowedTools | AllowedToolsChoice::Explicit => {
                // Use the dynamically specified tool names
                self.allowed_tools.tools.to_vec()
            }
        };

        // Collect static tools from config
        let static_tools: Vec<Tool> = tool_names
            .iter()
            .filter_map(|tool_name| {
                config.tools.get(tool_name).map(|static_tool| {
                    Tool::Function(FunctionTool {
                        description: static_tool.description.clone(),
                        parameters: static_tool.parameters.value.clone(),
                        name: static_tool.name.clone(),
                        strict: static_tool.strict,
                    })
                })
            })
            .collect();

        // Combine static tools and dynamic tools
        let all_tools = static_tools
            .into_iter()
            .chain(self.dynamic_tools.iter().cloned());

        Ok(all_tools)
    }
}

/// Updates the dynamic tool parameters with the provided request and returns the updated ToolCallConfigDatabaseInsert.
pub fn apply_dynamic_tool_params_update_to_tool_call_config(
    existing_tool_params: Option<ToolCallConfigDatabaseInsert>,
    update_request: UpdateDynamicToolParamsRequest,
    function_config: &FunctionConfig,
    static_tools: &HashMap<String, Arc<StaticToolConfig>>,
) -> Result<Option<ToolCallConfigDatabaseInsert>, Error> {
    if update_request.allowed_tools.is_none()
        && update_request.additional_tools.is_none()
        && update_request.tool_choice.is_none()
        && update_request.parallel_tool_calls.is_none()
        && update_request.provider_tools.is_none()
    {
        return Ok(existing_tool_params);
    }

    let mut merged_dynamic_tool_params: DynamicToolParams =
        existing_tool_params.unwrap_or_default().into();

    // Handle allowed_tools (three-state: omitted, null, value)
    // Omitted (None): no change
    // Some(None) = explicitly null -> clear to None
    // Some(Some(vec)) = set to explicit list
    if let Some(allowed_tools) = update_request.allowed_tools {
        merged_dynamic_tool_params.allowed_tools = allowed_tools;
    }

    // Handle additional_tools
    if let Some(additional_tools) = update_request.additional_tools {
        merged_dynamic_tool_params.additional_tools = Some(additional_tools);
    }

    // Handle tool_choice (three-state: omitted, null, value)
    if let Some(tool_choice_opt) = update_request.tool_choice {
        // Some(None) = explicitly null -> clear to None (use function default)
        // Some(Some(choice)) = set to specific value
        merged_dynamic_tool_params.tool_choice = tool_choice_opt;
    }

    // Handle parallel_tool_calls (three-state: omitted, null, value)
    if let Some(parallel_opt) = update_request.parallel_tool_calls {
        merged_dynamic_tool_params.parallel_tool_calls = parallel_opt;
    }

    // Handle provider_tools
    if let Some(provider_tools) = update_request.provider_tools {
        merged_dynamic_tool_params.provider_tools = provider_tools;
    }

    function_config.dynamic_tool_params_to_database_insert(merged_dynamic_tool_params, static_tools)
}

/// This is a legacy struct. We use it for deserializing historical data and
/// continuing to write the same format only.
/// This should not be used in new code.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct LegacyToolCallConfigDatabaseInsert {
    /// All tools available for this inference (merged static + dynamic tools)
    pub tools_available: Vec<FunctionTool>,
    /// The tool choice strategy
    pub tool_choice: ToolChoice,
    // TODO: decide what we want the Python interface to be for ToolChoice
    // This is complicated because ToolChoice is an enum with some simple arms and some
    // struct arms. We would likely need to land on one of the serde options for enums (tagged?)
    /// Whether parallel tool calls are enabled
    pub parallel_tool_calls: Option<bool>,
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
///
/// # Key Differences from ToolCallConfigDatabaseInsert
/// - **Separate lists**: Maintains distinction between static (`allowed_tools`) and dynamic (`additional_tools`) tools
/// - **By reference**: Static tools referenced by name, not duplicated
/// - **Has provider_tools**: Can specify provider-specific tool configurations
/// - **Has bindings**: Exposed to Python/TypeScript via `pyo3` and `ts_rs`
///
/// # Conversion to Storage Format
/// Converting from `DynamicToolParams` to `ToolCallConfigDatabaseInsert` is a **lossy** operation:
/// 1. Static tools (from `allowed_tools` names) are resolved from function config
/// 2. Dynamic tools (from `additional_tools`) are included as-is
/// 3. Both lists are merged into a single `tools_available` list
/// 4. The distinction between static and dynamic tools is lost
/// 5. `provider_tools` are dropped (not stored)
///
/// Use `FunctionConfig::dynamic_tool_params_to_database_insert()` for this conversion.
///
/// # Conversion from Storage Format
/// Converting from `ToolCallConfigDatabaseInsert` back to `DynamicToolParams` reconstructs the original:
/// 1. `dynamic_tools` → `additional_tools`
/// 2. `allowed_tools` → `allowed_tools` (based on choice enum)
/// 3. Other fields copied directly
///
/// Use `From<ToolCallConfigDatabaseInsert> for DynamicToolParams` for this conversion.
///
/// # Example
/// ```rust,ignore
/// // API request with dynamic tool params
/// let params = DynamicToolParams {
///     allowed_tools: Some(vec!["calculator".to_string()]),  // Use only the calculator tool from config
///     additional_tools: Some(vec![Tool {  runtime tool  }]),  // Add a new tool
///     tool_choice: Some(ToolChoice::Required),
///     parallel_tool_calls: Some(true),
///     provider_tools: vec![],
/// };
///
/// // Convert to storage format
/// let db_insert = function_config
///     .dynamic_tool_params_to_database_insert(params, &static_tools)?
///     .unwrap_or_default();
///
/// // db_insert.tools_available now contains both the calculator tool (from config)
/// // and the runtime tool (from additional_tools), merged together
/// ```
///
/// See also: [`ToolCallConfigDatabaseInsert`] for the storage/database format
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize, ts_rs::TS, JsonSchema)]
#[serde(deny_unknown_fields)]
#[ts(optional_fields, export)]
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
    pub fn provider_tools(&self) -> Vec<ProviderTool> {
        self.provider_tools.clone()
    }

    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}

impl From<ToolCallConfigDatabaseInsert> for DynamicToolParams {
    fn from(db_insert: ToolCallConfigDatabaseInsert) -> Self {
        let ToolCallConfigDatabaseInsert {
            dynamic_tools,
            dynamic_provider_tools,
            allowed_tools,
            tool_choice,
            parallel_tool_calls,
            .. // TODO: Ideally we can say all but private fields must be destructured here.
        } = db_insert;

        let allowed_tools = match allowed_tools.choice {
            AllowedToolsChoice::FunctionDefault => None,
            // We leave this in because historical data may have been written in this format
            #[expect(deprecated)]
            AllowedToolsChoice::DynamicAllowedTools => Some(allowed_tools.tools),
            AllowedToolsChoice::Explicit => Some(allowed_tools.tools),
        };

        let additional_tools = if dynamic_tools.is_empty() {
            None
        } else {
            Some(dynamic_tools)
        };

        DynamicToolParams {
            allowed_tools,
            additional_tools,
            tool_choice: Some(tool_choice),
            parallel_tool_calls,
            provider_tools: dynamic_provider_tools,
        }
    }
}

#[derive(Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct BatchDynamicToolParams {
    pub allowed_tools: Option<Vec<Option<Vec<String>>>>,
    pub additional_tools: Option<Vec<Option<Vec<Tool>>>>,
    pub tool_choice: Option<Vec<Option<ToolChoice>>>,
    pub parallel_tool_calls: Option<Vec<Option<bool>>>,
    pub provider_tools: Option<Vec<Option<Vec<ProviderTool>>>>,
}

// Helper type for converting BatchDynamicToolParams into a Vec<DynamicToolParams>
pub struct BatchDynamicToolParamsWithSize(pub BatchDynamicToolParams, pub usize);

/// In most cases, tool call arguments are a string.
/// However, when looping back from an inference response, they will be an object.
fn deserialize_tool_call_arguments<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::String(s) => Ok(s),
        serde_json::Value::Object(_) => Ok(value.to_string()),
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
#[serde(untagged)]
#[export_schema]
pub enum ToolCallWrapper {
    ToolCall(ToolCall), // the format we store in the database
    InferenceResponseToolCall(InferenceResponseToolCall), // the format we send on an inference response
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

/// An InferenceResponseToolCall is a request by a model to call a Tool
/// in the form that we return to the client / ClickHouse
/// This includes some synactic sugar (parsing / validation of the tool arguments)
/// in the `arguments` field and the name in the `name` field.
/// We support looping this back through the TensorZero inference API via the ToolCallWrapper
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, ts_rs::TS, JsonSchema)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[export_schema]
pub struct InferenceResponseToolCall {
    /// A Tool Call ID to match up with tool call responses. See #4058.
    pub id: String,

    /// The name of the tool to call, as generated by the model.
    pub raw_name: String,

    /// The raw arguments JSON string of the tool to call, as generated by the model.
    pub raw_arguments: String,

    /// The name of the tool to call, validated against tool configs. If not present, it means the tool call was invalid.
    pub name: Option<String>,

    /// The arguments of the tool to call, validated against tool configs. If not present, it means the tool call arguments were invalid.
    pub arguments: Option<Value>,
}

impl std::fmt::Display for InferenceResponseToolCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl InferenceResponseToolCall {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}

impl InferenceResponseToolCall {
    /// Validates that a ToolCall is compliant with the ToolCallConfig
    /// First, it finds the ToolConfig for the ToolCall
    /// Then, it validates the ToolCall arguments against the ToolConfig
    pub async fn new(tool_call: ToolCall, tool_cfg: Option<&ToolCallConfig>) -> Self {
        // Check if this is a function tool
        let function_tool = tool_cfg.and_then(|t| t.get_function_tool(&tool_call.name));

        // Check if this is a custom tool
        let is_custom_tool = tool_cfg
            .map(|t| {
                t.openai_custom_tools
                    .iter()
                    .any(|ct| ct.name == tool_call.name)
            })
            .unwrap_or(false);

        // Set parsed_name if tool exists (either function or custom)
        let parsed_name = if function_tool.is_some() || is_custom_tool {
            Some(tool_call.name.clone())
        } else {
            None
        };

        // Validate arguments only for function tools (custom tools don't use JSON schemas)
        let parsed_arguments = if let Some(tool) = function_tool {
            if let Ok(arguments) = serde_json::from_str(&tool_call.arguments) {
                if tool.validate_arguments(&arguments).await.is_ok() {
                    Some(arguments)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        Self {
            arguments: parsed_arguments,
            id: tool_call.id,
            name: parsed_name,
            raw_arguments: tool_call.arguments.clone(),
            raw_name: tool_call.name.clone(),
        }
    }
}

impl ToolCallConfig {
    #[cfg(test)]
    #[expect(clippy::missing_panics_doc)]
    pub fn implicit_from_value(value: &Value) -> Self {
        let parameters = StaticJSONSchema::from_value(value.clone()).unwrap();
        let implicit_tool_config = FunctionToolConfig::Implicit(ImplicitToolConfig { parameters });
        Self {
            static_tools_available: vec![implicit_tool_config],
            dynamic_tools_available: vec![],
            openai_custom_tools: vec![],
            tool_choice: ToolChoice::Specific(IMPLICIT_TOOL_NAME.to_string()),
            parallel_tool_calls: None,
            provider_tools: vec![],
            allowed_tools: AllowedTools::default(),
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

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolCallChunk {
    pub id: String,
    #[serde(serialize_with = "serialize_option_string_as_empty")]
    pub raw_name: Option<String>,
    pub raw_arguments: String,
}

fn serialize_option_string_as_empty<S>(
    value: &Option<String>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match value {
        Some(s) => serializer.serialize_str(s),
        None => serializer.serialize_str(""),
    }
}

pub const IMPLICIT_TOOL_NAME: &str = "respond";
pub const IMPLICIT_TOOL_DESCRIPTION: &str = "Respond to the user using the output schema provided.";

impl FunctionToolConfig {
    pub async fn validate_arguments(&self, arguments: &Value) -> Result<(), Error> {
        match self {
            FunctionToolConfig::Static(config) => config.parameters.validate(arguments),
            FunctionToolConfig::Dynamic(config) => config.parameters.validate(arguments).await,
            FunctionToolConfig::Implicit(config) => config.parameters.validate(arguments),
            FunctionToolConfig::DynamicImplicit(config) => {
                config.parameters.validate(arguments).await
            }
        }
    }

    pub fn description(&self) -> &str {
        match self {
            FunctionToolConfig::Static(config) => &config.description,
            FunctionToolConfig::Dynamic(config) => &config.description,
            FunctionToolConfig::Implicit(_config) => IMPLICIT_TOOL_DESCRIPTION,
            FunctionToolConfig::DynamicImplicit(_config) => IMPLICIT_TOOL_DESCRIPTION,
        }
    }

    pub fn parameters(&self) -> &Value {
        match self {
            FunctionToolConfig::Static(config) => &config.parameters.value,
            FunctionToolConfig::Dynamic(config) => &config.parameters.value,
            FunctionToolConfig::Implicit(config) => &config.parameters.value,
            FunctionToolConfig::DynamicImplicit(config) => &config.parameters.value,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            FunctionToolConfig::Static(config) => &config.name,
            FunctionToolConfig::Dynamic(config) => &config.name,
            FunctionToolConfig::Implicit(_config) => IMPLICIT_TOOL_NAME,
            FunctionToolConfig::DynamicImplicit(_config) => IMPLICIT_TOOL_NAME,
        }
    }

    pub fn strict(&self) -> bool {
        match self {
            FunctionToolConfig::Static(config) => config.strict,
            FunctionToolConfig::Dynamic(config) => config.strict,
            FunctionToolConfig::Implicit(_config) => false,
            FunctionToolConfig::DynamicImplicit(_config) => false,
        }
    }
}

fn tool_call_config_to_legacy_tool_database_insert(
    tool_call_config: &ToolCallConfig,
) -> LegacyToolCallConfigDatabaseInsert {
    LegacyToolCallConfigDatabaseInsert {
        tools_available: tool_call_config
            .static_tools_available
            .iter()
            .chain(tool_call_config.dynamic_tools_available.iter())
            .cloned()
            .map(FunctionToolConfig::into)
            .collect(),
        tool_choice: tool_call_config.tool_choice.clone(),
        parallel_tool_calls: tool_call_config.parallel_tool_calls,
    }
}

// For now, this is required to convert to LegacyToolCallConfigDatabaseInsert for writing to the databse
impl From<FunctionToolConfig> for FunctionTool {
    fn from(tool_config: FunctionToolConfig) -> Self {
        FunctionTool {
            description: tool_config.description().to_string(),
            parameters: tool_config.parameters().clone(),
            name: tool_config.name().to_string(),
            strict: tool_config.strict(),
        }
    }
}

impl From<ToolCallConfig> for ToolCallConfigDatabaseInsert {
    fn from(tool_call_config: ToolCallConfig) -> Self {
        let legacy_config = tool_call_config_to_legacy_tool_database_insert(&tool_call_config);
        let ToolCallConfig {
            // We explicitly don't store static_tools_available in the new ToolCallConfigDatabaseInsert
            // because we want these to be specified by the function and eventually function version.
            static_tools_available: _,
            dynamic_tools_available,
            openai_custom_tools,
            allowed_tools,
            tool_choice,
            parallel_tool_calls,
            provider_tools,
        } = tool_call_config;
        Self {
            tool_params: legacy_config,
            dynamic_tools: dynamic_tools_available
                .into_iter()
                .map(Tool::from)
                .chain(openai_custom_tools.into_iter().map(Tool::OpenAICustom))
                .collect(),
            dynamic_provider_tools: provider_tools,
            allowed_tools,
            tool_choice,
            parallel_tool_calls,
        }
    }
}

impl From<ToolConfig> for Tool {
    fn from(tool_config: ToolConfig) -> Self {
        match tool_config {
            ToolConfig::OpenAICustom(config) => Tool::OpenAICustom(config),
            ToolConfig::Function(tool) => tool.into(),
        }
    }
}

impl From<FunctionToolConfig> for Tool {
    fn from(tool_config: FunctionToolConfig) -> Self {
        Self::Function(FunctionTool {
            description: tool_config.description().to_string(),
            parameters: tool_config.parameters().clone(),
            name: tool_config.name().to_string(),
            strict: tool_config.strict(),
        })
    }
}

pub fn create_dynamic_implicit_tool_config(schema: Value) -> ToolCallConfig {
    let tool_schema = DynamicJSONSchema::new(schema);
    let implicit_tool = FunctionToolConfig::DynamicImplicit(DynamicImplicitToolConfig {
        parameters: tool_schema,
    });
    ToolCallConfig {
        static_tools_available: vec![],
        dynamic_tools_available: vec![implicit_tool],
        openai_custom_tools: vec![],
        tool_choice: ToolChoice::Specific(IMPLICIT_TOOL_NAME.to_string()),
        parallel_tool_calls: None,
        provider_tools: vec![],
        allowed_tools: AllowedTools::default(),
    }
}

impl TryFrom<BatchDynamicToolParamsWithSize> for Vec<DynamicToolParams> {
    type Error = Error;

    fn try_from(
        batch_dynamic_tool_params_with_size: BatchDynamicToolParamsWithSize,
    ) -> Result<Self, Self::Error> {
        let BatchDynamicToolParamsWithSize(batch_dynamic_tool_params, num_inferences) =
            batch_dynamic_tool_params_with_size;
        if num_inferences == 0 {
            return Ok(vec![
                DynamicToolParams {
                    allowed_tools: None,
                    additional_tools: None,
                    tool_choice: None,
                    parallel_tool_calls: None,
                    provider_tools: vec![],
                };
                num_inferences
            ]);
        }
        let BatchDynamicToolParams {
            allowed_tools,
            additional_tools,
            tool_choice,
            parallel_tool_calls,
            provider_tools,
        } = batch_dynamic_tool_params;

        // Verify all provided Vecs have the same length
        if let Some(allowed_tools) = &allowed_tools {
            if allowed_tools.len() != num_inferences {
                return Err(ErrorDetails::InvalidRequest {
                    message: format!(
                        "allowed_tools vector length ({}) does not match number of inferences ({})",
                        allowed_tools.len(),
                        num_inferences
                    ),
                }
                .into());
            }
        }
        if let Some(additional_tools) = &additional_tools {
            if additional_tools.len() != num_inferences {
                return Err(ErrorDetails::InvalidRequest {
                    message: format!(
                        "additional_tools vector length ({}) does not match number of inferences ({})",
                        additional_tools.len(),
                        num_inferences
                    ),
                }
                .into());
            }
        }
        if let Some(tool_choice) = &tool_choice {
            if tool_choice.len() != num_inferences {
                return Err(ErrorDetails::InvalidRequest {
                    message: format!(
                        "tool_choice vector length ({}) does not match number of inferences ({})",
                        tool_choice.len(),
                        num_inferences
                    ),
                }
                .into());
            }
        }
        if let Some(parallel_tool_calls) = &parallel_tool_calls {
            if parallel_tool_calls.len() != num_inferences {
                return Err(ErrorDetails::InvalidRequest {
                    message: format!(
                        "parallel_tool_calls vector length ({}) does not match number of inferences ({})",
                        parallel_tool_calls.len(),
                        num_inferences
                    ),
                }
                .into());
            }
        }
        if let Some(provider_tools) = &provider_tools {
            if provider_tools.len() != num_inferences {
                return Err(ErrorDetails::InvalidRequest {
                    message: format!(
                        "provider_tools vector length ({}) does not match number of inferences ({})",
                        provider_tools.len(),
                        num_inferences
                    )
                }.into());
            }
        }
        // Convert Option<Vec<Option<T>>> into Vec<Option<T>> by unwrapping or creating empty vec
        let allowed_tools = allowed_tools.unwrap_or_default();
        let additional_tools = additional_tools.unwrap_or_default();
        let tool_choice = tool_choice.unwrap_or_default();
        let parallel_tool_calls = parallel_tool_calls.unwrap_or_default();
        let provider_tools = provider_tools.unwrap_or_default();

        // Create iterators that take ownership
        let mut allowed_tools_iter = allowed_tools.into_iter();
        let mut additional_tools_iter = additional_tools.into_iter();
        let mut tool_choice_iter = tool_choice.into_iter();
        let mut parallel_tool_calls_iter = parallel_tool_calls.into_iter();
        let mut provider_tools_iter = provider_tools.into_iter();

        // Build params using the iterators
        let mut all_dynamic_tool_params = Vec::with_capacity(num_inferences);
        // Since we already verified that the vectors that were Some were the same length,
        // it is safe to do .next().unwrap_or() since we'll either be taking real elements or using an empty vector.
        for _ in 0..num_inferences {
            all_dynamic_tool_params.push(DynamicToolParams {
                allowed_tools: allowed_tools_iter.next().unwrap_or(None),
                additional_tools: additional_tools_iter.next().unwrap_or(None),
                tool_choice: tool_choice_iter.next().unwrap_or(None),
                parallel_tool_calls: parallel_tool_calls_iter.next().unwrap_or(None),
                provider_tools: provider_tools_iter.next().flatten().unwrap_or(vec![]),
            });
        }
        Ok(all_dynamic_tool_params)
    }
}

/// For use in initializing JSON functions
/// Creates a ToolCallConfig with a single implicit tool that takes the schema as arguments
pub fn create_json_mode_tool_call_config(schema: StaticJSONSchema) -> ToolCallConfig {
    create_json_mode_tool_call_config_with_allowed_tools(schema, AllowedTools::default())
}

pub fn create_json_mode_tool_call_config_with_allowed_tools(
    schema: StaticJSONSchema,
    allowed_tools: AllowedTools,
) -> ToolCallConfig {
    let implicit_tool = FunctionToolConfig::Implicit(ImplicitToolConfig { parameters: schema });
    ToolCallConfig {
        static_tools_available: vec![implicit_tool],
        dynamic_tools_available: vec![],
        tool_choice: ToolChoice::Specific(IMPLICIT_TOOL_NAME.to_string()),
        openai_custom_tools: vec![],
        parallel_tool_calls: None,
        provider_tools: vec![],
        allowed_tools,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lazy_static::lazy_static;
    use serde_json::json;
    lazy_static! {
        static ref TOOLS: HashMap<String, Arc<StaticToolConfig>> = {
            let mut map = HashMap::new();
            map.insert(
                "get_temperature".to_string(),
                Arc::new(StaticToolConfig {
                    name: "get_temperature".to_string(),
                    description: "Get the current temperature in a given location".to_string(),
                    parameters: StaticJSONSchema::from_value(json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                        "required": ["location"]
                    }))
                    .expect("Failed to create schema for get_temperature"),
                    strict: true,
                }),
            );
            map.insert(
                "query_articles".to_string(),
                Arc::new(StaticToolConfig {
                    name: "query_articles".to_string(),
                    description: "Query articles from a database based on given criteria"
                        .to_string(),
                    parameters: StaticJSONSchema::from_value(json!({
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string"},
                            "category": {"type": "string"},
                            "limit": {"type": "integer", "minimum": 1, "maximum": 100}
                        },
                        "required": ["keyword"]
                    }))
                    .expect("Failed to create schema for query_articles"),
                    strict: false,
                }),
            );
            map
        };
        static ref EMPTY_TOOLS: HashMap<String, Arc<StaticToolConfig>> = HashMap::new();
        static ref EMPTY_FUNCTION_TOOLS: Vec<String> = vec![];
        static ref ALL_FUNCTION_TOOLS: Vec<String> =
            vec!["get_temperature".to_string(), "query_articles".to_string()];
        static ref AUTO_TOOL_CHOICE: ToolChoice = ToolChoice::Auto;
        static ref WEATHER_TOOL_CHOICE: ToolChoice =
            ToolChoice::Specific("get_temperature".to_string());
    }

    #[tokio::test]
    async fn test_tool_call_config_new() {
        // Empty tools in function, no dynamic tools, tools are configured in the config
        // This should return no tools because the function does not specify any tools
        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &EMPTY_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            DynamicToolParams::default(),
        ))
        .unwrap();
        assert!(tool_call_config.is_none());

        // All tools available, no dynamic tools, tools are configured in the config
        // This should return all tools because the function specifies all tools
        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            DynamicToolParams::default(),
        ))
        .unwrap()
        .unwrap();
        assert_eq!(tool_call_config.tools_available().unwrap().count(), 2);

        // strict_tools_available should return all tools (FunctionDefault mode)
        assert_eq!(
            tool_call_config.strict_tools_available().unwrap().count(),
            2
        );
        assert!(matches!(
            tool_call_config.allowed_tools.choice,
            AllowedToolsChoice::FunctionDefault
        ));
        assert_eq!(tool_call_config.tool_choice, ToolChoice::Auto);
        assert_eq!(tool_call_config.parallel_tool_calls, Some(true));
        let tools: Vec<_> = tool_call_config.tools_available().unwrap().collect();
        assert!(tools[0].strict());
        assert!(!tools[1].strict());

        // Empty tools in function and config but we specify an allowed tool (should fail)
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec!["get_temperature".to_string()]),
            ..Default::default()
        };
        let err = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &EMPTY_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &EMPTY_TOOLS,
            dynamic_tool_params,
        ))
        .unwrap_err();
        assert_eq!(
            err,
            ErrorDetails::ToolNotFound {
                name: "get_temperature".to_string()
            }
            .into()
        );

        // Dynamic tool config specifies a particular tool to call and it's in the function tools list
        let dynamic_tool_params = DynamicToolParams {
            tool_choice: Some(ToolChoice::Specific("get_temperature".to_string())),
            ..Default::default()
        };
        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        ))
        .unwrap()
        .unwrap();
        assert_eq!(tool_call_config.tools_available().unwrap().count(), 2);
        assert_eq!(
            tool_call_config.tool_choice,
            ToolChoice::Specific("get_temperature".to_string())
        );
        assert_eq!(tool_call_config.parallel_tool_calls, Some(true));

        // Dynamic tool config specifies a particular tool to call and it's not in the function tools list
        let dynamic_tool_params = DynamicToolParams {
            tool_choice: Some(ToolChoice::Specific("establish_campground".to_string())),
            ..Default::default()
        };
        let err = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        ))
        .unwrap_err();
        assert_eq!(
            err,
            ErrorDetails::ToolNotFound {
                name: "establish_campground".to_string()
            }
            .into()
        );

        // We pass an empty list of allowed tools and then configure a new tool
        // All function tools are still included, plus the dynamic tool
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec![]),
            additional_tools: Some(vec![Tool::Function(FunctionTool {
                name: "establish_campground".to_string(),
                description: "Establish a campground".to_string(),
                parameters: json!({}),
                strict: false,
            })]),
            ..Default::default()
        };
        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        ))
        .unwrap()
        .unwrap();
        // Should have all function tools (get_temperature, query_articles) + dynamic tool (establish_campground)
        assert_eq!(tool_call_config.tools_available().unwrap().count(), 3);
        assert!(tool_call_config
            .tools_available()
            .unwrap()
            .any(|t| t.name() == "get_temperature"));
        assert!(tool_call_config
            .tools_available()
            .unwrap()
            .any(|t| t.name() == "query_articles"));
        assert!(tool_call_config
            .tools_available()
            .unwrap()
            .any(|t| t.name() == "establish_campground"));

        // We pass a list of a single allowed tool and then configure a new tool
        // All function tools are still included, plus the dynamic tool
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec!["get_temperature".to_string()]),
            additional_tools: Some(vec![Tool::Function(FunctionTool {
                name: "establish_campground".to_string(),
                description: "Establish a campground".to_string(),
                parameters: json!({}),
                strict: false,
            })]),
            parallel_tool_calls: Some(false),
            ..Default::default()
        };
        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        ))
        .unwrap()
        .unwrap();
        // Should have all function tools + dynamic tool
        assert_eq!(tool_call_config.tools_available().unwrap().count(), 3);
        assert!(tool_call_config
            .tools_available()
            .unwrap()
            .any(|t| t.name() == "get_temperature"));
        assert!(tool_call_config
            .tools_available()
            .unwrap()
            .any(|t| t.name() == "query_articles"));
        assert!(tool_call_config
            .tools_available()
            .unwrap()
            .any(|t| t.name() == "establish_campground"));
        assert_eq!(tool_call_config.parallel_tool_calls, Some(false));

        // We pass a list of no allowed tools and then configure a new tool
        // All function tools are still included, plus the dynamic tool
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec![]),
            additional_tools: Some(vec![Tool::Function(FunctionTool {
                name: "establish_campground".to_string(),
                description: "Establish a campground".to_string(),
                parameters: json!({}),
                strict: false,
            })]),
            tool_choice: Some(ToolChoice::Specific("establish_campground".to_string())),
            ..Default::default()
        };
        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        ))
        .unwrap()
        .unwrap();
        // Should have all function tools + dynamic tool
        assert_eq!(tool_call_config.tools_available().unwrap().count(), 3);
        assert!(tool_call_config
            .tools_available()
            .unwrap()
            .any(|t| t.name() == "establish_campground"));
        assert_eq!(tool_call_config.parallel_tool_calls, Some(true));
        assert_eq!(
            tool_call_config.tool_choice,
            ToolChoice::Specific("establish_campground".to_string())
        );
    }

    #[tokio::test]
    async fn test_inference_response_tool_call_new() {
        let tool_call = ToolCall {
            name: "get_temperature".to_string(),
            arguments: "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}".to_string(),
            id: "123".to_string(),
        };
        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            DynamicToolParams::default(),
        ))
        .unwrap()
        .unwrap();
        // Tool call is valid, so we should get a valid InferenceResponseToolCall
        let inference_response_tool_call =
            InferenceResponseToolCall::new(tool_call, Some(&tool_call_config)).await;
        assert_eq!(inference_response_tool_call.raw_name, "get_temperature");
        assert_eq!(
            inference_response_tool_call.raw_arguments,
            "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}"
        );
        assert_eq!(inference_response_tool_call.id, "123");
        assert_eq!(
            inference_response_tool_call.name,
            Some("get_temperature".to_string())
        );
        assert_eq!(
            inference_response_tool_call.arguments,
            Some(json!({
                "location": "San Francisco",
                "unit": "celsius"
            }))
        );

        // Bad arguments, but valid name (parsed_name is set but parsed_arguments is not)
        let tool_call = ToolCall {
            name: "get_temperature".to_string(),
            arguments: "{\"location\": \"San Francisco\", \"unit\": \"kelvin\"}".to_string(),
            id: "321".to_string(),
        };
        let inference_response_tool_call =
            InferenceResponseToolCall::new(tool_call, Some(&tool_call_config)).await;
        assert_eq!(
            inference_response_tool_call.name,
            Some("get_temperature".to_string())
        );
        assert_eq!(inference_response_tool_call.arguments, None);
        assert_eq!(inference_response_tool_call.id, "321");
        assert_eq!(inference_response_tool_call.raw_name, "get_temperature");
        assert_eq!(
            inference_response_tool_call.raw_arguments,
            "{\"location\": \"San Francisco\", \"unit\": \"kelvin\"}"
        );

        // Bad name, good arguments (both not set since the name is invalid and we can't be sure what tool this goes to)
        let tool_call = ToolCall {
            name: "not_get_weather".to_string(),
            arguments: "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}".to_string(),
            id: "321".to_string(),
        };
        let inference_response_tool_call =
            InferenceResponseToolCall::new(tool_call, Some(&tool_call_config)).await;
        assert_eq!(inference_response_tool_call.name, None);
        assert_eq!(inference_response_tool_call.arguments, None);
        assert_eq!(inference_response_tool_call.id, "321");
        assert_eq!(inference_response_tool_call.raw_name, "not_get_weather");
        assert_eq!(
            inference_response_tool_call.raw_arguments,
            "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}"
        );

        // Make sure validation works with dynamic tools
        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            DynamicToolParams {
                additional_tools: Some(vec![Tool::Function(FunctionTool {
                    name: "establish_campground".to_string(),
                    description: "Establish a campground".to_string(),
                    parameters: json!({"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}),
                    strict: false,
                })]),
                ..Default::default()
            },
        ))
        .unwrap()
        .unwrap();
        let tool_call = ToolCall {
            name: "establish_campground".to_string(),
            arguments: "{\"location\": \"Lucky Dog\"}".to_string(),
            id: "321".to_string(),
        };
        let inference_response_tool_call =
            InferenceResponseToolCall::new(tool_call, Some(&tool_call_config)).await;
        assert_eq!(
            inference_response_tool_call.raw_name,
            "establish_campground"
        );
        assert_eq!(
            inference_response_tool_call.raw_arguments,
            "{\"location\": \"Lucky Dog\"}"
        );
        assert_eq!(inference_response_tool_call.id, "321");
        assert_eq!(
            inference_response_tool_call.name,
            Some("establish_campground".to_string())
        );
        assert_eq!(
            inference_response_tool_call.arguments,
            Some(json!({"location": "Lucky Dog"}))
        );
    }

    #[tokio::test]
    async fn test_inference_response_tool_call_with_custom_tools() {
        // Create a ToolCallConfig with a custom tool
        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &EMPTY_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &EMPTY_TOOLS,
            DynamicToolParams {
                additional_tools: Some(vec![Tool::OpenAICustom(OpenAICustomTool {
                    name: "code_generator".to_string(),
                    description: Some("Generates code snippets".to_string()),
                    format: Some(OpenAICustomToolFormat::Text),
                })]),
                ..Default::default()
            },
        ))
        .unwrap()
        .unwrap();

        // Valid custom tool call - name should be validated
        let tool_call = ToolCall {
            name: "code_generator".to_string(),
            arguments: "{\"description\": \"Print hello world\"}".to_string(),
            id: "ctc_123".to_string(),
        };
        let inference_response_tool_call =
            InferenceResponseToolCall::new(tool_call, Some(&tool_call_config)).await;

        // The parsed_name should be set since this is a valid custom tool
        assert_eq!(
            inference_response_tool_call.name,
            Some("code_generator".to_string())
        );
        assert_eq!(inference_response_tool_call.raw_name, "code_generator");
        assert_eq!(
            inference_response_tool_call.raw_arguments,
            "{\"description\": \"Print hello world\"}"
        );
        assert_eq!(inference_response_tool_call.id, "ctc_123");
        // Custom tools don't validate arguments against JSON schemas, so parsed_arguments should be None
        assert_eq!(inference_response_tool_call.arguments, None);

        // Invalid custom tool name - name should not be validated
        let tool_call = ToolCall {
            name: "not_a_custom_tool".to_string(),
            arguments: "{\"description\": \"Test\"}".to_string(),
            id: "ctc_456".to_string(),
        };
        let inference_response_tool_call =
            InferenceResponseToolCall::new(tool_call, Some(&tool_call_config)).await;

        // The parsed_name should be None since this tool doesn't exist
        assert_eq!(inference_response_tool_call.name, None);
        assert_eq!(inference_response_tool_call.raw_name, "not_a_custom_tool");
        assert_eq!(
            inference_response_tool_call.raw_arguments,
            "{\"description\": \"Test\"}"
        );
        assert_eq!(inference_response_tool_call.id, "ctc_456");
        assert_eq!(inference_response_tool_call.arguments, None);

        // Test with both function tools and custom tools
        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            DynamicToolParams {
                additional_tools: Some(vec![Tool::OpenAICustom(OpenAICustomTool {
                    name: "calculator".to_string(),
                    description: Some("Performs calculations".to_string()),
                    format: Some(OpenAICustomToolFormat::Grammar {
                        grammar: OpenAIGrammarDefinition {
                            syntax: OpenAIGrammarSyntax::Lark,
                            definition: "start: NUMBER".to_string(),
                        },
                    }),
                })]),
                ..Default::default()
            },
        ))
        .unwrap()
        .unwrap();

        // Valid function tool should still work
        let tool_call = ToolCall {
            name: "get_temperature".to_string(),
            arguments: "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}".to_string(),
            id: "123".to_string(),
        };
        let inference_response_tool_call =
            InferenceResponseToolCall::new(tool_call, Some(&tool_call_config)).await;
        assert_eq!(
            inference_response_tool_call.name,
            Some("get_temperature".to_string())
        );
        assert_eq!(
            inference_response_tool_call.arguments,
            Some(json!({
                "location": "San Francisco",
                "unit": "celsius"
            }))
        );

        // Valid custom tool should also work
        let tool_call = ToolCall {
            name: "calculator".to_string(),
            arguments: "42".to_string(),
            id: "ctc_789".to_string(),
        };
        let inference_response_tool_call =
            InferenceResponseToolCall::new(tool_call, Some(&tool_call_config)).await;
        assert_eq!(
            inference_response_tool_call.name,
            Some("calculator".to_string())
        );
        assert_eq!(inference_response_tool_call.raw_name, "calculator");
        assert_eq!(inference_response_tool_call.raw_arguments, "42");
        assert_eq!(inference_response_tool_call.id, "ctc_789");
        // Custom tools don't validate arguments, so parsed_arguments is None
        assert_eq!(inference_response_tool_call.arguments, None);
    }

    #[test]
    fn test_tool_call_deserialize_plain_raw() {
        let tool_call = serde_json::json!({
            "name": "get_temperature",
            "raw_name": "should have ignored raw name",
            "arguments": "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}",
            "raw_arguments": "should have ignored raw arguments",
            "id": "123"
        });
        let tool_call: ToolCall = serde_json::from_value(tool_call).unwrap();
        assert_eq!(tool_call.name, "get_temperature");
        assert_eq!(
            tool_call.arguments,
            "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}"
        );
        assert_eq!(tool_call.id, "123");
    }

    #[test]
    fn test_tool_call_deserialize_raw_only() {
        let tool_call = serde_json::json!({
            "raw_name": "get_temperature",
            "raw_arguments": "my raw arguments",
            "id": "123"
        });
        let tool_call_wrapper: ToolCallWrapper = serde_json::from_value(tool_call).unwrap();
        let tool_call: ToolCall = tool_call_wrapper.try_into().unwrap();
        assert_eq!(tool_call.name, "get_temperature");
        assert_eq!(tool_call.arguments, "my raw arguments");
        assert_eq!(tool_call.id, "123");
    }

    #[test]
    fn test_tool_call_deserialize_arguments_object() {
        let tool_call = serde_json::json!({
            "name": "get_temperature",
            "arguments": {"my": "arguments"},
            "id": "123"
        });
        let tool_call_wrapper = serde_json::from_value::<ToolCallWrapper>(tool_call).unwrap();
        let tool_call = TryInto::<ToolCall>::try_into(tool_call_wrapper).unwrap();
        assert_eq!(tool_call.name, "get_temperature");
        assert_eq!(tool_call.arguments, "{\"my\":\"arguments\"}");
        assert_eq!(tool_call.id, "123");
    }

    #[test]
    fn test_tool_call_deserialize_arguments_string() {
        let tool_call = serde_json::json!({
            "name": "get_temperature",
            "arguments": "{\"my\": \"arguments\"}",
            "id": "123"
        });
        let tool_call: ToolCall = serde_json::from_value(tool_call).unwrap();
        assert_eq!(tool_call.name, "get_temperature");
        assert_eq!(tool_call.arguments, "{\"my\": \"arguments\"}");
        assert_eq!(tool_call.id, "123");
    }

    #[test]
    fn test_tool_call_deserialize_missing_name() {
        let tool_call = serde_json::json!({
            "arguments": "{\"my\": \"arguments\"}",
            "id": "123"
        });
        // Now we get an ugly error because of the untagged enum, but that's ok for now...
        // https://github.com/tensorzero/tensorzero/discussions/4258
        serde_json::from_value::<ToolCallWrapper>(tool_call).unwrap_err();
    }

    #[test]
    fn test_tool_call_deserialize_missing_arguments() {
        let tool_call = serde_json::json!({
            "name": "get_temperature",
            "id": "123"
        });
        let err_msg = serde_json::from_value::<ToolCall>(tool_call)
            .unwrap_err()
            .to_string();
        assert_eq!(err_msg, "missing field `arguments`");
    }

    #[test]
    fn test_tool_call_deserialize_object_arguments() {
        let tool_call = serde_json::json!({
            "name": "get_temperature",
            "id": "123",
            "arguments": {
                "role": "intern"
            }
        });
        let tool_call_wrapper = serde_json::from_value::<ToolCallWrapper>(tool_call).unwrap();
        let tool_call = TryInto::<ToolCall>::try_into(tool_call_wrapper).unwrap();
        assert_eq!(tool_call.arguments, "{\"role\":\"intern\"}");
        assert_eq!(tool_call.name, "get_temperature");
        assert_eq!(tool_call.id, "123");
    }

    #[tokio::test]
    async fn test_duplicate_tool_names_error() {
        // Test case where dynamic tool params add a tool with the same name as a static tool
        let dynamic_tool_params = DynamicToolParams {
            additional_tools: Some(vec![Tool::Function(FunctionTool {
                name: "get_temperature".to_string(), // Same name as static tool
                description: "Another temperature tool".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }),
                strict: false,
            })]),
            ..Default::default()
        };

        let err = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        ))
        .unwrap_err();

        assert_eq!(
            err,
            ErrorDetails::DuplicateTool {
                name: "get_temperature".to_string()
            }
            .into()
        );
    }

    #[tokio::test]
    async fn test_duplicate_custom_tool_names_error() {
        // Test case where two custom tools have the same name
        let dynamic_tool_params = DynamicToolParams {
            additional_tools: Some(vec![
                Tool::OpenAICustom(OpenAICustomTool {
                    name: "custom_tool".to_string(),
                    description: Some("First custom tool".to_string()),
                    format: None,
                }),
                Tool::OpenAICustom(OpenAICustomTool {
                    name: "custom_tool".to_string(), // Duplicate name
                    description: Some("Second custom tool".to_string()),
                    format: Some(OpenAICustomToolFormat::Text),
                }),
            ]),
            ..Default::default()
        };

        let err = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &EMPTY_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        ))
        .unwrap_err();

        assert_eq!(
            err,
            ErrorDetails::DuplicateTool {
                name: "custom_tool".to_string()
            }
            .into()
        );
    }

    #[tokio::test]
    async fn test_custom_tool_conflicts_with_function_tool() {
        // Test case where a custom tool has the same name as a function tool
        let dynamic_tool_params = DynamicToolParams {
            additional_tools: Some(vec![Tool::OpenAICustom(OpenAICustomTool {
                name: "get_temperature".to_string(), // Same name as static function tool
                description: Some("Custom temperature tool".to_string()),
                format: None,
            })]),
            ..Default::default()
        };

        let err = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        ))
        .unwrap_err();

        assert_eq!(
            err,
            ErrorDetails::DuplicateTool {
                name: "get_temperature".to_string()
            }
            .into()
        );
    }

    #[test]
    fn test_get_scoped_provider_tools() {
        // Set up provider tools with different scopes
        let provider_tools = vec![
            ProviderTool {
                scope: ProviderToolScope::Unscoped,
                tool: json!({"type": "unscoped_tool"}),
            },
            ProviderTool {
                scope: ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
                    model_name: "gpt-4".to_string(),
                    provider_name: Some("openai".to_string()),
                }),
                tool: json!({"type": "gpt4_tool"}),
            },
            ProviderTool {
                scope: ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
                    model_name: "claude-3".to_string(),
                    provider_name: Some("anthropic".to_string()),
                }),
                tool: json!({"type": "claude_tool"}),
            },
        ];

        let config = ToolCallConfig {
            provider_tools,
            ..Default::default()
        };

        // Test matching gpt-4/openai: should return unscoped + gpt4_tool
        let result = config.get_scoped_provider_tools("gpt-4", "openai");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].tool, json!({"type": "unscoped_tool"}));
        assert_eq!(result[1].tool, json!({"type": "gpt4_tool"}));

        // Test matching claude-3/anthropic: should return unscoped + claude_tool
        let result = config.get_scoped_provider_tools("claude-3", "anthropic");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].tool, json!({"type": "unscoped_tool"}));
        assert_eq!(result[1].tool, json!({"type": "claude_tool"}));

        // Test non-matching model: should return only unscoped
        let result = config.get_scoped_provider_tools("llama-2", "meta");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].tool, json!({"type": "unscoped_tool"}));

        // Test partial match (correct model, wrong provider): should return only unscoped
        let result = config.get_scoped_provider_tools("gpt-4", "azure");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].tool, json!({"type": "unscoped_tool"}));

        // Test with None provider_tools
        let config_no_tools = ToolCallConfig::with_tools_available(vec![], vec![]);
        let result = config_no_tools.get_scoped_provider_tools("gpt-4", "openai");
        assert_eq!(result.len(), 0);
    }

    #[tokio::test]
    async fn test_dynamic_tool_in_allowed_tools() {
        // Test that a dynamic tool name in allowed_tools is recognized and doesn't error
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec![
                "get_temperature".to_string(),
                "establish_campground".to_string(),
            ]),
            additional_tools: Some(vec![Tool::Function(FunctionTool {
                name: "establish_campground".to_string(),
                description: "Establish a campground".to_string(),
                parameters: json!({"type": "object", "properties": {"location": {"type": "string"}}}),
                strict: false,
            })]),
            ..Default::default()
        };

        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        ))
        .unwrap()
        .unwrap();

        // Should have all function tools plus dynamic tools
        // function_tools: get_temperature, query_articles
        // dynamic tools: establish_campground
        assert_eq!(tool_call_config.tools_available().unwrap().count(), 3);

        // Verify the static tools are included
        assert!(tool_call_config
            .tools_available()
            .unwrap()
            .any(|t| t.name() == "get_temperature"));
        assert!(tool_call_config
            .tools_available()
            .unwrap()
            .any(|t| t.name() == "query_articles"));

        // Verify the dynamic tool is included
        assert!(tool_call_config
            .tools_available()
            .unwrap()
            .any(|t| t.name() == "establish_campground"));

        // strict_tools_available should filter to only allowed_tools (AllAllowedTools mode)
        assert_eq!(
            tool_call_config.strict_tools_available().unwrap().count(),
            2
        );
        assert!(tool_call_config
            .strict_tools_available()
            .unwrap()
            .any(|t| t.name() == "get_temperature"));
        assert!(tool_call_config
            .strict_tools_available()
            .unwrap()
            .any(|t| t.name() == "establish_campground"));
    }

    #[tokio::test]
    async fn test_allowed_tool_not_found_in_static_or_dynamic() {
        // Test that a tool name in allowed_tools that's not in static_tools or additional_tools throws error
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec![
                "get_temperature".to_string(),
                "nonexistent_tool".to_string(),
            ]),
            additional_tools: Some(vec![Tool::Function(FunctionTool {
                name: "establish_campground".to_string(),
                description: "Establish a campground".to_string(),
                parameters: json!({"type": "object"}),
                strict: false,
            })]),
            ..Default::default()
        };

        let err = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        ))
        .unwrap_err();

        assert_eq!(
            err,
            ErrorDetails::ToolNotFound {
                name: "nonexistent_tool".to_string()
            }
            .into()
        );
    }

    #[tokio::test]
    async fn test_dynamic_tool_not_auto_added_to_allowed_tools() {
        // Test that dynamic tools are sent as definitions but not added to allowed_tools
        // when allowed_tools is explicitly set (AllAllowedTools mode)
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec!["get_temperature".to_string()]),
            additional_tools: Some(vec![Tool::Function(FunctionTool {
                name: "establish_campground".to_string(),
                description: "Establish a campground".to_string(),
                parameters: json!({"type": "object", "properties": {"location": {"type": "string"}}}),
                strict: false,
            })]),
            ..Default::default()
        };

        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        ))
        .unwrap()
        .unwrap();

        // All tool definitions should be available (sent to provider)
        // function_tools: get_temperature, query_articles
        // dynamic tools: establish_campground
        assert_eq!(tool_call_config.tools_available().unwrap().count(), 3);
        assert!(tool_call_config
            .tools_available()
            .unwrap()
            .any(|t| t.name() == "get_temperature"));
        assert!(tool_call_config
            .tools_available()
            .unwrap()
            .any(|t| t.name() == "query_articles"));
        assert!(tool_call_config
            .tools_available()
            .unwrap()
            .any(|t| t.name() == "establish_campground"));

        // But only get_temperature should be in allowed_tools
        assert_eq!(tool_call_config.allowed_tools.tools.len(), 1);
        assert!(tool_call_config
            .allowed_tools
            .tools
            .contains(&"get_temperature".to_string()));
        assert!(matches!(
            tool_call_config.allowed_tools.choice,
            AllowedToolsChoice::Explicit
        ));

        // strict_tools_available should filter to only allowed_tools (AllAllowedTools mode)
        assert_eq!(
            tool_call_config.strict_tools_available().unwrap().count(),
            1
        );
        assert!(tool_call_config
            .strict_tools_available()
            .unwrap()
            .any(|t| t.name() == "get_temperature"));
    }

    // Helper struct to test deserialization with flattening
    #[derive(Debug, Deserialize, PartialEq)]
    struct ToolCallConfigDeserializeTestHelper {
        baz: String,
        #[serde(flatten)]
        #[serde(deserialize_with = "deserialize_optional_tool_info")]
        tool_info: Option<ToolCallConfigDatabaseInsert>,
    }

    // Helper function to assert that deserialization results in None for tool_info
    fn assert_deserialize_to_none(json: serde_json::Value, expected_baz: &str) {
        let result: ToolCallConfigDeserializeTestHelper =
            serde_json::from_value(json).expect("Deserialization should succeed");
        assert_eq!(result.baz, expected_baz);
        assert_eq!(result.tool_info, None, "tool_info should be None");
    }

    #[test]
    fn test_tool_call_config_database_insert_deserialize_ragged_with_flatten() {
        // Test with a flattened struct (ragged case)
        // Note: dynamic_tools and dynamic_provider_tools are arrays of JSON strings
        // allowed_tools is a JSON string, tool_choice is a bare string/object
        let json = json!({
            "baz": "test_value",
            "dynamic_tools": [
                r#"{"type":"function","name":"ragged_tool","description":"A ragged tool","parameters":{"type":"string"},"strict":true}"#
            ],
            "dynamic_provider_tools": [],
            "allowed_tools": r#"{"tools":["ragged_tool"],"choice":"function_default"}"#,
            "tool_choice": {"specific": "ragged_tool"},
            "parallel_tool_calls": null,
            "tool_params": {
                "tools_available": [],
                "tool_choice": {"specific": "ragged_tool"},
                "parallel_tool_calls": null
            }
        });

        let result: ToolCallConfigDeserializeTestHelper = serde_json::from_value(json).unwrap();

        assert_eq!(result.baz, "test_value");
        let tool_info = result.tool_info.unwrap();
        assert_eq!(tool_info.dynamic_tools.len(), 1);
        assert_eq!(tool_info.dynamic_tools[0].name(), "ragged_tool");
        assert_eq!(tool_info.dynamic_provider_tools.len(), 0);
        assert_eq!(tool_info.allowed_tools.tools, vec!["ragged_tool"]);
        assert_eq!(
            tool_info.tool_choice,
            ToolChoice::Specific("ragged_tool".to_string())
        );
        assert_eq!(tool_info.parallel_tool_calls, None);
    }

    #[test]
    fn test_tool_call_config_database_insert_deserialize_ragged_legacy() {
        // Test legacy format with flattening
        let json = json!({
            "baz": "legacy_value",
            "tool_params": {
                "tools_available": [
                    {
                        "name": "legacy_ragged_tool",
                        "description": "A legacy ragged tool",
                        "parameters": {"type": "number"},
                        "strict": false
                    }
                ],
                "tool_choice": "none",
                "parallel_tool_calls": true
            }
        });

        let result: ToolCallConfigDeserializeTestHelper = serde_json::from_value(json).unwrap();

        assert_eq!(result.baz, "legacy_value");
        let tool_info = result.tool_info.unwrap();
        assert_eq!(tool_info.dynamic_tools.len(), 0);
        assert_eq!(tool_info.dynamic_provider_tools.len(), 0);
        assert_eq!(tool_info.tool_choice, ToolChoice::None);
        assert_eq!(tool_info.parallel_tool_calls, Some(true));
    }

    #[test]
    fn test_tool_call_config_database_insert_deserialize_ragged_empty() {
        // Test empty format with flattening
        let json = json!({
            "baz": "empty_value"
        });
        assert_deserialize_to_none(json, "empty_value");
    }

    #[test]
    fn test_tool_call_config_database_insert_deserialize_legacy_null_tool_params() {
        // Test legacy format with explicit null tool_params
        // Should return None, same as missing tool_params
        let json = json!({
            "baz": "test_value",
            "tool_params": null
        });
        assert_deserialize_to_none(json, "test_value");
    }

    #[test]
    fn test_tool_call_config_database_insert_deserialize_legacy_empty_tool_params() {
        // Test legacy format with empty string tool_params
        // Should return None
        let json = json!({
            "baz": "test_value",
            "tool_params": ""
        });
        assert_deserialize_to_none(json, "test_value");
    }

    #[test]
    fn test_tool_call_config_database_insert_deserialize_legacy_missing_vs_null() {
        // Test that missing tool_params behaves the same as null tool_params
        // Both should return None for tool_info

        // Missing tool_params
        let json_missing = json!({
            "baz": "test_missing"
        });
        assert_deserialize_to_none(json_missing, "test_missing");

        // Null tool_params
        let json_null = json!({
            "baz": "test_null",
            "tool_params": null
        });
        assert_deserialize_to_none(json_null, "test_null");
    }

    #[test]
    fn test_tool_call_config_database_insert_deserialize_invalid_tool_type() {
        // Test with an invalid tool type
        let json = json!({
            "baz": "test",
            "dynamic_tools": [
                r#"{"type":"invalid_type","name":"test_tool","description":"A test tool","parameters":{"type":"object"}}"#
            ],
            "dynamic_provider_tools": [],
            "allowed_tools": r#"{"tools":["test_tool"],"choice":"function_default"}"#,
            "tool_choice": r#""auto""#,
            "tool_params": {
                "tools_available": [],
                "tool_choice": "auto"
            }
        });

        let result: Result<ToolCallConfigDeserializeTestHelper, _> = serde_json::from_value(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_call_config_database_insert_deserialize_missing_tool_name() {
        // Test with missing required tool name field
        let json = json!({
            "baz": "test",
            "dynamic_tools": [
                r#"{"type":"function","description":"A test tool","parameters":{"type":"object"}}"#
            ],
            "dynamic_provider_tools": [],
            "allowed_tools": r#"{"tools":["test_tool"],"choice":"function_default"}"#,
            "tool_choice": r#""auto""#,
            "tool_params": {
                "tools_available": [],
                "tool_choice": "auto"
            }
        });

        let result: Result<ToolCallConfigDeserializeTestHelper, _> = serde_json::from_value(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_call_config_database_insert_deserialize_invalid_tool_choice() {
        // Test with invalid tool_choice enum value
        let json = json!({
            "baz": "test",
            "dynamic_tools": [],
            "dynamic_provider_tools": [],
            "allowed_tools": r#"{"tools":[],"choice":"function_default"}"#,
            "tool_choice": r#""invalid_choice""#,
            "tool_params": {
                "tools_available": [],
                "tool_choice": "auto"
            }
        });

        let result: Result<ToolCallConfigDeserializeTestHelper, _> = serde_json::from_value(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_call_config_database_insert_deserialize_invalid_allowed_tools_choice() {
        // Test with invalid allowed_tools.choice enum value
        let json = json!({
            "baz": "test",
            "dynamic_tools": [],
            "dynamic_provider_tools": [],
            "allowed_tools": r#"{"tools":[],"choice":"invalid_choice"}"#,
            "tool_choice": r#""auto""#,
            "tool_params": {
                "tools_available": [],
                "tool_choice": "auto"
            }
        });

        let result: Result<ToolCallConfigDeserializeTestHelper, _> = serde_json::from_value(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_call_config_database_insert_deserialize_wrong_type_for_tools() {
        // Test with wrong type for dynamic_tools (string instead of array)
        let json = json!({
            "baz": "test",
            "dynamic_tools": "not_an_array",
            "dynamic_provider_tools": [],
            "allowed_tools": r#"{"tools":[],"choice":"function_default"}"#,
            "tool_choice": r#""auto""#,
            "tool_params": {
                "tools_available": [],
                "tool_choice": "auto"
            }
        });

        let result: Result<ToolCallConfigDeserializeTestHelper, _> = serde_json::from_value(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_call_config_database_insert_deserialize_wrong_type_for_parallel_tool_calls() {
        // Test with wrong type for parallel_tool_calls (string instead of bool)
        let json = json!({
            "baz": "test",
            "dynamic_tools": [],
            "dynamic_provider_tools": [],
            "allowed_tools": r#"{"tools":[],"choice":"function_default"}"#,
            "tool_choice": r#""auto""#,
            "parallel_tool_calls": "not_a_bool",
            "tool_params": {
                "tools_available": [],
                "tool_choice": "auto"
            }
        });

        let result: Result<ToolCallConfigDeserializeTestHelper, _> = serde_json::from_value(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_call_config_database_insert_deserialize_malformed_provider_tool() {
        // Test with provider tool missing scope field - should use default
        let json = json!({
            "baz": "test",
            "dynamic_tools": [],
            "dynamic_provider_tools": [
                r#"{"tool":{"type":"test"}}"#
                // Missing scope field - should default to Unscoped
            ],
            "allowed_tools": r#"{"tools":[],"choice":"function_default"}"#,
            "tool_choice": "auto",
            "tool_params": {
                "tools_available": [],
                "tool_choice": "auto"
            }
        });

        let result: ToolCallConfigDeserializeTestHelper = serde_json::from_value(json).unwrap();
        assert_eq!(result.baz, "test");
        let tool_info = result.tool_info.unwrap();
        assert_eq!(tool_info.dynamic_provider_tools.len(), 1);
        // Verify that scope defaulted to Unscoped
        assert_eq!(
            tool_info.dynamic_provider_tools[0].scope,
            ProviderToolScope::Unscoped
        );
    }

    #[test]
    fn test_tool_call_config_database_insert_deserialize_null_required_field() {
        // Test with null for a required field (tool_choice)
        let json = json!({
            "baz": "test",
            "dynamic_tools": [],
            "dynamic_provider_tools": [],
            "allowed_tools": r#"{"tools":[],"choice":"function_default"}"#,
            "tool_choice": null,
            "tool_params": {
                "tools_available": [],
                "tool_choice": "auto"
            }
        });

        let result: Result<ToolCallConfigDeserializeTestHelper, _> = serde_json::from_value(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_call_config_database_insert_deserialize_empty_tool_name() {
        // Test with empty string for tool name
        let json = json!({
            "baz": "test",
            "dynamic_tools": [
                r#"{"type":"function","name":"","description":"A test tool","parameters":{"type":"object"}}"#
            ],
            "dynamic_provider_tools": [],
            "allowed_tools": r#"{"tools":[""],"choice":"function_default"}"#,
            "tool_choice": "auto",
            "tool_params": {
                "tools_available": [],
                "tool_choice": "auto"
            }
        });

        // Empty strings should deserialize successfully but may be caught by validation logic elsewhere
        let result: Result<ToolCallConfigDeserializeTestHelper, _> = serde_json::from_value(json);
        assert!(result.is_ok());
    }

    #[test]
    fn test_tool_call_config_database_insert_deserialize_specific_tool_choice_with_value() {
        // Test with specific tool choice that has a value
        let json = json!({
            "baz": "test",
            "dynamic_tools": [
                r#"{"type":"function","name":"specific_tool","description":"A specific tool","parameters":{"type":"object"}}"#
            ],
            "dynamic_provider_tools": [],
            "allowed_tools": r#"{"tools":["specific_tool"],"choice":"function_default"}"#,
            "tool_choice": {"specific": "specific_tool"},
            "tool_params": {
                "tools_available": [],
                "tool_choice": {"specific": "specific_tool"}
            }
        });

        let result: ToolCallConfigDeserializeTestHelper = serde_json::from_value(json).unwrap();
        assert_eq!(result.baz, "test");
        let tool_info = result.tool_info.unwrap();
        assert_eq!(
            tool_info.tool_choice,
            ToolChoice::Specific("specific_tool".to_string())
        );
    }

    #[test]
    fn test_tool_call_config_database_insert_deserialize_mixed_valid_invalid_tools() {
        // Test with some valid and some invalid tools
        let json = json!({
            "baz": "test",
            "dynamic_tools": [
                r#"{"type":"function","name":"valid_tool","description":"A valid tool","parameters":{"type":"object"}}"#,
                r#"{"type":"invalid_type","name":"invalid_tool"}"#
            ],
            "dynamic_provider_tools": [],
            "allowed_tools": r#"{"tools":["valid_tool"],"choice":"function_default"}"#,
            "tool_choice": r#""auto""#,
            "tool_params": {
                "tools_available": [],
                "tool_choice": "auto"
            }
        });

        let result: Result<ToolCallConfigDeserializeTestHelper, _> = serde_json::from_value(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_call_config_database_insert_deserialize_extra_fields_ignored() {
        // Test that extra unknown fields are ignored (thanks to flatten)
        let json = json!({
            "baz": "test",
            "unknown_field": "should_be_ignored",
            "dynamic_tools": [],
            "dynamic_provider_tools": [],
            "allowed_tools": r#"{"tools":[],"choice":"function_default"}"#,
            "tool_choice": "auto",
            "tool_params": {
                "tools_available": [],
                "tool_choice": "auto"
            }
        });

        let result: ToolCallConfigDeserializeTestHelper = serde_json::from_value(json).unwrap();
        assert_eq!(result.baz, "test");
        assert!(result.tool_info.is_some());
    }

    #[test]
    fn test_tool_call_config_database_insert_deserialize_legacy_empty_arrays_filtered() {
        // Test that empty dynamic_tools and dynamic_provider_tools arrays are filtered out
        // when NOT in full format mode (i.e., legacy data without allowed_tools/tool_choice)
        // This handles the case where ClickHouse returns default values (empty arrays) for these columns
        // when they weren't explicitly set
        let json = json!({
            "baz": "legacy_value",
            "dynamic_tools": [],
            "dynamic_provider_tools": [],
            "tool_params": {
                "tools_available": [
                    {
                        "name": "get_temperature",
                        "description": "Get temperature",
                        "parameters": {"type": "object"},
                        "strict": true
                    }
                ],
                "tool_choice": "auto"
            }
        });

        // This should deserialize successfully with tool_info:
        // 1. We're NOT in full format mode (no allowed_tools/tool_choice fields)
        // 2. Empty arrays for dynamic_tools and dynamic_provider_tools get filtered out
        // 3. After filtering, only tool_params remains, which is legacy format
        // 4. tool_params with tools_available is valid in legacy format
        let result: ToolCallConfigDeserializeTestHelper = serde_json::from_value(json).unwrap();
        assert_eq!(result.baz, "legacy_value");
        assert!(
            result.tool_info.is_some(),
            "tool_info should be Some with valid tool_params"
        );
    }

    #[test]
    fn test_tool_call_config_database_insert_deserialize_full_format_empty_arrays_kept() {
        // Test that empty dynamic_tools and dynamic_provider_tools arrays are KEPT
        // when in full format mode (i.e., has allowed_tools and tool_choice fields)
        // In full format, empty arrays are valid and should not be filtered out
        let json = json!({
            "baz": "full_format_value",
            "dynamic_tools": [],
            "dynamic_provider_tools": [],
            "allowed_tools": r#"{"tools":[],"choice":"function_default"}"#,
            "tool_choice": "auto",
            "tool_params": {
                "tools_available": [],
                "tool_choice": "auto"
            }
        });

        // This should deserialize to Some because:
        // 1. We ARE in full format mode (has allowed_tools and tool_choice)
        // 2. Empty arrays are valid in full format and should not be filtered
        // 3. The presence of allowed_tools/tool_choice indicates this is valid full format data
        let result: ToolCallConfigDeserializeTestHelper = serde_json::from_value(json).unwrap();
        assert_eq!(result.baz, "full_format_value");
        assert!(
            result.tool_info.is_some(),
            "tool_info should be Some in full format mode even with empty arrays"
        );

        let tool_info = result.tool_info.unwrap();
        assert_eq!(tool_info.tool_choice, ToolChoice::Auto);
        assert_eq!(tool_info.dynamic_tools.len(), 0);
        assert_eq!(tool_info.dynamic_provider_tools.len(), 0);
    }

    #[test]
    fn test_tool_call_config_database_insert_deserialize_legacy_only_empty_arrays() {
        // Test that when ONLY empty dynamic_tools and dynamic_provider_tools arrays are present
        // (no tool_params, no allowed_tools/tool_choice), they get filtered out and we get None
        // This handles the case where ClickHouse returns default empty arrays but no actual tool data
        let json = json!({
            "baz": "only_empty_arrays",
            "dynamic_tools": [],
            "dynamic_provider_tools": []
        });

        // This should deserialize to None because:
        // 1. We're NOT in full format mode (no allowed_tools/tool_choice)
        // 2. Empty arrays for dynamic_tools and dynamic_provider_tools get filtered out
        // 3. After filtering, values is empty
        // 4. Empty values results in None
        assert_deserialize_to_none(json, "only_empty_arrays");
    }

    #[test]
    fn test_strict_tools_available_with_function_default() {
        // Test that FunctionDefault returns all available tools
        let config = ToolCallConfig {
            static_tools_available: vec![
                FunctionToolConfig::Static(TOOLS.get("get_temperature").unwrap().clone()),
                FunctionToolConfig::Static(TOOLS.get("query_articles").unwrap().clone()),
            ],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
            allowed_tools: AllowedTools::default(), // FunctionDefault
        };

        let tools: Vec<_> = config.strict_tools_available().unwrap().collect();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name(), "get_temperature");
        assert_eq!(tools[1].name(), "query_articles");
    }

    #[test]
    fn test_strict_tools_available_with_all_allowed_tools() {
        // Test that AllAllowedTools filters to the specified subset
        let config = ToolCallConfig {
            static_tools_available: vec![
                FunctionToolConfig::Static(TOOLS.get("get_temperature").unwrap().clone()),
                FunctionToolConfig::Static(TOOLS.get("query_articles").unwrap().clone()),
            ],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
            allowed_tools: AllowedTools {
                tools: vec!["get_temperature".to_string()].into_iter().collect(),
                choice: AllowedToolsChoice::Explicit,
            },
        };

        let tools: Vec<_> = config.strict_tools_available().unwrap().collect();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "get_temperature");
    }

    #[tokio::test]
    async fn test_config_only_allowed_tool_added_to_static_tools() {
        // Test that a tool from the config that is NOT in function_tools
        // but IS in allowed_tools gets added to static_tools_available

        // Function only has get_temperature in its tools list
        let function_tools = vec!["get_temperature".to_string()];

        // But allowed_tools includes query_articles (which exists in config)
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec![
                "get_temperature".to_string(),
                "query_articles".to_string(), // This is in config but not in function_tools
            ]),
            ..Default::default()
        };

        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &function_tools,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        ))
        .unwrap()
        .unwrap();

        // Should have 2 tools available (both get_temperature and query_articles)
        assert_eq!(tool_call_config.tools_available().unwrap().count(), 2);

        // Both should be in strict_tools_available since they're in allowed_tools
        assert_eq!(
            tool_call_config.strict_tools_available().unwrap().count(),
            2
        );

        let tool_names: Vec<_> = tool_call_config
            .tools_available()
            .unwrap()
            .map(|t| t.name())
            .collect();
        assert!(tool_names.contains(&"get_temperature"));
        assert!(tool_names.contains(&"query_articles"));
    }

    #[tokio::test]
    async fn test_multiple_config_only_allowed_tools() {
        // Test that multiple tools from config (not in function_tools) can be added via allowed_tools

        // Function has no tools configured
        let function_tools: Vec<String> = vec![];

        // But allowed_tools includes both tools from config
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec![
                "get_temperature".to_string(),
                "query_articles".to_string(),
            ]),
            ..Default::default()
        };

        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &function_tools,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        ))
        .unwrap()
        .unwrap();

        // Should have 2 tools available (both from config via allowed_tools)
        assert_eq!(tool_call_config.tools_available().unwrap().count(), 2);
        assert_eq!(
            tool_call_config.strict_tools_available().unwrap().count(),
            2
        );
    }

    #[tokio::test]
    async fn test_mix_of_function_and_config_only_allowed_tools() {
        // Test mixing function tools and config-only allowed tools

        // Function only has get_temperature
        let function_tools = vec!["get_temperature".to_string()];

        // allowed_tools has get_temperature (from function) and query_articles (config-only)
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec![
                "get_temperature".to_string(),
                "query_articles".to_string(),
            ]),
            ..Default::default()
        };

        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &function_tools,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        ))
        .unwrap()
        .unwrap();

        assert_eq!(tool_call_config.tools_available().unwrap().count(), 2);
        assert_eq!(
            tool_call_config.strict_tools_available().unwrap().count(),
            2
        );

        // Verify choice is AllAllowedTools
        assert!(matches!(
            tool_call_config.allowed_tools.choice,
            AllowedToolsChoice::Explicit
        ));
    }

    #[tokio::test]
    async fn test_config_only_tool_with_dynamic_tools() {
        // Test that config-only allowed tools work alongside dynamic tools

        let function_tools = vec!["get_temperature".to_string()];

        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec![
                "get_temperature".to_string(),
                "query_articles".to_string(),       // config-only
                "establish_campground".to_string(), // dynamic
            ]),
            additional_tools: Some(vec![Tool::Function(FunctionTool {
                name: "establish_campground".to_string(),
                description: "Establish a campground".to_string(),
                parameters: json!({"type": "object", "properties": {"location": {"type": "string"}}}),
                strict: false,
            })]),
            ..Default::default()
        };

        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &function_tools,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        ))
        .unwrap()
        .unwrap();

        // Should have 3 tools: get_temperature (function), query_articles (config-only), establish_campground (dynamic)
        assert_eq!(tool_call_config.tools_available().unwrap().count(), 3);
        assert_eq!(
            tool_call_config.strict_tools_available().unwrap().count(),
            3
        );

        let tool_names: Vec<_> = tool_call_config
            .tools_available()
            .unwrap()
            .map(|t| t.name())
            .collect();
        assert!(tool_names.contains(&"get_temperature"));
        assert!(tool_names.contains(&"query_articles"));
        assert!(tool_names.contains(&"establish_campground"));
    }

    #[tokio::test]
    async fn test_existing_function_tools_behavior_unchanged() {
        // Test that existing behavior for function_tools without allowed_tools still works

        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            DynamicToolParams::default(),
        ))
        .unwrap()
        .unwrap();

        // Should have all function tools
        assert_eq!(tool_call_config.tools_available().unwrap().count(), 2);
        assert_eq!(
            tool_call_config.strict_tools_available().unwrap().count(),
            2
        );

        // Should be FunctionDefault mode
        assert!(matches!(
            tool_call_config.allowed_tools.choice,
            AllowedToolsChoice::FunctionDefault
        ));
    }

    // ============================================================================
    // Tests for OpenAI Custom Tools (DynamicTool wrapper and Tool::Custom variant)
    // ============================================================================

    /// Test DynamicTool deserialization with untagged (legacy) format for backward compatibility
    #[test]
    fn test_dynamic_tool_deserialize_untagged_function() {
        let json = json!({
            "name": "legacy_tool",
            "description": "A tool in legacy format",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                }
            },
            "strict": false
        });

        let result: Tool = serde_json::from_value(json).unwrap();
        assert!(matches!(result, Tool::Function(_)));
        if let Tool::Function(func) = result {
            assert_eq!(func.name, "legacy_tool");
            assert_eq!(func.description, "A tool in legacy format");
            assert!(!func.strict);
        }
    }

    /// Test DynamicTool deserialization with new tagged format
    #[test]
    fn test_dynamic_tool_deserialize_tagged_formats() {
        // Test tagged function format
        let function_json = json!({
            "type": "function",
            "name": "tagged_function",
            "description": "A function tool with tag",
            "parameters": {
                "type": "object",
                "properties": {
                    "arg": {"type": "string"}
                }
            },
            "strict": true
        });

        let result: Tool = serde_json::from_value(function_json).unwrap();
        assert!(matches!(result, Tool::Function(_)));
        assert!(result.is_function());
        assert!(!result.is_custom());

        // Test tagged custom format
        let custom_json = json!({
            "type": "openai_custom",
            "name": "custom_tool",
            "description": "A custom tool",
            "format": {
                "type": "text"
            }
        });

        let result: Tool = serde_json::from_value(custom_json).unwrap();
        assert!(matches!(result, Tool::OpenAICustom(_)));
        assert!(result.is_custom());
        assert!(!result.is_function());
    }

    /// Test DynamicTool with various custom tool formats
    #[test]
    fn test_dynamic_tool_custom_variants() {
        // Text format
        let text_json = json!({
            "type": "openai_custom",
            "name": "text_tool",
            "format": {
                "type": "text"
            }
        });
        let result: Tool = serde_json::from_value(text_json).unwrap();
        if let Tool::OpenAICustom(custom) = result {
            assert_eq!(custom.name, "text_tool");
            assert!(matches!(custom.format, Some(OpenAICustomToolFormat::Text)));
        } else {
            panic!("Expected Tool::Custom");
        }

        // Lark grammar format
        let lark_json = json!({
            "type": "openai_custom",
            "name": "lark_tool",
            "description": "Uses Lark grammar",
            "format": {
                "type": "grammar",
                "grammar": {
                    "syntax": "lark",
                    "definition": "start: \"hello\""
                }
            }
        });
        let result: Tool = serde_json::from_value(lark_json).unwrap();
        if let Tool::OpenAICustom(custom) = result {
            assert_eq!(custom.name, "lark_tool");
            assert_eq!(custom.description, Some("Uses Lark grammar".to_string()));
            if let Some(OpenAICustomToolFormat::Grammar { grammar }) = custom.format {
                assert!(matches!(grammar.syntax, OpenAIGrammarSyntax::Lark));
                assert_eq!(grammar.definition, "start: \"hello\"");
            } else {
                panic!("Expected Grammar format");
            }
        }

        // Regex grammar format
        let regex_json = json!({
            "type": "openai_custom",
            "name": "regex_tool",
            "format": {
                "type": "grammar",
                "grammar": {
                    "syntax": "regex",
                    "definition": "[a-z]+"
                }
            }
        });
        let result: Tool = serde_json::from_value(regex_json).unwrap();
        if let Tool::OpenAICustom(custom) = result {
            if let Some(OpenAICustomToolFormat::Grammar { grammar }) = custom.format {
                assert!(matches!(grammar.syntax, OpenAIGrammarSyntax::Regex));
            }
        }

        // Minimal custom tool (no description, no format)
        let minimal_json = json!({
            "type": "openai_custom",
            "name": "minimal_tool"
        });
        let result: Tool = serde_json::from_value(minimal_json).unwrap();
        if let Tool::OpenAICustom(custom) = result {
            assert_eq!(custom.name, "minimal_tool");
            assert!(custom.description.is_none());
            assert!(custom.format.is_none());
        }
    }

    /// Test DynamicTool error handling with invalid formats
    #[test]
    fn test_dynamic_tool_deserialize_invalid() {
        // Missing required name field
        let invalid_json = json!({
            "type": "openai_custom",
            "description": "Missing name"
        });
        let result = serde_json::from_value::<Tool>(invalid_json);
        assert!(result.is_err());

        // Unknown type
        let invalid_json = json!({
            "type": "unknown_type",
            "name": "test"
        });
        let result = serde_json::from_value::<Tool>(invalid_json);
        assert!(result.is_err());

        // Invalid structure (neither Tool nor FunctionTool)
        let invalid_json = json!({
            "completely": "wrong"
        });
        let result = serde_json::from_value::<Tool>(invalid_json);
        assert!(result.is_err());
    }

    /// Test Tool deserialization fallback from tagged to untagged format
    #[test]
    fn test_tool_deserialize_fallback_behavior() {
        // Test that a valid FunctionTool without type field uses fallback path (legacy format)
        let untagged_function = json!({
            "name": "fallback_tool",
            "description": "Testing fallback",
            "parameters": {
                "type": "object",
                "properties": {
                    "param": {"type": "string"}
                }
            },
            "strict": false
        });

        let result: Tool = serde_json::from_value(untagged_function).unwrap();
        assert!(result.is_function());
        assert_eq!(result.name(), "fallback_tool");
        let Tool::Function(func) = result else {
            panic!("Expected Tool::Function, got different variant");
        };
        assert_eq!(func.description, "Testing fallback");
        assert!(!func.strict);

        // Test error message when neither tagged nor untagged format matches
        let completely_invalid = json!({
            "type": "invalid_type",
            "wrong_field": "value"
        });

        let result = serde_json::from_value::<Tool>(completely_invalid);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(
            error_msg.contains("Failed to parse as `Tool` (tagged) or `FunctionTool` (untagged)")
        );

        // Test that missing required fields in both formats produces an error
        let missing_name = json!({
            "description": "No name field",
            "parameters": {"type": "object"},
            "strict": false
        });

        let result = serde_json::from_value::<Tool>(missing_name);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Failed to parse"));
    }

    /// Test that both "function" and "client_side_function" tags are accepted for Function variant
    #[test]
    fn test_tool_deserialize_function_tag_variants() {
        // Test new "function" tag
        let function_tag_json = json!({
            "type": "function",
            "name": "new_format_tool",
            "description": "Uses new function tag",
            "parameters": {
                "type": "object",
                "properties": {
                    "arg": {"type": "string"}
                }
            },
            "strict": true
        });

        let result: Tool = serde_json::from_value(function_tag_json).unwrap();
        let Tool::Function(func) = result else {
            panic!("Expected Tool::Function, got different variant");
        };
        assert_eq!(func.name, "new_format_tool");
        assert_eq!(func.description, "Uses new function tag");
        assert!(func.strict);

        // Test legacy "client_side_function" tag (backward compatibility)
        // We've stored the former in the database, so we can't remove this alias.
        let legacy_tag_json = json!({
            "type": "client_side_function",
            "name": "legacy_format_tool",
            "description": "Uses legacy client_side_function tag",
            "parameters": {
                "type": "object",
                "properties": {
                    "arg": {"type": "number"}
                }
            },
            "strict": false
        });

        let result: Tool = serde_json::from_value(legacy_tag_json).unwrap();
        let Tool::Function(func) = result else {
            panic!("Expected Tool::Function, got different variant");
        };
        assert_eq!(func.name, "legacy_format_tool");
        assert_eq!(func.description, "Uses legacy client_side_function tag");
        assert!(!func.strict);
    }

    /// Test Tool enum serialization round-trip and helper methods
    #[test]
    fn test_tool_serialization_and_methods() {
        // Function tool round-trip
        let function_tool = Tool::Function(FunctionTool {
            name: "test_func".to_string(),
            description: "Test function".to_string(),
            parameters: json!({"type": "object"}),
            strict: true,
        });

        let json = serde_json::to_value(&function_tool).unwrap();
        let deserialized: Tool = serde_json::from_value(json).unwrap();
        assert_eq!(function_tool, deserialized);
        assert_eq!(deserialized.name(), "test_func");
        assert!(!deserialized.is_custom());
        assert!(deserialized.is_function());

        // Custom tool round-trip
        let custom_tool = Tool::OpenAICustom(OpenAICustomTool {
            name: "custom_func".to_string(),
            description: Some("Custom function".to_string()),
            format: Some(OpenAICustomToolFormat::Text),
        });

        let json = serde_json::to_value(&custom_tool).unwrap();
        let deserialized: Tool = serde_json::from_value(json).unwrap();
        assert_eq!(custom_tool, deserialized);
        assert_eq!(deserialized.name(), "custom_func");
        assert!(deserialized.is_custom());
        assert!(!deserialized.is_function());
    }

    /// Test DynamicTool <-> Tool conversions
    #[test]
    fn test_dynamic_tool_conversions() {
        // Tool -> DynamicTool
        let tool = Tool::OpenAICustom(OpenAICustomTool {
            name: "test".to_string(),
            description: None,
            format: None,
        });
        let dynamic: Tool = tool.clone();
        assert_eq!(dynamic, tool);

        // DynamicTool -> Tool
        let tool_back: Tool = dynamic;
        assert_eq!(tool_back, tool);

        // Mixed vector conversion
        let tools = vec![
            Tool::Function(FunctionTool {
                name: "func1".to_string(),
                description: "Function 1".to_string(),
                parameters: json!({"type": "object"}),
                strict: false,
            }),
            Tool::OpenAICustom(OpenAICustomTool {
                name: "custom1".to_string(),
                description: None,
                format: Some(OpenAICustomToolFormat::Text),
            }),
        ];

        let extracted: Vec<Tool> = tools.into_iter().collect();
        assert_eq!(extracted.len(), 2);
        assert!(extracted[0].is_function());
        assert!(extracted[1].is_custom());
    }

    /// Test DynamicToolParams with custom tools
    #[test]
    fn test_dynamic_tool_params_with_custom_tools() {
        let params = DynamicToolParams {
            allowed_tools: Some(vec!["tool1".to_string()]),
            additional_tools: Some(vec![
                Tool::Function(FunctionTool {
                    name: "func_tool".to_string(),
                    description: "Function tool".to_string(),
                    parameters: json!({"type": "object"}),
                    strict: false,
                }),
                Tool::OpenAICustom(OpenAICustomTool {
                    name: "custom_tool".to_string(),
                    description: Some("Custom tool".to_string()),
                    format: Some(OpenAICustomToolFormat::Text),
                }),
            ]),
            tool_choice: Some(ToolChoice::Auto),
            parallel_tool_calls: Some(true),
            provider_tools: vec![],
        };

        // Test serialization
        let json = serde_json::to_value(&params).unwrap();
        assert!(json["additional_tools"].is_array());
        assert_eq!(json["additional_tools"].as_array().unwrap().len(), 2);

        // Test deserialization round-trip
        let deserialized: DynamicToolParams = serde_json::from_value(json).unwrap();
        assert_eq!(deserialized.additional_tools.as_ref().unwrap().len(), 2);
        assert!(deserialized.additional_tools.as_ref().unwrap()[0].is_function());
        assert!(deserialized.additional_tools.as_ref().unwrap()[1].is_custom());
    }

    /// Test DynamicToolParams with mixed tagged and untagged tools
    #[test]
    fn test_dynamic_tool_params_mixed_deserialization() {
        let json = json!({
            "additional_tools": [
                {
                    "name": "untagged_tool",
                    "description": "Legacy format",
                    "parameters": {"type": "object"},
                    "strict": false
                },
                {
                    "type": "function",
                    "name": "tagged_function",
                    "description": "New format",
                    "parameters": {"type": "object"},
                    "strict": true
                },
                {
                    "type": "openai_custom",
                    "name": "custom_tool",
                    "format": {"type": "text"}
                }
            ]
        });

        let params: DynamicToolParams = serde_json::from_value(json).unwrap();
        let tools = params.additional_tools.unwrap();
        assert_eq!(tools.len(), 3);

        // First should be function (from untagged)
        assert!(tools[0].is_function());
        // Second should be function (from tagged)
        assert!(tools[1].is_function());
        // Third should be custom
        assert!(tools[2].is_custom());
    }

    /// Test ToolCallConfig iterator behavior with custom tools
    #[tokio::test]
    async fn test_tool_call_config_iterators_with_custom_tools() {
        let dynamic_params = DynamicToolParams {
            allowed_tools: None,
            additional_tools: Some(vec![
                Tool::Function(FunctionTool {
                    name: "dynamic_func".to_string(),
                    description: "Dynamic function".to_string(),
                    parameters: json!({"type": "object"}),
                    strict: false,
                }),
                Tool::OpenAICustom(OpenAICustomTool {
                    name: "dynamic_custom".to_string(),
                    description: Some("Dynamic custom".to_string()),
                    format: Some(OpenAICustomToolFormat::Text),
                }),
            ]),
            tool_choice: Some(ToolChoice::Auto),
            parallel_tool_calls: Some(true),
            provider_tools: vec![],
        };

        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_params,
        ))
        .unwrap()
        .unwrap();

        // Verify we have both function and custom tools
        // tools_available() should error when custom tools are present
        assert!(tool_call_config.tools_available().is_err());
        let custom_tools_count = tool_call_config.openai_custom_tools.len();
        assert_eq!(custom_tools_count, 1); // dynamic_custom

        // Verify the function tool is still accessible by name even with custom tools present
        let func_tool = tool_call_config.get_function_tool("dynamic_func").unwrap();
        assert_eq!(func_tool.name(), "dynamic_func");

        // Verify the custom tool is in the custom tools list
        let custom_tool = &tool_call_config.openai_custom_tools[0];
        assert_eq!(custom_tool.name, "dynamic_custom");
    }

    /// Test ToolCallConfig strict filtering with custom tools
    #[tokio::test]
    async fn test_tool_call_config_strict_filtering_with_custom() {
        let dynamic_params = DynamicToolParams {
            allowed_tools: Some(vec![
                "get_temperature".to_string(),
                "dynamic_custom".to_string(),
            ]),
            additional_tools: Some(vec![Tool::OpenAICustom(OpenAICustomTool {
                name: "dynamic_custom".to_string(),
                description: None,
                format: Some(OpenAICustomToolFormat::Text),
            })]),
            tool_choice: None,
            parallel_tool_calls: None,
            provider_tools: vec![],
        };

        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_params,
        ))
        .unwrap()
        .unwrap();

        // strict_tools_available() should error when custom tools are present
        assert!(tool_call_config.strict_tools_available().is_err());

        // Custom tools are separate - check that the custom tool is in the list
        assert_eq!(tool_call_config.openai_custom_tools.len(), 1);
        assert_eq!(
            tool_call_config.openai_custom_tools[0].name,
            "dynamic_custom"
        );
    }

    /// Test ToolCallConfig construction creates correct ToolConfig variants
    #[tokio::test]
    async fn test_tool_call_config_construction_with_custom_tools() {
        let dynamic_params = DynamicToolParams {
            allowed_tools: None,
            additional_tools: Some(vec![
                Tool::Function(FunctionTool {
                    name: "dynamic_func".to_string(),
                    description: "Func".to_string(),
                    parameters: json!({"type": "object"}),
                    strict: true,
                }),
                Tool::OpenAICustom(OpenAICustomTool {
                    name: "dynamic_custom".to_string(),
                    description: Some("Custom".to_string()),
                    format: Some(OpenAICustomToolFormat::Text),
                }),
            ]),
            tool_choice: None,
            parallel_tool_calls: None,
            provider_tools: vec![],
        };

        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &EMPTY_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_params,
        ))
        .unwrap()
        .unwrap();

        // Verify function tool became DynamicToolConfig
        let func_config = tool_call_config.get_function_tool("dynamic_func").unwrap();
        assert!(matches!(func_config, FunctionToolConfig::Dynamic(_)));

        // Verify custom tool is in the custom tools list
        let custom_config = tool_call_config
            .openai_custom_tools
            .iter()
            .find(|t| t.name == "dynamic_custom")
            .expect("dynamic_custom should be in openai_custom_tools");
        assert_eq!(custom_config.description, Some("Custom".to_string()));
    }

    /// Test ToolCallConfigDatabaseInsert with custom tools
    #[tokio::test]
    async fn test_database_insert_with_custom_tools() {
        let tools = vec![
            Tool::Function(FunctionTool {
                name: "func_tool".to_string(),
                description: "Function".to_string(),
                parameters: json!({"type": "object"}),
                strict: false,
            }),
            Tool::OpenAICustom(OpenAICustomTool {
                name: "custom_tool".to_string(),
                description: Some("Custom".to_string()),
                format: Some(OpenAICustomToolFormat::Text),
            }),
        ];

        let db_insert = ToolCallConfigDatabaseInsert {
            allowed_tools: AllowedTools::default(),
            dynamic_tools: tools.clone(),
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: Some(true),
            dynamic_provider_tools: vec![],
            tool_params: Default::default(),
        };

        // Test serialization
        let json = serde_json::to_value(&db_insert).unwrap();
        assert!(json["dynamic_tools"].is_array());
        assert_eq!(json["dynamic_tools"].as_array().unwrap().len(), 2);

        // Test deserialization round-trip
        let deserialized: ToolCallConfigDatabaseInsert = serde_json::from_value(json).unwrap();
        assert_eq!(deserialized.dynamic_tools.len(), 2);
        assert!(deserialized.dynamic_tools[0].is_function());
        assert!(deserialized.dynamic_tools[1].is_custom());

        // Verify custom tool data preserved
        if let Tool::OpenAICustom(custom) = &deserialized.dynamic_tools[1] {
            assert_eq!(custom.name, "custom_tool");
            assert_eq!(custom.description, Some("Custom".to_string()));
        } else {
            panic!("Expected custom tool");
        }
    }

    /// Test backward compatibility with legacy database records
    #[tokio::test]
    async fn test_database_insert_legacy_compatibility() {
        // Simulate old database format without custom tools (using legacy format)
        let legacy_json = json!({
            "allowed_tools": {
                "choice": "function_default",
                "tools": []
            },
            "dynamic_tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": true,
            "dynamic_provider_tools": []
        });

        let result: Result<ToolCallConfigDatabaseInsert, _> = serde_json::from_value(legacy_json);
        assert!(result.is_ok());
        let db_insert = result.unwrap();
        assert_eq!(db_insert.dynamic_tools.len(), 0);

        // Old records with only function tools should still work
        let function_only_json = json!({
            "allowed_tools": {
                "choice": "function_default",
                "tools": []
            },
            "dynamic_tools": [
                {
                    "type": "function",
                    "name": "old_tool",
                    "description": "Old function tool",
                    "parameters": {"type": "object"},
                    "strict": false
                }
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": false,
            "dynamic_provider_tools": []
        });

        let result: ToolCallConfigDatabaseInsert =
            serde_json::from_value(function_only_json).unwrap();
        assert_eq!(result.dynamic_tools.len(), 1);
        assert!(result.dynamic_tools[0].is_function());
    }

    /// Test edge cases for custom tools
    #[test]
    fn test_custom_tools_edge_cases() {
        // Empty additional_tools vec
        let params = DynamicToolParams {
            allowed_tools: None,
            additional_tools: Some(vec![]),
            tool_choice: None,
            parallel_tool_calls: None,
            provider_tools: vec![],
        };
        let json = serde_json::to_value(&params).unwrap();
        let deserialized: DynamicToolParams = serde_json::from_value(json).unwrap();
        assert_eq!(deserialized.additional_tools.unwrap().len(), 0);

        // None additional_tools
        let params = DynamicToolParams {
            allowed_tools: None,
            additional_tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            provider_tools: vec![],
        };
        let json = serde_json::to_value(&params).unwrap();
        let deserialized: DynamicToolParams = serde_json::from_value(json).unwrap();
        assert!(deserialized.additional_tools.is_none());

        // Custom tool with minimal fields
        let minimal_custom = Tool::OpenAICustom(OpenAICustomTool {
            name: "min".to_string(),
            description: None,
            format: None,
        });
        let json = serde_json::to_value(&minimal_custom).unwrap();
        let deserialized: Tool = serde_json::from_value(json).unwrap();
        assert_eq!(deserialized.name(), "min");

        // Tool name deduplication should work with custom tools (name() method works)
        let custom1 = Tool::OpenAICustom(OpenAICustomTool {
            name: "duplicate".to_string(),
            description: None,
            format: None,
        });
        let custom2 = Tool::OpenAICustom(OpenAICustomTool {
            name: "duplicate".to_string(),
            description: Some("Different description".to_string()),
            format: Some(OpenAICustomToolFormat::Text),
        });
        assert_eq!(custom1.name(), custom2.name());
    }

    /// Test that ToolCallConfig is created (not None) when ONLY custom tools are provided
    /// This is a regression test for a bug where ToolCallConfig::new would return None
    /// if there were custom tools but no function tools.
    #[tokio::test]
    async fn test_tool_call_config_with_only_custom_tools() {
        // Create params with ONLY custom tools - no function tools, no provider tools
        let dynamic_params = DynamicToolParams {
            allowed_tools: None,
            additional_tools: Some(vec![
                Tool::OpenAICustom(OpenAICustomTool {
                    name: "only_custom_1".to_string(),
                    description: Some("First custom tool".to_string()),
                    format: Some(OpenAICustomToolFormat::Text),
                }),
                Tool::OpenAICustom(OpenAICustomTool {
                    name: "only_custom_2".to_string(),
                    description: Some("Second custom tool".to_string()),
                    format: Some(OpenAICustomToolFormat::Grammar {
                        grammar: OpenAIGrammarDefinition {
                            syntax: OpenAIGrammarSyntax::Lark,
                            definition: "start: WORD+".to_string(),
                        },
                    }),
                }),
            ]),
            tool_choice: None,
            parallel_tool_calls: None,
            provider_tools: vec![],
        };

        // Create ToolCallConfig with NO static function tools
        let tool_call_config_result =
            ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
                &EMPTY_FUNCTION_TOOLS, // No static function tools
                &AUTO_TOOL_CHOICE,
                Some(true),
                &TOOLS,
                dynamic_params,
            ))
            .unwrap();

        // CRITICAL: This should be Some, not None, even though there are no function tools
        assert!(
            tool_call_config_result.is_some(),
            "ToolCallConfig should be Some when only custom tools are provided"
        );

        let tool_call_config = tool_call_config_result.unwrap();

        // Verify custom tools are present
        assert_eq!(tool_call_config.openai_custom_tools.len(), 2);
        assert_eq!(
            tool_call_config.openai_custom_tools[0].name,
            "only_custom_1"
        );
        assert_eq!(
            tool_call_config.openai_custom_tools[1].name,
            "only_custom_2"
        );

        // Verify no function tools are present
        // tools_available() should error when custom tools are present
        assert!(tool_call_config.tools_available().is_err());

        // Verify tool choice and parallel_tool_calls are set correctly
        assert!(matches!(
            tool_call_config.tool_choice,
            ToolChoice::Auto | ToolChoice::Required
        ));
        assert_eq!(tool_call_config.parallel_tool_calls, Some(true));
    }

    /// Test that tools_available() returns an error when custom tools are present
    #[tokio::test]
    async fn test_tools_available_errors_with_custom_tools() {
        let dynamic_params = DynamicToolParams {
            allowed_tools: Some(vec!["get_temperature".to_string()]),
            additional_tools: Some(vec![Tool::OpenAICustom(OpenAICustomTool {
                name: "custom_tool".to_string(),
                description: Some("A custom tool".to_string()),
                format: Some(OpenAICustomToolFormat::Text),
            })]),
            tool_choice: None,
            parallel_tool_calls: None,
            provider_tools: vec![],
        };

        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &["get_temperature".to_string()],
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_params,
        ))
        .unwrap()
        .unwrap();

        // tools_available() should error
        let result = tool_call_config.tools_available();
        assert!(result.is_err());

        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("Expected error, got Ok"),
        };
        assert!(matches!(
            err.get_details(),
            ErrorDetails::IncompatibleTool { .. }
        ));

        // Check error message
        if let ErrorDetails::IncompatibleTool { message } = err.get_details() {
            assert!(message.contains("OpenAI custom tools are not supported by this provider"));
        }
    }

    /// Test that strict_tools_available() returns an error when custom tools are present
    #[tokio::test]
    async fn test_strict_tools_available_errors_with_custom_tools() {
        let dynamic_params = DynamicToolParams {
            allowed_tools: Some(vec!["get_temperature".to_string()]),
            additional_tools: Some(vec![Tool::OpenAICustom(OpenAICustomTool {
                name: "custom_tool".to_string(),
                description: Some("A custom tool".to_string()),
                format: Some(OpenAICustomToolFormat::Text),
            })]),
            tool_choice: None,
            parallel_tool_calls: None,
            provider_tools: vec![],
        };

        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &["get_temperature".to_string()],
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_params,
        ))
        .unwrap()
        .unwrap();

        // strict_tools_available() should error
        let result = tool_call_config.strict_tools_available();
        assert!(result.is_err());

        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("Expected error, got Ok"),
        };
        assert!(matches!(
            err.get_details(),
            ErrorDetails::IncompatibleTool { .. }
        ));
    }

    /// Test that tools_available_with_openai_custom() works correctly with only function tools
    #[tokio::test]
    async fn test_tools_available_with_openai_custom_function_tools_only() {
        let dynamic_params = DynamicToolParams {
            allowed_tools: Some(vec![
                "get_temperature".to_string(),
                "query_articles".to_string(),
            ]),
            additional_tools: None, // No custom tools
            tool_choice: None,
            parallel_tool_calls: None,
            provider_tools: vec![],
        };

        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &["get_temperature".to_string()],
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_params,
        ))
        .unwrap()
        .unwrap();

        // Should have 2 function tools
        let tools: Vec<_> = tool_call_config
            .tools_available_with_openai_custom()
            .collect();
        assert_eq!(tools.len(), 2);

        // All should be function tools
        for tool_ref in tools {
            assert!(matches!(tool_ref, ToolConfigRef::Function(_)));
        }
    }

    /// Test that tools_available_with_openai_custom() works correctly with only custom tools
    #[tokio::test]
    async fn test_tools_available_with_openai_custom_custom_tools_only() {
        let dynamic_params = DynamicToolParams {
            allowed_tools: None,
            additional_tools: Some(vec![
                Tool::OpenAICustom(OpenAICustomTool {
                    name: "custom_1".to_string(),
                    description: Some("First custom tool".to_string()),
                    format: Some(OpenAICustomToolFormat::Text),
                }),
                Tool::OpenAICustom(OpenAICustomTool {
                    name: "custom_2".to_string(),
                    description: Some("Second custom tool".to_string()),
                    format: Some(OpenAICustomToolFormat::Text),
                }),
            ]),
            tool_choice: None,
            parallel_tool_calls: None,
            provider_tools: vec![],
        };

        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &EMPTY_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_params,
        ))
        .unwrap()
        .unwrap();

        // Should have 2 custom tools
        let tools: Vec<_> = tool_call_config
            .tools_available_with_openai_custom()
            .collect();
        assert_eq!(tools.len(), 2);

        // All should be custom tools
        for tool_ref in tools {
            assert!(matches!(tool_ref, ToolConfigRef::OpenAICustom(_)));
        }
    }

    /// Test that tools_available_with_openai_custom() works correctly with both function and custom tools
    #[tokio::test]
    async fn test_tools_available_with_openai_custom_mixed_tools() {
        let dynamic_params = DynamicToolParams {
            allowed_tools: Some(vec![
                "get_temperature".to_string(),
                "query_articles".to_string(),
            ]),
            additional_tools: Some(vec![Tool::OpenAICustom(OpenAICustomTool {
                name: "custom_1".to_string(),
                description: Some("Custom tool".to_string()),
                format: Some(OpenAICustomToolFormat::Text),
            })]),
            tool_choice: None,
            parallel_tool_calls: None,
            provider_tools: vec![],
        };

        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &["get_temperature".to_string()],
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_params,
        ))
        .unwrap()
        .unwrap();

        // Should have 3 tools total (2 function + 1 custom)
        let tools: Vec<_> = tool_call_config
            .tools_available_with_openai_custom()
            .collect();
        assert_eq!(tools.len(), 3);

        // First 2 should be function tools, last should be custom
        let mut function_count = 0;
        let mut custom_count = 0;

        for tool_ref in tools {
            match tool_ref {
                ToolConfigRef::Function(_) => function_count += 1,
                ToolConfigRef::OpenAICustom(_) => custom_count += 1,
            }
        }

        assert_eq!(function_count, 2);
        assert_eq!(custom_count, 1);
    }

    /// Test that tools_available() succeeds when no custom tools are present
    #[tokio::test]
    async fn test_tools_available_succeeds_without_custom_tools() {
        let dynamic_params = DynamicToolParams {
            allowed_tools: Some(vec![
                "get_temperature".to_string(),
                "query_articles".to_string(),
            ]),
            additional_tools: None, // No custom tools
            tool_choice: None,
            parallel_tool_calls: None,
            provider_tools: vec![],
        };

        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &["get_temperature".to_string()],
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_params,
        ))
        .unwrap()
        .unwrap();

        // tools_available() should succeed when no custom tools
        let result = tool_call_config.tools_available();
        assert!(result.is_ok());

        let tools: Vec<_> = result.unwrap().collect();
        assert_eq!(tools.len(), 2);
    }

    /// Test that strict_tools_available() succeeds when no custom tools are present
    #[tokio::test]
    async fn test_strict_tools_available_succeeds_without_custom_tools() {
        let dynamic_params = DynamicToolParams {
            allowed_tools: Some(vec![
                "get_temperature".to_string(),
                "query_articles".to_string(),
            ]),
            additional_tools: None, // No custom tools
            tool_choice: None,
            parallel_tool_calls: None,
            provider_tools: vec![],
        };

        let tool_call_config = ToolCallConfig::new(ToolCallConfigConstructorArgs::new_for_test(
            &["get_temperature".to_string()],
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_params,
        ))
        .unwrap()
        .unwrap();

        // strict_tools_available() should succeed when no custom tools
        let result = tool_call_config.strict_tools_available();
        assert!(result.is_ok());

        let tools: Vec<_> = result.unwrap().collect();
        assert_eq!(tools.len(), 2);
    }

    #[test]
    fn test_provider_tool_scope_deserialize_new_format_with_provider() {
        let json = r#"{"model_name": "gpt-4", "provider_name": "openai"}"#;
        let scope: ProviderToolScope = serde_json::from_str(json).unwrap();
        assert_eq!(
            scope,
            ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
                model_name: "gpt-4".to_string(),
                provider_name: Some("openai".to_string()),
            })
        );
    }

    #[test]
    fn test_provider_tool_scope_deserialize_new_format_without_provider() {
        let json = r#"{"model_name": "gpt-4"}"#;
        let scope: ProviderToolScope = serde_json::from_str(json).unwrap();
        assert_eq!(
            scope,
            ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
                model_name: "gpt-4".to_string(),
                provider_name: None,
            })
        );
    }

    #[test]
    fn test_provider_tool_scope_deserialize_old_format_backward_compat() {
        // Old format with model_provider_name should still work
        let json = r#"{"model_name": "gpt-4", "model_provider_name": "openai"}"#;
        let scope: ProviderToolScope = serde_json::from_str(json).unwrap();
        assert_eq!(
            scope,
            ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
                model_name: "gpt-4".to_string(),
                provider_name: Some("openai".to_string()),
            })
        );
    }

    #[test]
    fn test_provider_tool_scope_deserialize_null() {
        let json = "null";
        let scope: ProviderToolScope = serde_json::from_str(json).unwrap();
        assert_eq!(scope, ProviderToolScope::Unscoped);
    }

    #[test]
    fn test_provider_tool_scope_serialize_with_provider() {
        let scope = ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
            model_name: "gpt-4".to_string(),
            provider_name: Some("openai".to_string()),
        });
        let json = serde_json::to_string(&scope).unwrap();
        // Should serialize with provider_name (new format)
        assert_eq!(json, r#"{"model_name":"gpt-4","provider_name":"openai"}"#);
    }

    #[test]
    fn test_provider_tool_scope_serialize_without_provider() {
        let scope = ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
            model_name: "gpt-4".to_string(),
            provider_name: None,
        });
        let json = serde_json::to_string(&scope).unwrap();
        // Should serialize without provider_name field when None
        assert_eq!(json, r#"{"model_name":"gpt-4"}"#);
    }

    #[test]
    fn test_provider_tool_scope_serialize_unscoped() {
        let scope = ProviderToolScope::Unscoped;
        let json = serde_json::to_string(&scope).unwrap();
        assert_eq!(json, "null");
    }

    #[test]
    fn test_provider_tool_scope_matches_with_provider() {
        let scope = ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
            model_name: "gpt-4".to_string(),
            provider_name: Some("openai".to_string()),
        });
        assert!(scope.matches("gpt-4", "openai"));
        assert!(!scope.matches("gpt-4", "azure"));
        assert!(!scope.matches("claude-3", "openai"));
    }

    #[test]
    fn test_provider_tool_scope_matches_without_provider() {
        // When provider_name is None, should match any provider for the model
        let scope = ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
            model_name: "gpt-4".to_string(),
            provider_name: None,
        });
        assert!(scope.matches("gpt-4", "openai"));
        assert!(scope.matches("gpt-4", "azure"));
        assert!(scope.matches("gpt-4", "any-provider"));
        assert!(!scope.matches("claude-3", "anthropic"));
    }
}
