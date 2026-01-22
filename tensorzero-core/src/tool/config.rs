//! Tool configuration types for runtime inference.
//!
//! This module contains types that represent tool configuration at inference time:
//! - `ToolConfig` / `FunctionToolConfig` - Enum variants for different tool configurations
//! - `StaticToolConfig` / `DynamicToolConfig` - Tool configs from static config vs runtime
//! - `ToolCallConfig` - The main configuration passed to inference providers
//! - `AllowedTools` / `AllowedToolsChoice` - Which tools are allowed for a given inference

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, ErrorDetails};
use crate::jsonschema_util::JSONSchema;

use super::IMPLICIT_TOOL_DESCRIPTION;
use super::types::{FunctionTool, OpenAICustomTool, ProviderTool, Tool};
use super::wire::ToolChoice;

#[cfg(test)]
use super::params::DynamicToolParams;

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum ToolConfig {
    Function(FunctionToolConfig),
    OpenAICustom(OpenAICustomTool),
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum FunctionToolConfig {
    Static(Arc<StaticToolConfig>),
    Dynamic(DynamicToolConfig),
    Implicit(ImplicitToolConfig),
    DynamicImplicit(DynamicImplicitToolConfig),
}

/// Contains the configuration information for a specific tool
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[derive(Debug, PartialEq, Serialize)]
pub struct StaticToolConfig {
    pub description: String,
    pub parameters: JSONSchema,
    /// The display name sent to the LLM (can be overridden via config)
    pub name: String,
    /// The key used to reference this tool in allowed_tools and function config
    pub key: String,
    pub strict: bool,
}

/// Contains the configuration information for a tool defined at runtime
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[derive(Debug, PartialEq, Clone, Serialize)]
pub struct DynamicToolConfig {
    pub description: String,
    pub parameters: JSONSchema,
    pub name: String,
    pub strict: bool,
}

/// Contains the configuration information for a tool used in implicit tool calling for
/// JSON schema enforcement
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ImplicitToolConfig {
    pub parameters: JSONSchema,
}

/// Contains the configuration information for a tool used in implicit tool calling for
/// JSON schema enforcement for a JSON schema that is dynamically passed at inference time
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DynamicImplicitToolConfig {
    pub parameters: JSONSchema,
}

/// Records / lists the tools that were allowed in the request
/// Also lists how they were set (default, dynamically set)
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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
                // Filter by allowed_tools list (using key, not display name), then apply type filter
                Ok(Box::new(
                    self.static_tools_available
                        .iter()
                        .chain(self.dynamic_tools_available.iter())
                        .filter(|tool| self.allowed_tools.tools.iter().any(|t| t == tool.key())),
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

impl FunctionToolConfig {
    pub async fn validate_arguments(&self, arguments: &Value) -> Result<(), Error> {
        match self {
            FunctionToolConfig::Static(config) => config.parameters.validate(arguments).await,
            FunctionToolConfig::Dynamic(config) => config.parameters.validate(arguments).await,
            FunctionToolConfig::Implicit(config) => config.parameters.validate(arguments).await,
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
            FunctionToolConfig::Implicit(_config) => super::IMPLICIT_TOOL_NAME,
            FunctionToolConfig::DynamicImplicit(_config) => super::IMPLICIT_TOOL_NAME,
        }
    }

    /// Returns the key used to reference this tool in allowed_tools and function config.
    /// For static tools, this is the TOML table key. For dynamic tools, this is the same as the name.
    pub fn key(&self) -> &str {
        match self {
            FunctionToolConfig::Static(config) => &config.key,
            FunctionToolConfig::Dynamic(config) => &config.name,
            FunctionToolConfig::Implicit(_config) => super::IMPLICIT_TOOL_NAME,
            FunctionToolConfig::DynamicImplicit(_config) => super::IMPLICIT_TOOL_NAME,
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

// For now, this is required to convert to LegacyToolCallConfigDatabaseInsert for writing to the database
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
