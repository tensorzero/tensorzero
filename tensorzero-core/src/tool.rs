use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;

#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::serialize_to_dict;
use crate::{
    error::{Error, ErrorDetails},
    jsonschema_util::{DynamicJSONSchema, StaticJSONSchema},
    rate_limiting::{get_estimated_tokens, RateLimitedInputContent},
};

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

/// A Tool object describes how a tool can be dynamically configured by the user.
#[derive(ts_rs::TS, Clone, Debug, Deserialize, PartialEq, Serialize)]
#[ts(export)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct Tool {
    pub description: String,
    pub parameters: Value,
    pub name: String,
    #[serde(default)]
    pub strict: bool,
}

impl std::fmt::Display for Tool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl Tool {
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

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[serde(untagged)]
pub enum ProviderToolScope {
    #[default]
    Unscoped,
    ModelProvider {
        model_name: String,
        model_provider_name: String,
    },
}

impl ProviderToolScope {
    fn matches(&self, scope_model_name: &str, scope_model_provider_name: &str) -> bool {
        match self {
            ProviderToolScope::Unscoped => true,
            ProviderToolScope::ModelProvider {
                model_name,
                model_provider_name,
            } => scope_model_name == model_name && scope_model_provider_name == model_provider_name,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
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
#[ts(export)]
pub struct AllowedTools {
    pub tools: Vec<String>,
    pub choice: AllowedToolsChoice,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, ts_rs::TS)]
#[ts(export)]
#[serde(rename_all = "snake_case")]
pub enum AllowedToolsChoice {
    // If `allowed_tools` is not explicitly passed, we set the function tools
    // by default and add any dynamic tools
    #[default]
    FunctionDefault,
    // If `allowed_tools` was explicitly passed we use that list only and then automatically add dynamically set tools
    DynamicAllowedTools,
    // We may add a third behavior if we deprecate the current default.
}

/// Contains all information required to tell an LLM what tools it can call
/// and what sorts of tool calls (parallel, none, etc) it is allowed to respond with.
/// Most inference providers can convert this into their desired tool format.
#[derive(Clone, Debug, Default, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct ToolCallConfig {
    pub(crate) static_tools_available: Vec<ToolConfig>,
    pub(crate) dynamic_tools_available: Vec<ToolConfig>,
    pub provider_tools: Vec<ProviderTool>,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: Option<bool>,
    pub allowed_tools: AllowedTools,
}

impl ToolCallConfig {
    pub fn new(
        function_tools: &[String],
        function_tool_choice: &ToolChoice,
        function_parallel_tool_calls: Option<bool>,
        static_tools: &HashMap<String, Arc<StaticToolConfig>>,
        dynamic_tool_params: DynamicToolParams,
    ) -> Result<Option<Self>, Error> {
        // If `allowed_tools` is not provided, use the function's configured tools.
        // This means we allow all tools for the function.
        let mut allowed_tools = match dynamic_tool_params.allowed_tools {
            Some(allowed_tools) => AllowedTools {
                tools: allowed_tools,
                choice: AllowedToolsChoice::DynamicAllowedTools,
            },
            None => AllowedTools {
                tools: function_tools.to_vec(),
                choice: AllowedToolsChoice::FunctionDefault,
            },
        };

        // Make a set for all names in additional tools
        let additional_tool_names: HashSet<&str> = dynamic_tool_params
            .additional_tools
            .as_ref()
            .map(|tools| tools.iter().map(|t| t.name.as_str()).collect())
            .unwrap_or_default();

        // Get each tool from the static tool config.
        // If a tool name is in allowed_tools but not in static_tools, check if it's a dynamic tool.
        // If it's neither static nor dynamic, throw an error.
        let static_tools_available: Vec<ToolConfig> = allowed_tools
            .tools
            .iter()
            .filter_map(|tool_name| {
                if let Some(static_tool) = static_tools.get(tool_name) {
                    // Found in static tools, add it
                    Some(Ok(ToolConfig::Static(static_tool.clone())))
                } else if additional_tool_names.contains(tool_name.as_str()) {
                    // Found in dynamic tools, skip it (will be added in the next loop)
                    None
                } else {
                    // Not found in either static or dynamic tools
                    Some(Err(Error::new(ErrorDetails::ToolNotFound {
                        name: tool_name.clone(),
                    })))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut dynamic_tools_available = vec![];
        if let Some(additional_tools) = dynamic_tool_params.additional_tools {
            for tool in additional_tools {
                // Today we automatically add dynamically configured tools to the allowed tools list but in future we may
                // change this behavior to be more in line with OpenAI's (if allowed_tools is set do not add tools.
                // This warning is unusable today.
                if !allowed_tools.tools.contains(&tool.name) {
                    tracing::info!(
                        tool_name = %tool.name,
                        "Currently, the gateway automatically includes all dynamic tools in the list of allowed tools. \
                         In a near-future release, dynamic tools will no longer be included automatically. \
                         If you intend for your dynamic tools to be allowed, please allow them explicitly; \
                         otherwise, disregard this warning."
                    );
                }
                dynamic_tools_available.push(ToolConfig::Dynamic(DynamicToolConfig {
                    description: tool.description,
                    parameters: DynamicJSONSchema::new(tool.parameters),
                    name: tool.name.clone(),
                    strict: tool.strict,
                }));
                allowed_tools.tools.push(tool.name);
            }
        }

        let mut tool_display_names = HashSet::new();

        // Check for duplicate tool names.
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

        let tool_choice = dynamic_tool_params
            .tool_choice
            .unwrap_or_else(|| function_tool_choice.clone());

        // If the tool choice is a specific tool, make sure it's in the list of available tools
        if let ToolChoice::Specific(tool_name) = &tool_choice {
            let tool_found = static_tools_available
                .iter()
                .chain(dynamic_tools_available.iter())
                .any(|tool| match tool {
                    ToolConfig::Static(config) => config.name == *tool_name,
                    ToolConfig::Dynamic(config) => config.name == *tool_name,
                    ToolConfig::Implicit(_) => false,
                    ToolConfig::DynamicImplicit(_) => false,
                });

            if !tool_found {
                return Err(ErrorDetails::ToolNotFound {
                    name: tool_name.clone(),
                }
                .into());
            }
        }

        let parallel_tool_calls = dynamic_tool_params
            .parallel_tool_calls
            .or(function_parallel_tool_calls);

        let tool_call_config_option = if static_tools_available.is_empty()
            && dynamic_tools_available.is_empty()
            && dynamic_tool_params.provider_tools.is_none()
        {
            None
        } else {
            Some(Self {
                static_tools_available,
                dynamic_tools_available,
                tool_choice,
                provider_tools: dynamic_tool_params.provider_tools.unwrap_or_default(),
                parallel_tool_calls,
                allowed_tools,
            })
        };

        Ok(tool_call_config_option)
    }

    /// Returns an iterator over references to all tools (both static and dynamic)
    pub fn tools_available(&self) -> impl Iterator<Item = &ToolConfig> {
        self.static_tools_available
            .iter()
            .chain(self.dynamic_tools_available.iter())
    }

    pub fn any_tools_available(&self) -> bool {
        !(self.static_tools_available.is_empty() && self.dynamic_tools_available.is_empty())
    }

    pub fn get_tool(&self, name: &str) -> Option<&ToolConfig> {
        self.tools_available().find(|tool_cfg| match tool_cfg {
            ToolConfig::Static(config) => config.name == name,
            ToolConfig::Dynamic(config) => config.name == name,
            ToolConfig::Implicit(_config) => false,
            ToolConfig::DynamicImplicit(_config) => false,
        })
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

    #[cfg(test)]
    pub fn with_tools_available(
        static_tools_available: Vec<ToolConfig>,
        dynamic_tools_available: Vec<ToolConfig>,
    ) -> Self {
        Self {
            static_tools_available,
            dynamic_tools_available,
            ..Default::default()
        }
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
/// - **To wire type**: Use `FunctionConfig::database_insert_to_dynamic_tool_params()` to convert `ToolCallConfigDatabaseInsert` → `DynamicToolParams`
/// - **To ToolCallConfig**: Use the `into_tool_call_config()` method for a direct conversion to `ToolCallConfig`
///
/// See also: [`DynamicToolParams`] for the wire/API format
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ToolCallConfigDatabaseInsert {
    /// All tools available for this inference (merged static + dynamic tools)
    pub tools_available: Vec<Tool>,
    /// The tool choice strategy
    pub tool_choice: ToolChoice,
    // TODO: decide what we want the Python interface to be for ToolChoice
    // This is complicated because ToolChoice is an enum with some simple arms and some
    // struct arms. We would likely need to land on one of the serde options for enums (tagged?)
    /// Whether parallel tool calls are enabled
    pub parallel_tool_calls: Option<bool>,
}

impl ToolCallConfigDatabaseInsert {
    /// Converts this database representation back into a `ToolCallConfig`.
    /// Errors if there are tools specified in the function that are not in the
    /// static tools (shouldn't happen). Or if the function is a JSON function (no tools).
    ///
    /// This method performs the reverse transformation of the lossy conversion that occurs
    /// when storing `ToolCallConfig` in the database. It reconstructs the tool configuration
    /// by:
    /// 1. Converting the stored tools into `DynamicToolParams`
    /// 2. Using the function config to prepare a full `ToolCallConfig`
    ///
    ///
    /// # Lossy Conversion
    /// Note that this conversion cannot fully restore the original `ToolCallConfig`:
    /// - `provider_tools` are not stored in the database and will be `None`
    /// - The distinction between static/dynamic tools is reconstructed based on function config
    /// This will be fixed in a follow-up PR.
    ///
    /// # Parameters
    /// - `function_config`: The function configuration containing static tool definitions
    /// - `static_tools`: Map of static tool names to their compiled configurations
    ///
    /// # Returns
    /// - `Ok(Some(ToolCallConfig))` if tools were configured
    /// - `Ok(None)` if no tools were available (e.g., JSON functions)
    /// - `Err(Error)` if reconstruction fails (e.g., tool not found, duplicate tools)
    ///
    /// # Example
    /// ```rust,ignore
    /// let db_insert = get_tool_config_from_database();
    /// let tool_config = db_insert.into_tool_call_config(&function_config, &static_tools)?;
    /// ```
    pub fn into_tool_call_config(
        self,
        function_config: &crate::function::FunctionConfig,
        static_tools: &HashMap<String, Arc<StaticToolConfig>>,
    ) -> Result<Option<ToolCallConfig>, Error> {
        let dynamic_params = function_config.database_insert_to_dynamic_tool_params(self);
        function_config.prepare_tool_config(dynamic_params, static_tools)
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
/// Converting from `ToolCallConfigDatabaseInsert` back to `DynamicToolParams` attempts to reconstruct the original:
/// 1. Tools that match function config tool names → `allowed_tools`
/// 2. Tools that don't match function config → `additional_tools`
/// 3. `provider_tools` is set to `None` (cannot be recovered)
///
/// Use `FunctionConfig::database_insert_to_dynamic_tool_params()` for this conversion.
///
/// # Example
/// ```rust,ignore
/// // API request with dynamic tool params
/// let params = DynamicToolParams {
///     allowed_tools: Some(vec!["calculator".to_string()]),  // Use only the calculator tool from config
///     additional_tools: Some(vec![Tool {  runtime tool  }]),  // Add a new tool
///     tool_choice: Some(ToolChoice::Required),
///     parallel_tool_calls: Some(true),
///     provider_tools: None,
/// };
///
/// // Convert to storage format (merge tools, lose distinction)
/// let db_insert = function_config
///     .dynamic_tool_params_to_database_insert(params, &static_tools)?
///     .unwrap_or_default();
///
/// // db_insert.tools_available now contains both the calculator tool (from config)
/// // and the runtime tool (from additional_tools), merged together
/// ```
///
/// See also: [`ToolCallConfigDatabaseInsert`] for the storage/database format
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
#[derive(ts_rs::TS)]
#[ts(optional_fields)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct DynamicToolParams {
    /// A subset of static tools configured for the function that the inference is allowed to use. Optional.
    /// If not provided, all static tools are allowed.
    pub allowed_tools: Option<Vec<String>>,

    /// Tools that the user provided at inference time (not in function config), in addition to the function-configured
    /// tools, that are also allowed.
    pub additional_tools: Option<Vec<Tool>>,
    /// User-specified tool choice strategy. If provided during inference, it will override the function-configured tool choice.
    /// Optional.
    pub tool_choice: Option<ToolChoice>,

    /// Whether to use parallel tool calls in the inference. Optional.
    /// If provided during inference, it will override the function-configured parallel tool calls.
    pub parallel_tool_calls: Option<bool>,

    /// Provider-specific tool configurations (not persisted to database)
    pub provider_tools: Option<Vec<ProviderTool>>,
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
    pub fn provider_tools(&self) -> Option<Vec<ProviderTool>> {
        self.provider_tools.clone()
    }

    pub fn __repr__(&self) -> String {
        self.to_string()
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

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(get_all, str))]
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
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(untagged)]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
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
        let implicit_tool_config = ToolConfig::Implicit(ImplicitToolConfig { parameters });
        Self {
            static_tools_available: vec![implicit_tool_config],
            dynamic_tools_available: vec![],
            tool_choice: ToolChoice::Specific(IMPLICIT_TOOL_NAME.to_string()),
            parallel_tool_calls: None,
            provider_tools: vec![],
            allowed_tools: AllowedTools::default(),
        }
    }
}

/// A ToolResult is the outcome of a ToolCall, which we may want to present back to the model
#[cfg_attr(feature = "pyo3", pyclass(get_all, str))]
#[derive(ts_rs::TS, Clone, Debug, Deserialize, PartialEq, Serialize)]
#[ts(export)]
#[serde(deny_unknown_fields)]
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
#[derive(ts_rs::TS, Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[ts(export)]
#[serde(rename_all = "lowercase")]
#[serde(deny_unknown_fields)]
pub enum ToolChoice {
    None,
    #[default]
    Auto,
    Required,
    // Forces the LLM to call a specific tool. The String is the name of the tool.
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

impl ToolConfig {
    pub async fn validate_arguments(&self, arguments: &Value) -> Result<(), Error> {
        match self {
            ToolConfig::Static(config) => config.parameters.validate(arguments),
            ToolConfig::Dynamic(config) => config.parameters.validate(arguments).await,
            ToolConfig::Implicit(config) => config.parameters.validate(arguments),
            ToolConfig::DynamicImplicit(config) => config.parameters.validate(arguments).await,
        }
    }

    pub fn description(&self) -> &str {
        match self {
            ToolConfig::Static(config) => &config.description,
            ToolConfig::Dynamic(config) => &config.description,
            ToolConfig::Implicit(_config) => IMPLICIT_TOOL_DESCRIPTION,
            ToolConfig::DynamicImplicit(_config) => IMPLICIT_TOOL_DESCRIPTION,
        }
    }

    pub fn parameters(&self) -> &Value {
        match self {
            ToolConfig::Static(config) => &config.parameters.value,
            ToolConfig::Dynamic(config) => &config.parameters.value,
            ToolConfig::Implicit(config) => &config.parameters.value,
            ToolConfig::DynamicImplicit(config) => &config.parameters.value,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            ToolConfig::Static(config) => &config.name,
            ToolConfig::Dynamic(config) => &config.name,
            ToolConfig::Implicit(_config) => IMPLICIT_TOOL_NAME,
            ToolConfig::DynamicImplicit(_config) => IMPLICIT_TOOL_NAME,
        }
    }

    pub fn strict(&self) -> bool {
        match self {
            ToolConfig::Static(config) => config.strict,
            ToolConfig::Dynamic(config) => config.strict,
            ToolConfig::Implicit(_config) => false,
            ToolConfig::DynamicImplicit(_config) => false,
        }
    }
}

impl From<ToolCallConfig> for ToolCallConfigDatabaseInsert {
    fn from(tool_call_config: ToolCallConfig) -> Self {
        Self {
            tools_available: tool_call_config
                .static_tools_available
                .into_iter()
                .chain(tool_call_config.dynamic_tools_available)
                .map(ToolConfig::into)
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
            strict: tool_config.strict(),
        }
    }
}

pub fn create_dynamic_implicit_tool_config(schema: Value) -> ToolCallConfig {
    let tool_schema = DynamicJSONSchema::new(schema);
    let implicit_tool = ToolConfig::DynamicImplicit(DynamicImplicitToolConfig {
        parameters: tool_schema,
    });
    ToolCallConfig {
        static_tools_available: vec![],
        dynamic_tools_available: vec![implicit_tool],
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
                    provider_tools: None,
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
                provider_tools: provider_tools_iter.next().unwrap_or(None),
            });
        }
        Ok(all_dynamic_tool_params)
    }
}

/// For use in initializing JSON functions
/// Creates a ToolCallConfig with a single implicit tool that takes the schema as arguments
pub fn create_implicit_tool_call_config(schema: StaticJSONSchema) -> ToolCallConfig {
    create_implicit_tool_call_config_with_allowed_tools(schema, AllowedTools::default())
}

pub fn create_implicit_tool_call_config_with_allowed_tools(
    schema: StaticJSONSchema,
    allowed_tools: AllowedTools,
) -> ToolCallConfig {
    let implicit_tool = ToolConfig::Implicit(ImplicitToolConfig { parameters: schema });
    ToolCallConfig {
        static_tools_available: vec![implicit_tool],
        dynamic_tools_available: vec![],
        tool_choice: ToolChoice::Specific(IMPLICIT_TOOL_NAME.to_string()),
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
        let tool_call_config = ToolCallConfig::new(
            &EMPTY_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
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
            Some(true),
            &TOOLS,
            DynamicToolParams::default(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(tool_call_config.tools_available().count(), 2);
        assert_eq!(tool_call_config.tool_choice, ToolChoice::Auto);
        assert_eq!(tool_call_config.parallel_tool_calls, Some(true));
        let tools: Vec<_> = tool_call_config.tools_available().collect();
        assert!(tools[0].strict());
        assert!(!tools[1].strict());

        // Empty tools in function and config but we specify an allowed tool (should fail)
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec!["get_temperature".to_string()]),
            ..Default::default()
        };
        let err = ToolCallConfig::new(
            &EMPTY_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &EMPTY_TOOLS,
            dynamic_tool_params,
        )
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
        let tool_call_config = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        )
        .unwrap()
        .unwrap();
        assert_eq!(tool_call_config.tools_available().count(), 2);
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
        let err = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        )
        .unwrap_err();
        assert_eq!(
            err,
            ErrorDetails::ToolNotFound {
                name: "establish_campground".to_string()
            }
            .into()
        );

        // We pass an empty list of allowed tools and then configure a new tool
        // This should remove all configured tools and add the new tool
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec![]),
            additional_tools: Some(vec![Tool {
                name: "establish_campground".to_string(),
                description: "Establish a campground".to_string(),
                parameters: json!({}),
                strict: false,
            }]),
            ..Default::default()
        };
        let tool_call_config = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        )
        .unwrap()
        .unwrap();
        assert_eq!(tool_call_config.tools_available().count(), 1);
        let first_tool = tool_call_config.tools_available().next().unwrap();
        assert_eq!(first_tool.name(), "establish_campground");
        assert!(!first_tool.strict());

        // We pass a list of a single allowed tool and then configure a new tool
        // This should remove the other configured tools and add the new tool
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec!["get_temperature".to_string()]),
            additional_tools: Some(vec![Tool {
                name: "establish_campground".to_string(),
                description: "Establish a campground".to_string(),
                parameters: json!({}),
                strict: false,
            }]),
            parallel_tool_calls: Some(false),
            ..Default::default()
        };
        let tool_call_config = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        )
        .unwrap()
        .unwrap();
        assert_eq!(tool_call_config.tools_available().count(), 2);
        // The following code depends on an implementation detail for this ordering,
        // might break if we change the order
        let tools: Vec<_> = tool_call_config.tools_available().collect();
        assert_eq!(tools[0].name(), "get_temperature");
        assert_eq!(tools[1].name(), "establish_campground");
        assert_eq!(tool_call_config.parallel_tool_calls, Some(false));

        // We pass a list of no allowed tools and then configure a new tool
        // This should remove all configured tools and add the new tool
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec![]),
            additional_tools: Some(vec![Tool {
                name: "establish_campground".to_string(),
                description: "Establish a campground".to_string(),
                parameters: json!({}),
                strict: false,
            }]),
            tool_choice: Some(ToolChoice::Specific("establish_campground".to_string())),
            ..Default::default()
        };
        let tool_call_config = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        )
        .unwrap()
        .unwrap();
        assert_eq!(tool_call_config.tools_available().count(), 1);
        let first_tool = tool_call_config.tools_available().next().unwrap();
        assert_eq!(first_tool.name(), "establish_campground");
        assert_eq!(tool_call_config.parallel_tool_calls, Some(true));
        assert_eq!(
            tool_call_config.tool_choice,
            ToolChoice::Specific("establish_campground".to_string())
        );
        assert!(!first_tool.strict());
    }

    #[tokio::test]
    async fn test_inference_response_tool_call_new() {
        let tool_call = ToolCall {
            name: "get_temperature".to_string(),
            arguments: "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}".to_string(),
            id: "123".to_string(),
        };
        let tool_call_config = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            DynamicToolParams::default(),
        )
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
        let tool_call_config = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            DynamicToolParams {
                additional_tools: Some(vec![Tool {
                    name: "establish_campground".to_string(),
                    description: "Establish a campground".to_string(),
                    parameters: json!({"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}),
                    strict: false,
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
            additional_tools: Some(vec![Tool {
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
            }]),
            ..Default::default()
        };

        let err = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        )
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
                scope: ProviderToolScope::ModelProvider {
                    model_name: "gpt-4".to_string(),
                    model_provider_name: "openai".to_string(),
                },
                tool: json!({"type": "gpt4_tool"}),
            },
            ProviderTool {
                scope: ProviderToolScope::ModelProvider {
                    model_name: "claude-3".to_string(),
                    model_provider_name: "anthropic".to_string(),
                },
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
            additional_tools: Some(vec![Tool {
                name: "establish_campground".to_string(),
                description: "Establish a campground".to_string(),
                parameters: json!({"type": "object", "properties": {"location": {"type": "string"}}}),
                strict: false,
            }]),
            ..Default::default()
        };

        let tool_call_config = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        )
        .unwrap()
        .unwrap();

        // Should have both static and dynamic tools
        assert_eq!(tool_call_config.tools_available().count(), 2);

        // Verify the static tool is included
        assert!(tool_call_config
            .tools_available()
            .any(|t| t.name() == "get_temperature"));

        // Verify the dynamic tool is included
        assert!(tool_call_config
            .tools_available()
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
            additional_tools: Some(vec![Tool {
                name: "establish_campground".to_string(),
                description: "Establish a campground".to_string(),
                parameters: json!({"type": "object"}),
                strict: false,
            }]),
            ..Default::default()
        };

        let err = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        )
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
    async fn test_dynamic_tool_auto_added_with_warning() {
        let logs_contain = crate::utils::testing::capture_logs();
        // Test that dynamic tools are still auto-added even when not in allowed_tools (with warning)
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec!["get_temperature".to_string()]),
            additional_tools: Some(vec![Tool {
                name: "establish_campground".to_string(),
                description: "Establish a campground".to_string(),
                parameters: json!({"type": "object", "properties": {"location": {"type": "string"}}}),
                strict: false,
            }]),
            ..Default::default()
        };

        let tool_call_config = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        )
        .unwrap()
        .unwrap();

        // Both tools should be included (dynamic tool auto-added despite not being in allowed_tools)
        assert_eq!(tool_call_config.tools_available().count(), 2);
        assert!(tool_call_config
            .tools_available()
            .any(|t| t.name() == "get_temperature"));
        assert!(tool_call_config
            .tools_available()
            .any(|t| t.name() == "establish_campground"));

        // Check that warning was logged
        assert!(logs_contain(
            "Currently, the gateway automatically includes all dynamic tools"
        ));
    }
}
