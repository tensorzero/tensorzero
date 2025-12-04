//! Database storage types for tool configuration.
//!
//! This module contains types for persisting tool configuration to the database:
//! - `ToolCallConfigDatabaseInsert` - The storage format for tool configurations
//! - `LegacyToolCallConfigDatabaseInsert` - Legacy format for backward compatibility
//! - Custom deserializers for reading from ClickHouse

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::config::Config;
use crate::endpoints::datasets::v1::types::UpdateDynamicToolParamsRequest;
use crate::error::Error;
use crate::function::FunctionConfig;

use super::config::{
    AllowedTools, AllowedToolsChoice, FunctionToolConfig, StaticToolConfig, ToolCallConfig,
    ToolCallConfigConstructorArgs,
};
use super::params::DynamicToolParams;
use super::types::{FunctionTool, ProviderTool, Tool};
use super::wire::ToolChoice;

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
/// - **From wire type**: Use `FunctionConfig::dynamic_tool_params_to_database_insert()` to convert `DynamicToolParams` -> `ToolCallConfigDatabaseInsert`
/// - **To wire type**: Use `From<ToolCallConfigDatabaseInsert> for DynamicToolParams` trait to convert `ToolCallConfigDatabaseInsert` -> `DynamicToolParams`
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
///    - Parsing JSON strings into nested types (e.g., `Vec<String>` � `Vec<Tool>`)
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
    /// - If `choice == FunctionDefault` � `allowed_tools = None` (use function defaults)
    /// - If `choice == DynamicAllowedTools` � `allowed_tools = Some(tools)` (explicit override)
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

pub(super) fn tool_call_config_to_legacy_tool_database_insert(
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
