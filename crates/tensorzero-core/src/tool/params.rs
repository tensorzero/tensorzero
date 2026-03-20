//! Dynamic tool parameters for API requests.
//!
//! This module contains types for specifying tool parameters at runtime:
//! - `DynamicToolParams` - Wire format for tool configuration in API requests
//! - `BatchDynamicToolParams` - Batch version for multiple inferences

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tensorzero_derive::export_schema;

use crate::error::{Error, ErrorDetails};

use super::types::{ProviderTool, Tool};
use super::wire::ToolChoice;

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
/// 1. `dynamic_tools` -> `additional_tools`
/// 2. `allowed_tools` -> `allowed_tools` (based on choice enum)
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
    pub fn provider_tools(&self) -> Vec<ProviderTool> {
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
        if let Some(allowed_tools) = &allowed_tools
            && allowed_tools.len() != num_inferences
        {
            return Err(ErrorDetails::InvalidRequest {
                message: format!(
                    "allowed_tools vector length ({}) does not match number of inferences ({})",
                    allowed_tools.len(),
                    num_inferences
                ),
            }
            .into());
        }
        if let Some(additional_tools) = &additional_tools
            && additional_tools.len() != num_inferences
        {
            return Err(ErrorDetails::InvalidRequest {
                message: format!(
                    "additional_tools vector length ({}) does not match number of inferences ({})",
                    additional_tools.len(),
                    num_inferences
                ),
            }
            .into());
        }
        if let Some(tool_choice) = &tool_choice
            && tool_choice.len() != num_inferences
        {
            return Err(ErrorDetails::InvalidRequest {
                message: format!(
                    "tool_choice vector length ({}) does not match number of inferences ({})",
                    tool_choice.len(),
                    num_inferences
                ),
            }
            .into());
        }
        if let Some(parallel_tool_calls) = &parallel_tool_calls
            && parallel_tool_calls.len() != num_inferences
        {
            return Err(ErrorDetails::InvalidRequest {
                    message: format!(
                        "parallel_tool_calls vector length ({}) does not match number of inferences ({})",
                        parallel_tool_calls.len(),
                        num_inferences
                    ),
                }
                .into());
        }
        if let Some(provider_tools) = &provider_tools
            && provider_tools.len() != num_inferences
        {
            return Err(ErrorDetails::InvalidRequest {
                message: format!(
                    "provider_tools vector length ({}) does not match number of inferences ({})",
                    provider_tools.len(),
                    num_inferences
                ),
            }
            .into());
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
