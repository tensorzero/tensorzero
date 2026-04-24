//! Batch variant of `DynamicToolParams`.
//!
//! The single-inference `DynamicToolParams` lives in
//! `tensorzero_inference_types::tool`; this module keeps the batch variant
//! (which depends on `tensorzero-core`'s `Error` type) here.

use serde::{Deserialize, Serialize};
use tensorzero_inference_types::ProviderTool;
use tensorzero_inference_types::tool::{DynamicToolParams, Tool};
use tensorzero_types::ToolChoice;

use crate::error::{Error, ErrorDetails};

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
