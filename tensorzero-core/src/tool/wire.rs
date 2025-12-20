//! Extension traits and implementations for tool wire types.
//!
//! The core wire types (`ToolCall`, `ToolResult`, `ToolCallWrapper`, `ToolChoice`)
//! are re-exported from `tensorzero-types`. This module provides additional
//! functionality specific to `tensorzero-core`.

use crate::error::Error;
use crate::rate_limiting::{RateLimitedInputContent, get_estimated_tokens};

// Re-export types from tensorzero-types
pub use tensorzero_types::{ToolCall, ToolCallWrapper, ToolChoice, ToolResult};

/// Extension trait for `ToolCall` providing core-specific functionality.
pub trait ToolCallExt {
    /// Estimates the input token usage for rate limiting purposes.
    fn estimated_input_token_usage(&self) -> u64;
}

impl ToolCallExt for ToolCall {
    fn estimated_input_token_usage(&self) -> u64 {
        get_estimated_tokens(&self.name) + get_estimated_tokens(&self.arguments)
    }
}

// Implement RateLimitedInputContent for the re-exported type
impl RateLimitedInputContent for ToolCall {
    fn estimated_input_token_usage(&self) -> u64 {
        ToolCallExt::estimated_input_token_usage(self)
    }
}

/// Extension trait for `ToolResult` providing core-specific functionality.
pub trait ToolResultExt {
    /// Estimates the input token usage for rate limiting purposes.
    fn estimated_input_token_usage(&self) -> u64;
}

impl ToolResultExt for ToolResult {
    fn estimated_input_token_usage(&self) -> u64 {
        get_estimated_tokens(&self.name) + get_estimated_tokens(&self.result)
    }
}

// Implement RateLimitedInputContent for the re-exported type
impl RateLimitedInputContent for ToolResult {
    fn estimated_input_token_usage(&self) -> u64 {
        ToolResultExt::estimated_input_token_usage(self)
    }
}

/// Extension trait for `ToolCallWrapper` providing core-specific functionality.
pub trait ToolCallWrapperExt {
    /// Converts a `ToolCallWrapper` into a `ToolCall`.
    ///
    /// - `ToolCallWrapper::ToolCall`: passthrough
    /// - `ToolCallWrapper::InferenceResponseToolCall`: uses raw values, ignores parsed values
    fn into_tool_call(self) -> Result<ToolCall, Error>;
}

impl ToolCallWrapperExt for ToolCallWrapper {
    fn into_tool_call(self) -> Result<ToolCall, Error> {
        match self {
            ToolCallWrapper::ToolCall(tc) => Ok(tc),
            ToolCallWrapper::InferenceResponseToolCall(tc) => Ok(ToolCall {
                id: tc.id,
                name: tc.raw_name,
                arguments: tc.raw_arguments,
            }),
        }
    }
}
