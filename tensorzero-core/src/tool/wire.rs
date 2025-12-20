//! Extension traits and implementations for tool wire types.
//!
//! The core wire types (`ToolCall`, `ToolResult`, `ToolCallWrapper`, `ToolChoice`)
//! are re-exported from `tensorzero-types`. This module provides additional
//! functionality specific to `tensorzero-core`.

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
