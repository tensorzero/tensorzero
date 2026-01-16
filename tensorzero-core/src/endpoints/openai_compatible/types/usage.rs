//! Usage tracking types for OpenAI-compatible API.
//!
//! This module provides types for token usage reporting in OpenAI-compatible responses,
//! including prompt tokens, completion tokens, and total token counts.

use serde::Serialize;

use crate::inference::types::{TensorzeroCacheHit, TensorzeroTokenDetails, Usage};

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct OpenAICompatibleUsage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,
    pub prompt_tokens_details: TensorzeroTokenDetails,
    pub completion_tokens_details: TensorzeroTokenDetails,
    pub tensorzero_cache_hit: TensorzeroCacheHit,
}

impl OpenAICompatibleUsage {
    pub fn zero() -> Self {
        Self {
            prompt_tokens: Some(0),
            completion_tokens: Some(0),
            total_tokens: Some(0),
            prompt_tokens_details: TensorzeroTokenDetails::zero(),
            completion_tokens_details: TensorzeroTokenDetails::zero(),
            tensorzero_cache_hit: TensorzeroCacheHit::No,
        }
    }

    /// Sum `OpenAICompatibleUsage` and `Usage` instances.
    /// `None` contaminates on both sides.
    pub fn sum_usage_strict(&mut self, other: &Usage) {
        self.prompt_tokens = match (self.prompt_tokens, other.input_tokens) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };

        self.completion_tokens = match (self.completion_tokens, other.output_tokens) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };

        self.total_tokens = match (self.total_tokens, other.total_tokens()) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };
    }
}

impl From<Usage> for OpenAICompatibleUsage {
    fn from(usage: Usage) -> Self {
        OpenAICompatibleUsage {
            prompt_tokens: usage.input_tokens,
            completion_tokens: usage.output_tokens,
            total_tokens: usage.total_tokens(),
            prompt_tokens_details: usage.input_tokens_details,
            completion_tokens_details: usage.output_tokens_details,
            tensorzero_cache_hit: usage.tensorzero_cache_hit,
        }
    }
}
