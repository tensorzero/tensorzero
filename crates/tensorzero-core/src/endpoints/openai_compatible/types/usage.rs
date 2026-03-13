//! Usage tracking types for OpenAI-compatible API.
//!
//! This module provides types for token usage reporting in OpenAI-compatible responses,
//! including prompt tokens, completion tokens, and total token counts.

use rust_decimal::Decimal;
use serde::Serialize;

use crate::inference::types::Usage;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct OpenAICompatibleUsage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_input_tokens: Option<u32>,
    #[serde(with = "rust_decimal::serde::float_option")]
    pub tensorzero_cost: Option<Decimal>,
}

impl OpenAICompatibleUsage {
    pub fn zero() -> Self {
        Self {
            prompt_tokens: Some(0),
            completion_tokens: Some(0),
            total_tokens: Some(0),
            cache_read_input_tokens: Some(0),
            cache_write_input_tokens: Some(0),
            tensorzero_cost: Some(Decimal::ZERO),
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

        self.cache_read_input_tokens =
            match (self.cache_read_input_tokens, other.cache_read_input_tokens) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            };

        self.cache_write_input_tokens = match (
            self.cache_write_input_tokens,
            other.cache_write_input_tokens,
        ) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };

        self.tensorzero_cost = match (self.tensorzero_cost, other.cost) {
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
            cache_read_input_tokens: usage.cache_read_input_tokens,
            cache_write_input_tokens: usage.cache_write_input_tokens,
            tensorzero_cost: usage.cost,
        }
    }
}
