//! Usage tracking types for OpenAI-compatible API.
//!
//! This module provides types for token usage reporting in OpenAI-compatible responses,
//! including prompt tokens, completion tokens, and total token counts.

use serde::Serialize;

use crate::inference::types::usage::RawUsageEntry;
use crate::inference::types::{Usage, UsageWithRaw};

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct OpenAICompatibleUsage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,
}

/// OpenAI-compatible usage with optional raw provider-specific usage data.
/// Uses `#[serde(flatten)]` to inline the base usage fields, producing JSON like:
/// `{ "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "tensorzero_raw_usage": [...] }`
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct OpenAICompatibleUsageWithRaw {
    #[serde(flatten)]
    pub usage: OpenAICompatibleUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensorzero_raw_usage: Option<Vec<RawUsageEntry>>,
}

impl OpenAICompatibleUsage {
    pub fn zero() -> Self {
        Self {
            prompt_tokens: Some(0),
            completion_tokens: Some(0),
            total_tokens: Some(0),
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
        }
    }
}

impl From<UsageWithRaw> for OpenAICompatibleUsageWithRaw {
    fn from(usage_with_raw: UsageWithRaw) -> Self {
        OpenAICompatibleUsageWithRaw {
            usage: usage_with_raw.usage.into(),
            tensorzero_raw_usage: usage_with_raw.raw_usage,
        }
    }
}
