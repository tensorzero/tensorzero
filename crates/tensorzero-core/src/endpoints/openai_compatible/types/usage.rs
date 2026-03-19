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
    pub provider_cache_read_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_cache_write_input_tokens: Option<u32>,
    #[serde(with = "rust_decimal::serde::float_option")]
    pub tensorzero_cost: Option<Decimal>,
}

impl OpenAICompatibleUsage {
    pub fn zero() -> Self {
        Self {
            prompt_tokens: Some(0),
            completion_tokens: Some(0),
            total_tokens: Some(0),
            provider_cache_read_input_tokens: Some(0),
            provider_cache_write_input_tokens: Some(0),
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

        // For cache tokens, None means "not reported by provider" — preserve
        // the known value rather than dropping the entire aggregate.
        self.provider_cache_read_input_tokens = match (
            self.provider_cache_read_input_tokens,
            other.provider_cache_read_input_tokens,
        ) {
            (Some(a), Some(b)) => Some(a + b),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        };

        self.provider_cache_write_input_tokens = match (
            self.provider_cache_write_input_tokens,
            other.provider_cache_write_input_tokens,
        ) {
            (Some(a), Some(b)) => Some(a + b),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
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
            provider_cache_read_input_tokens: usage.provider_cache_read_input_tokens,
            provider_cache_write_input_tokens: usage.provider_cache_write_input_tokens,
            tensorzero_cost: usage.cost,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;

    #[gtest]
    fn test_sum_usage_strict_both_some() {
        let mut usage = OpenAICompatibleUsage::zero();
        let other = Usage {
            input_tokens: Some(10),
            output_tokens: Some(20),
            provider_cache_read_input_tokens: Some(5),
            provider_cache_write_input_tokens: Some(3),
            cost: Some(Decimal::new(1, 2)),
        };
        usage.sum_usage_strict(&other);
        expect_that!(usage.prompt_tokens, some(eq(10)));
        expect_that!(usage.completion_tokens, some(eq(20)));
        expect_that!(usage.total_tokens, some(eq(30)));
        expect_that!(usage.provider_cache_read_input_tokens, some(eq(5)));
        expect_that!(usage.provider_cache_write_input_tokens, some(eq(3)));
    }

    #[gtest]
    fn test_sum_usage_strict_cache_none_preserves_known_value() {
        // Start with known cache values, sum with None cache — should preserve existing
        let mut usage = OpenAICompatibleUsage {
            prompt_tokens: Some(100),
            completion_tokens: Some(50),
            total_tokens: Some(150),
            provider_cache_read_input_tokens: Some(80),
            provider_cache_write_input_tokens: Some(20),
            tensorzero_cost: Some(Decimal::new(5, 2)),
        };
        let other = Usage {
            input_tokens: Some(10),
            output_tokens: Some(5),
            provider_cache_read_input_tokens: None,
            provider_cache_write_input_tokens: None,
            cost: Some(Decimal::new(1, 2)),
        };
        usage.sum_usage_strict(&other);
        expect_that!(usage.prompt_tokens, some(eq(110)));
        expect_that!(
            usage.provider_cache_read_input_tokens,
            some(eq(80)),
            "None cache should preserve the existing value, not contaminate to None"
        );
        expect_that!(
            usage.provider_cache_write_input_tokens,
            some(eq(20)),
            "None cache should preserve the existing value, not contaminate to None"
        );
    }

    #[gtest]
    fn test_sum_usage_strict_cache_none_from_accumulator_preserves_other() {
        // Start with None cache, sum with Some — should pick up the value
        let mut usage = OpenAICompatibleUsage {
            prompt_tokens: Some(0),
            completion_tokens: Some(0),
            total_tokens: Some(0),
            provider_cache_read_input_tokens: None,
            provider_cache_write_input_tokens: None,
            tensorzero_cost: Some(Decimal::ZERO),
        };
        let other = Usage {
            input_tokens: Some(10),
            output_tokens: Some(5),
            provider_cache_read_input_tokens: Some(8),
            provider_cache_write_input_tokens: Some(2),
            cost: Some(Decimal::new(1, 2)),
        };
        usage.sum_usage_strict(&other);
        expect_that!(
            usage.provider_cache_read_input_tokens,
            some(eq(8)),
            "Should pick up cache value from other when accumulator is None"
        );
        expect_that!(
            usage.provider_cache_write_input_tokens,
            some(eq(2)),
            "Should pick up cache value from other when accumulator is None"
        );
    }

    #[gtest]
    fn test_sum_usage_strict_both_none_cache_stays_none() {
        let mut usage = OpenAICompatibleUsage::default();
        let other = Usage {
            input_tokens: Some(10),
            output_tokens: Some(5),
            provider_cache_read_input_tokens: None,
            provider_cache_write_input_tokens: None,
            cost: None,
        };
        usage.sum_usage_strict(&other);
        expect_that!(usage.provider_cache_read_input_tokens, none());
        expect_that!(usage.provider_cache_write_input_tokens, none());
    }

    #[gtest]
    fn test_sum_usage_strict_input_tokens_none_contaminates() {
        // For non-cache fields, None should contaminate (strict behavior)
        let mut usage = OpenAICompatibleUsage::zero();
        let other = Usage {
            input_tokens: None,
            output_tokens: Some(5),
            provider_cache_read_input_tokens: None,
            provider_cache_write_input_tokens: None,
            cost: None,
        };
        usage.sum_usage_strict(&other);
        expect_that!(
            usage.prompt_tokens,
            none(),
            "None input_tokens should contaminate to None"
        );
        expect_that!(usage.completion_tokens, some(eq(5)));
    }
}
