use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::serde_utils::decimal_float_option;

/// The type of API used for a model inference.
/// Used in raw usage reporting to help consumers interpret provider-specific usage data.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(rename_all = "snake_case")]
pub enum ApiType {
    ChatCompletions,
    Responses,
    Embeddings,
    Other,
}

/// A single entry in the raw response array, representing raw response data from one model inference.
/// This preserves the original provider-specific response string that TensorZero normalizes.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct RawResponseEntry {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_inference_id: Option<Uuid>,
    pub provider_type: String,
    pub api_type: ApiType,
    pub data: String,
}

// =============================================================================
// Usage
// =============================================================================

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct Usage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    // Omit from serialized output when None (per AGENTS.md convention for optional fields).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_cache_read_input_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_cache_write_input_tokens: Option<u32>,
    #[serde(default, with = "decimal_float_option")]
    #[cfg_attr(feature = "ts-bindings", ts(type = "number | null"))]
    pub cost: Option<Decimal>,
}

impl Usage {
    /// Returns a `Usage` with core fields at zero and cache fields at `None`.
    ///
    /// Cache fields start as `None` (meaning "not reported") because not all
    /// providers support prompt caching. The lenient aggregation helpers
    /// (`aggregate_usage_across_model_inferences`, `sum_usage_strict`) will
    /// preserve any `Some` value they encounter rather than contaminating to `None`.
    pub fn zero() -> Usage {
        Usage {
            input_tokens: Some(0),
            output_tokens: Some(0),
            provider_cache_read_input_tokens: None,
            provider_cache_write_input_tokens: None,
            cost: Some(Decimal::ZERO),
        }
    }

    pub fn total_tokens(&self) -> Option<u32> {
        match (self.input_tokens, self.output_tokens) {
            (Some(input), Some(output)) => Some(input + output),
            _ => None,
        }
    }
}

// =============================================================================
// OpenAI-compatible usage types
// =============================================================================

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct OpenAIPromptTokensDetails {
    pub cached_tokens: Option<u32>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OpenAIUsage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    #[serde(default)]
    pub prompt_tokens_details: Option<OpenAIPromptTokensDetails>,
}

impl From<OpenAIUsage> for Usage {
    fn from(usage: OpenAIUsage) -> Self {
        Usage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            provider_cache_read_input_tokens: usage
                .prompt_tokens_details
                .and_then(|d| d.cached_tokens),
            provider_cache_write_input_tokens: None,
            cost: None,
        }
    }
}

impl From<Option<OpenAIUsage>> for Usage {
    fn from(usage: Option<OpenAIUsage>) -> Self {
        match usage {
            Some(u) => u.into(),
            None => Usage::default(),
        }
    }
}

// =============================================================================
// RawUsageEntry
// =============================================================================

/// A single entry in the raw usage array, representing usage data from one model inference.
/// This preserves the original provider-specific usage object for fields that TensorZero
/// normalizes away (e.g., OpenAI's `reasoning_tokens`, Anthropic's `cache_read_input_tokens`).
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct RawUsageEntry {
    pub model_inference_id: Uuid,
    pub provider_type: String,
    pub api_type: ApiType,
    pub data: Value,
}

pub fn raw_usage_entries_from_value(
    model_inference_id: Uuid,
    provider_type: &str,
    api_type: ApiType,
    usage: Value,
) -> Vec<RawUsageEntry> {
    vec![RawUsageEntry {
        model_inference_id,
        provider_type: provider_type.to_string(),
        api_type,
        data: usage,
    }]
}

// =============================================================================
// FinishReason
// =============================================================================

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize, sqlx::Type)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(rename_all = "snake_case")]
#[sqlx(type_name = "text", rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    StopSequence,
    Length,
    ToolCall,
    ContentFilter,
    Unknown,
}
