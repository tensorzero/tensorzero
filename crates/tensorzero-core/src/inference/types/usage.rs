use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

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

/// A single entry in the raw usage array, representing usage data from one model inference.
/// This preserves the original provider-specific usage object for fields that TensorZero
/// normalizes away (e.g., OpenAI's `reasoning_tokens`, Anthropic's `provider_cache_read_input_tokens`).
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct RawUsageEntry {
    pub model_inference_id: Uuid,
    pub provider_type: String,
    pub api_type: ApiType,
    pub data: serde_json::Value,
}

pub fn raw_usage_entries_from_value(
    model_inference_id: Uuid,
    provider_type: &str,
    api_type: ApiType,
    usage: serde_json::Value,
) -> Vec<RawUsageEntry> {
    vec![RawUsageEntry {
        model_inference_id,
        provider_type: provider_type.to_string(),
        api_type,
        data: usage,
    }]
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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct Usage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    pub provider_cache_read_input_tokens: Option<u32>,
    pub provider_cache_write_input_tokens: Option<u32>,
    #[serde(default, with = "tensorzero_types::serde_utils::decimal_float_option")]
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

/// Take the cumulative max of two `Option<u32>` values from streaming chunks.
///
/// Returns `acc` unchanged if the chunk is `None`. Warns and retains the higher
/// value if the chunk is unexpectedly smaller (non-cumulative).
fn cumulative_max_u32(acc: Option<u32>, chunk: Option<u32>, field_name: &str) -> Option<u32> {
    match (acc, chunk) {
        (_, None) => acc,
        (None, v) => v,
        (Some(current), Some(new)) => {
            if current > new {
                tracing::warn!(
                    "Unexpected non-cumulative `{field_name}` in streaming response ({current} > {new}); using the higher value. Please open a bug report: https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports",
                );
                debug_assert!(
                    false,
                    "Unexpected non-cumulative `{field_name}` in streaming response ({current} > {new}); using the higher value."
                );
            }
            Some(current.max(new))
        }
    }
}

/// Same as [`cumulative_max_u32`] but for `Decimal` (used for cost).
fn cumulative_max_decimal(
    acc: Option<Decimal>,
    chunk: Option<Decimal>,
    field_name: &str,
) -> Option<Decimal> {
    match (acc, chunk) {
        (_, None) => acc,
        (None, v) => v,
        (Some(current), Some(new)) => {
            if current > new {
                tracing::warn!(
                    "Unexpected non-cumulative `{field_name}` in streaming response ({current} > {new}); using the higher value. Please open a bug report: https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports",
                );
                debug_assert!(
                    false,
                    "Unexpected non-cumulative `{field_name}` in streaming response ({current} > {new}); using the higher value."
                );
            }
            Some(current.max(new))
        }
    }
}

/// Aggregate `Usage` from a single streaming model inference.
///
/// Different model providers report usage differently while streaming:
///
/// - **OpenAI**: Sends a single chunk with usage at the end.
/// - **Anthropic**: Sends partial, cumulative usage objects across multiple chunks.
///
/// To handle these variations, we take the **maximum** value observed for each field,
/// returning `None` for any field that was never reported.
///
/// This approach works correctly for:
/// - Providers that send a single final usage chunk (the max is that value)
/// - Providers that send cumulative partial chunks (the max is the final cumulative value)
///
/// If a later chunk contains a *smaller* value than a previous chunk (suggesting
/// non-cumulative reporting), we log a warning and retain the higher value. No known
/// providers exhibit this behavior. If you encounter it, please file a bug report.
pub fn aggregate_usage_from_single_streaming_model_inference<I>(usages: I) -> Usage
where
    I: IntoIterator<Item = Usage>,
{
    usages
        .into_iter()
        .fold(Usage::default(), |mut acc, chunk_usage| {
            let Usage {
                input_tokens: chunk_input_tokens,
                output_tokens: chunk_output_tokens,
                provider_cache_read_input_tokens: chunk_cache_read,
                provider_cache_write_input_tokens: chunk_cache_write,
                cost: chunk_cost,
            } = chunk_usage;

            acc.input_tokens =
                cumulative_max_u32(acc.input_tokens, chunk_input_tokens, "input_tokens");
            acc.output_tokens =
                cumulative_max_u32(acc.output_tokens, chunk_output_tokens, "output_tokens");
            acc.provider_cache_read_input_tokens = cumulative_max_u32(
                acc.provider_cache_read_input_tokens,
                chunk_cache_read,
                "provider_cache_read_input_tokens",
            );
            acc.provider_cache_write_input_tokens = cumulative_max_u32(
                acc.provider_cache_write_input_tokens,
                chunk_cache_write,
                "provider_cache_write_input_tokens",
            );
            acc.cost = cumulative_max_decimal(acc.cost, chunk_cost, "cost");

            acc
        })
}

/// Aggregate `Usage` from multiple model inferences.
///
/// Used to combine usage from advanced variant types that invoke multiple models
/// (e.g., mixture-of-N: multiple candidates + fuser).
///
/// For each field (e.g. `input_tokens`), we sum the values across all inferences. However, if any
/// inference has `None` for a field, the aggregated result for that field is also `None`. In other
/// words, unknown values propagate per-field, not per-object.
pub fn aggregate_usage_across_model_inferences<I>(usages: I) -> Usage
where
    I: IntoIterator<Item = Usage>,
{
    usages.into_iter().fold(Usage::zero(), |acc, mi_usage| {
        let Usage {
            input_tokens: mi_input_tokens,
            output_tokens: mi_output_tokens,
            provider_cache_read_input_tokens: mi_cache_read,
            provider_cache_write_input_tokens: mi_cache_write,
            cost: mi_cost,
        } = mi_usage;

        Usage {
            input_tokens: match (acc.input_tokens, mi_input_tokens) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            },
            output_tokens: match (acc.output_tokens, mi_output_tokens) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            },
            // For cache tokens, None means "not reported by provider" — skip it
            // rather than dropping the entire aggregate.
            provider_cache_read_input_tokens: match (
                acc.provider_cache_read_input_tokens,
                mi_cache_read,
            ) {
                (Some(a), Some(b)) => Some(a + b),
                (Some(a), None) | (None, Some(a)) => Some(a),
                (None, None) => None,
            },
            provider_cache_write_input_tokens: match (
                acc.provider_cache_write_input_tokens,
                mi_cache_write,
            ) {
                (Some(a), Some(b)) => Some(a + b),
                (Some(a), None) | (None, Some(a)) => Some(a),
                (None, None) => None,
            },
            cost: match (acc.cost, mi_cost) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            },
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_type_serialization() {
        assert_eq!(
            serde_json::to_string(&ApiType::ChatCompletions).unwrap(),
            "\"chat_completions\""
        );
        assert_eq!(
            serde_json::to_string(&ApiType::Responses).unwrap(),
            "\"responses\""
        );
        assert_eq!(
            serde_json::to_string(&ApiType::Embeddings).unwrap(),
            "\"embeddings\""
        );
    }

    #[test]
    fn test_raw_usage_entry_serialization() {
        let entry = RawUsageEntry {
            model_inference_id: Uuid::nil(),
            provider_type: "openai".to_string(),
            api_type: ApiType::ChatCompletions,
            data: serde_json::json!({
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "reasoning_tokens": 30
            }),
        };
        let json = serde_json::to_value(&entry).unwrap();
        assert_eq!(json["provider_type"], "openai");
        assert_eq!(json["api_type"], "chat_completions");
        assert_eq!(json["data"]["prompt_tokens"], 100);
        assert_eq!(json["data"]["reasoning_tokens"], 30);
    }

    #[test]
    fn test_raw_usage_entry_null_data() {
        let entry = RawUsageEntry {
            model_inference_id: Uuid::nil(),
            provider_type: "openai".to_string(),
            api_type: ApiType::ChatCompletions,
            data: serde_json::Value::Null,
        };
        let json = serde_json::to_value(&entry).unwrap();
        // data field should be present as null
        assert_eq!(json["data"], serde_json::Value::Null, "data should be null");
    }

    // Tests for aggregate_usage_from_single_streaming_model_inference

    #[test]
    fn test_aggregate_usage_from_single_streaming_model_inference_empty() {
        let result = aggregate_usage_from_single_streaming_model_inference(vec![]);
        assert_eq!(
            result.input_tokens, None,
            "empty iterator should result in None for input_tokens"
        );
        assert_eq!(
            result.output_tokens, None,
            "empty iterator should result in None for output_tokens"
        );
    }

    #[test]
    fn test_aggregate_usage_from_single_streaming_model_inference_single_chunk() {
        let usage = Usage {
            input_tokens: Some(100),
            output_tokens: Some(50),
            provider_cache_read_input_tokens: None,
            provider_cache_write_input_tokens: None,
            cost: None,
        };
        let result = aggregate_usage_from_single_streaming_model_inference(vec![usage]);
        assert_eq!(
            result.input_tokens,
            Some(100),
            "single chunk input_tokens should be preserved"
        );
        assert_eq!(
            result.output_tokens,
            Some(50),
            "single chunk output_tokens should be preserved"
        );
    }

    #[test]
    fn test_aggregate_usage_from_single_streaming_model_inference_cumulative_chunks() {
        // Simulates Anthropic-style cumulative usage reporting
        let chunks = vec![
            Usage {
                input_tokens: Some(100),
                output_tokens: Some(10),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            },
            Usage {
                input_tokens: Some(100),
                output_tokens: Some(25),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            },
            Usage {
                input_tokens: Some(100),
                output_tokens: Some(50),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            },
        ];
        let result = aggregate_usage_from_single_streaming_model_inference(chunks);
        assert_eq!(
            result.input_tokens,
            Some(100),
            "cumulative input_tokens should take the max value"
        );
        assert_eq!(
            result.output_tokens,
            Some(50),
            "cumulative output_tokens should take the max value"
        );
    }

    #[test]
    fn test_aggregate_usage_from_single_streaming_model_inference_final_chunk_only() {
        // Simulates OpenAI-style single final usage chunk
        let chunks = vec![
            Usage {
                input_tokens: None,
                output_tokens: None,
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            },
            Usage {
                input_tokens: None,
                output_tokens: None,
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            },
            Usage {
                input_tokens: Some(200),
                output_tokens: Some(100),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            },
        ];
        let result = aggregate_usage_from_single_streaming_model_inference(chunks);
        assert_eq!(
            result.input_tokens,
            Some(200),
            "final chunk input_tokens should be used"
        );
        assert_eq!(
            result.output_tokens,
            Some(100),
            "final chunk output_tokens should be used"
        );
    }

    #[test]
    fn test_aggregate_usage_from_single_streaming_model_inference_all_none() {
        let chunks = vec![
            Usage {
                input_tokens: None,
                output_tokens: None,
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            },
            Usage {
                input_tokens: None,
                output_tokens: None,
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            },
        ];
        let result = aggregate_usage_from_single_streaming_model_inference(chunks);
        assert_eq!(
            result.input_tokens, None,
            "all None input_tokens should result in None"
        );
        assert_eq!(
            result.output_tokens, None,
            "all None output_tokens should result in None"
        );
    }

    #[test]
    fn test_aggregate_usage_from_single_streaming_model_inference_mixed_none_and_some() {
        let chunks = vec![
            Usage {
                input_tokens: Some(100),
                output_tokens: None,
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            },
            Usage {
                input_tokens: None,
                output_tokens: Some(50),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            },
        ];
        let result = aggregate_usage_from_single_streaming_model_inference(chunks);
        assert_eq!(
            result.input_tokens,
            Some(100),
            "input_tokens should preserve Some value even with later None"
        );
        assert_eq!(
            result.output_tokens,
            Some(50),
            "output_tokens should preserve Some value"
        );
    }

    #[test]
    #[should_panic(expected = "Unexpected non-cumulative")]
    fn test_aggregate_usage_from_single_streaming_model_inference_non_cumulative_uses_max() {
        // Tests the edge case where a later chunk has a smaller value.
        // This triggers a debug_assert! because it indicates unexpected provider behavior.
        let chunks = vec![
            Usage {
                input_tokens: Some(100),
                output_tokens: Some(50),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            },
            Usage {
                input_tokens: Some(80),  // Smaller than previous (unexpected)
                output_tokens: Some(30), // Smaller than previous (unexpected)
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            },
        ];
        // This will panic due to debug_assert! when non-cumulative values are detected
        let _ = aggregate_usage_from_single_streaming_model_inference(chunks);
    }

    // Tests for aggregate_usage_across_model_inferences

    #[test]
    fn test_aggregate_usage_across_model_inferences_empty() {
        let result = aggregate_usage_across_model_inferences(vec![]);
        assert_eq!(
            result.input_tokens,
            Some(0),
            "empty iterator should result in Some(0) for input_tokens (starts with Usage::zero())"
        );
        assert_eq!(
            result.output_tokens,
            Some(0),
            "empty iterator should result in Some(0) for output_tokens (starts with Usage::zero())"
        );
    }

    #[test]
    fn test_aggregate_usage_across_model_inferences_single_inference() {
        let usage = Usage {
            input_tokens: Some(100),
            output_tokens: Some(50),
            provider_cache_read_input_tokens: None,
            provider_cache_write_input_tokens: None,
            cost: None,
        };
        let result = aggregate_usage_across_model_inferences(vec![usage]);
        assert_eq!(
            result.input_tokens,
            Some(100),
            "single inference input_tokens should be summed with zero"
        );
        assert_eq!(
            result.output_tokens,
            Some(50),
            "single inference output_tokens should be summed with zero"
        );
    }

    #[test]
    fn test_aggregate_usage_across_model_inferences_multiple_all_some() {
        let usages = vec![
            Usage {
                input_tokens: Some(100),
                output_tokens: Some(50),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            },
            Usage {
                input_tokens: Some(200),
                output_tokens: Some(100),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            },
        ];
        let result = aggregate_usage_across_model_inferences(usages);
        assert_eq!(
            result.input_tokens,
            Some(300),
            "input_tokens should be sum of all inferences"
        );
        assert_eq!(
            result.output_tokens,
            Some(150),
            "output_tokens should be sum of all inferences"
        );
    }

    #[test]
    fn test_aggregate_usage_across_model_inferences_none_propagates() {
        let usages = vec![
            Usage {
                input_tokens: Some(100),
                output_tokens: Some(50),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            },
            Usage {
                input_tokens: None, // This should propagate None for input_tokens
                output_tokens: Some(100),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            },
        ];
        let result = aggregate_usage_across_model_inferences(usages);
        assert_eq!(
            result.input_tokens, None,
            "None in any inference should propagate to result"
        );
        assert_eq!(
            result.output_tokens,
            Some(150),
            "output_tokens should still sum when all are Some"
        );
    }

    #[test]
    fn test_aggregate_usage_across_model_inferences_both_none_propagate() {
        let usages = vec![
            Usage {
                input_tokens: None,
                output_tokens: Some(50),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            },
            Usage {
                input_tokens: Some(100),
                output_tokens: None,
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            },
        ];
        let result = aggregate_usage_across_model_inferences(usages);
        assert_eq!(
            result.input_tokens, None,
            "None input_tokens should propagate"
        );
        assert_eq!(
            result.output_tokens, None,
            "None output_tokens should propagate"
        );
    }

    #[test]
    fn test_aggregate_usage_across_model_inferences_all_none() {
        let usages = vec![
            Usage {
                input_tokens: None,
                output_tokens: None,
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            },
            Usage {
                input_tokens: None,
                output_tokens: None,
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            },
        ];
        let result = aggregate_usage_across_model_inferences(usages);
        assert_eq!(result.input_tokens, None, "all None should result in None");
        assert_eq!(result.output_tokens, None, "all None should result in None");
    }

    #[test]
    fn test_aggregate_usage_from_single_streaming_model_inference_anthropic_pattern() {
        // Simulates real Anthropic streaming behavior where:
        // - message_start sends input_tokens and initial output_tokens
        // - message_delta sends only output_tokens (input_tokens is not present)
        // This should NOT trigger the non-cumulative warning.
        let logs_contain = crate::utils::testing::capture_logs();

        let chunks = vec![
            // message_start chunk: has both input_tokens and output_tokens
            Usage {
                input_tokens: Some(69),
                output_tokens: Some(1),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            },
            // message_delta chunk: only output_tokens, no input_tokens
            Usage {
                input_tokens: None,
                output_tokens: Some(100),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            },
        ];

        let result = aggregate_usage_from_single_streaming_model_inference(chunks);

        assert_eq!(
            result.input_tokens,
            Some(69),
            "input_tokens should be 69 from first chunk"
        );
        assert_eq!(
            result.output_tokens,
            Some(100),
            "output_tokens should be 100 (max of cumulative values)"
        );

        // Verify NO warning was logged (this was the bug being fixed)
        assert!(
            !logs_contain("Unexpected non-cumulative"),
            "should NOT log a warning for Anthropic-style streaming where input_tokens is only in first chunk"
        );
    }

    // Tests for cost in streaming aggregation

    #[test]
    fn test_aggregate_single_streaming_with_cost() {
        let usage = Usage {
            input_tokens: Some(100),
            output_tokens: Some(50),
            provider_cache_read_input_tokens: None,
            provider_cache_write_input_tokens: None,
            cost: Some(Decimal::new(18, 5)), // 0.00018
        };
        let result = aggregate_usage_from_single_streaming_model_inference(vec![usage]);
        assert_eq!(
            result.cost,
            Some(Decimal::new(18, 5)),
            "single chunk cost should be preserved as-is"
        );
    }

    #[test]
    fn test_aggregate_single_streaming_cumulative_cost() {
        let chunks = vec![
            Usage {
                input_tokens: Some(100),
                output_tokens: Some(10),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: Some(Decimal::new(5, 5)), // 0.00005
            },
            Usage {
                input_tokens: Some(100),
                output_tokens: Some(25),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: Some(Decimal::new(12, 5)), // 0.00012
            },
            Usage {
                input_tokens: Some(100),
                output_tokens: Some(50),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: Some(Decimal::new(18, 5)), // 0.00018
            },
        ];
        let result = aggregate_usage_from_single_streaming_model_inference(chunks);
        assert_eq!(
            result.cost,
            Some(Decimal::new(18, 5)),
            "cumulative cost should take the max value"
        );
    }

    // Tests for cost in cross-inference aggregation

    #[test]
    fn test_aggregate_across_inferences_with_cost() {
        let usages = vec![
            Usage {
                input_tokens: Some(100),
                output_tokens: Some(50),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: Some(Decimal::new(18, 5)), // 0.00018
            },
            Usage {
                input_tokens: Some(200),
                output_tokens: Some(100),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: Some(Decimal::new(27, 5)), // 0.00027
            },
        ];
        let result = aggregate_usage_across_model_inferences(usages);
        assert_eq!(
            result.cost,
            Some(Decimal::new(45, 5)),
            "cost should be summed across inferences"
        );
    }

    #[test]
    fn test_aggregate_across_inferences_cache_both_some() {
        let usages = vec![
            Usage {
                input_tokens: Some(100),
                output_tokens: Some(50),
                provider_cache_read_input_tokens: Some(100),
                provider_cache_write_input_tokens: Some(10),
                cost: None,
            },
            Usage {
                input_tokens: Some(200),
                output_tokens: Some(100),
                provider_cache_read_input_tokens: Some(200),
                provider_cache_write_input_tokens: Some(20),
                cost: None,
            },
        ];
        let result = aggregate_usage_across_model_inferences(usages);
        assert_eq!(
            result.provider_cache_read_input_tokens,
            Some(300),
            "cache_read should be summed when both are Some"
        );
        assert_eq!(
            result.provider_cache_write_input_tokens,
            Some(30),
            "cache_write should be summed when both are Some"
        );
    }

    #[test]
    fn test_aggregate_across_inferences_cache_mixed_none() {
        let usages = vec![
            Usage {
                input_tokens: Some(100),
                output_tokens: Some(50),
                provider_cache_read_input_tokens: Some(100),
                provider_cache_write_input_tokens: None,
                cost: None,
            },
            Usage {
                input_tokens: Some(200),
                output_tokens: Some(100),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            },
        ];
        let result = aggregate_usage_across_model_inferences(usages);
        assert_eq!(
            result.provider_cache_read_input_tokens,
            Some(100),
            "cache_read should preserve known value when other is None (lenient)"
        );
        assert_eq!(
            result.provider_cache_write_input_tokens, None,
            "cache_write should stay None when both are None"
        );
    }

    #[test]
    fn test_aggregate_across_inferences_cache_independent_fields() {
        let usages = vec![
            Usage {
                input_tokens: Some(100),
                output_tokens: Some(50),
                provider_cache_read_input_tokens: Some(80),
                provider_cache_write_input_tokens: None,
                cost: None,
            },
            Usage {
                input_tokens: Some(200),
                output_tokens: Some(100),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: Some(40),
                cost: None,
            },
        ];
        let result = aggregate_usage_across_model_inferences(usages);
        assert_eq!(
            result.provider_cache_read_input_tokens,
            Some(80),
            "cache_read should preserve its own known value independently"
        );
        assert_eq!(
            result.provider_cache_write_input_tokens,
            Some(40),
            "cache_write should preserve its own known value independently"
        );
    }

    #[test]
    fn test_aggregate_streaming_cache_first_chunk_only() {
        // Anthropic pattern: first chunk reports cache tokens, later chunks do not
        let chunks = vec![
            Usage {
                input_tokens: Some(100),
                output_tokens: Some(1),
                provider_cache_read_input_tokens: Some(4000),
                provider_cache_write_input_tokens: Some(500),
                cost: None,
            },
            Usage {
                input_tokens: None,
                output_tokens: Some(50),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            },
            Usage {
                input_tokens: None,
                output_tokens: Some(100),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            },
        ];
        let result = aggregate_usage_from_single_streaming_model_inference(chunks);
        assert_eq!(
            result.provider_cache_read_input_tokens,
            Some(4000),
            "cache_read from first chunk should be preserved through later None chunks"
        );
        assert_eq!(
            result.provider_cache_write_input_tokens,
            Some(500),
            "cache_write from first chunk should be preserved through later None chunks"
        );
    }

    #[test]
    fn test_aggregate_streaming_cache_cumulative() {
        // Cumulative cache values across streaming chunks — should take max
        let chunks = vec![
            Usage {
                input_tokens: Some(100),
                output_tokens: Some(10),
                provider_cache_read_input_tokens: Some(2000),
                provider_cache_write_input_tokens: Some(300),
                cost: None,
            },
            Usage {
                input_tokens: Some(100),
                output_tokens: Some(50),
                provider_cache_read_input_tokens: Some(4000),
                provider_cache_write_input_tokens: Some(500),
                cost: None,
            },
        ];
        let result = aggregate_usage_from_single_streaming_model_inference(chunks);
        assert_eq!(
            result.provider_cache_read_input_tokens,
            Some(4000),
            "cumulative cache_read should take the max value"
        );
        assert_eq!(
            result.provider_cache_write_input_tokens,
            Some(500),
            "cumulative cache_write should take the max value"
        );
    }

    #[test]
    fn test_aggregate_across_inferences_cost_none_propagates() {
        let usages = vec![
            Usage {
                input_tokens: Some(100),
                output_tokens: Some(50),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: Some(Decimal::new(18, 5)), // 0.00018
            },
            Usage {
                input_tokens: Some(200),
                output_tokens: Some(100),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            },
        ];
        let result = aggregate_usage_across_model_inferences(usages);
        assert_eq!(
            result.cost, None,
            "None cost in any inference should propagate to result"
        );
    }
}
