use serde::{Deserialize, Serialize};

/// Detailed breakdown of input tokens for cost and performance analysis.
/// 
/// This structure captures provider-specific token categorization for input/prompt tokens.
/// All fields are optional to support providers with varying levels of detail.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct InputTokensDetails {
    /// Number of tokens served from cache (reducing cost and latency).
    /// - OpenAI: Direct from `prompt_tokens_details.cached_tokens`
    /// - Anthropic: Maps from `cache_read_input_tokens`
    /// - GCP Vertex Gemini: Maps from `cachedContentTokenCount`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
    /// Number of audio input tokens (for multimodal models).
    /// Currently supported by OpenAI for audio input processing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u32>,
}

/// Detailed breakdown of output tokens for cost and performance analysis.
/// 
/// This structure captures provider-specific token categorization for completion/output tokens.
/// All fields are optional to support providers with varying levels of detail.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct OutputTokensDetails {
    /// Number of tokens used for reasoning/thinking (e.g., o1/o3 models).
    /// These tokens represent the model's internal reasoning process.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
    /// Number of audio output tokens (for multimodal models).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u32>,
    /// Number of tokens accepted from speculative decoding/assisted generation.
    /// Indicates successful prediction optimization.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accepted_prediction_tokens: Option<u32>,
    /// Number of tokens rejected from speculative decoding/assisted generation.
    /// Indicates prediction misses that required re-generation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejected_prediction_tokens: Option<u32>,
}

/// Token usage information with optional detailed breakdowns.
/// 
/// This is the core usage type used throughout TensorZero for tracking token consumption.
/// It supports basic token counts plus optional detailed categorization for providers
/// that expose granular usage information (e.g., cached tokens, reasoning tokens).
/// 
/// Provider mapping examples:
/// - OpenAI: Full detail support for both input and output tokens
/// - Anthropic: Maps cache tokens to input_tokens_details.cached_tokens
/// - GCP Vertex Gemini: Maps cached content tokens to input_tokens_details.cached_tokens
/// - Azure OpenAI: Inherits OpenAI's full detail support
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
#[cfg_attr(test, ts(optional_fields))]
pub struct Usage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    /// Detailed breakdown of input token categories (cached, audio, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<InputTokensDetails>,
    /// Detailed breakdown of output token categories (reasoning, prediction, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,
}

impl Usage {
    pub fn zero() -> Usage {
        Usage {
            input_tokens: Some(0),
            output_tokens: Some(0),
            input_tokens_details: Some(InputTokensDetails {
                cached_tokens: Some(0),
                audio_tokens: Some(0),
            }),
            output_tokens_details: Some(OutputTokensDetails {
                reasoning_tokens: Some(0),
                audio_tokens: Some(0),
                accepted_prediction_tokens: Some(0),
                rejected_prediction_tokens: Some(0),
            }),
        }
    }

    pub fn total_tokens(&self) -> Option<u32> {
        match (self.input_tokens, self.output_tokens) {
            (Some(input), Some(output)) => Some(input + output),
            _ => None,
        }
    }

    /// Sum an iterator of Usage values.
    /// If any usage has None for a field, the sum for that field becomes None.
    pub fn sum_iter_strict<I: Iterator<Item = Usage>>(iter: I) -> Usage {
        iter.fold(Usage::zero(), |acc, usage| Usage {
            input_tokens: match (acc.input_tokens, usage.input_tokens) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            },
            output_tokens: match (acc.output_tokens, usage.output_tokens) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            },
            input_tokens_details: sum_input_details_strict(
                acc.input_tokens_details,
                usage.input_tokens_details,
            ),
            output_tokens_details: sum_output_details_strict(
                acc.output_tokens_details,
                usage.output_tokens_details,
            ),
        })
    }

    /// Sum two `Usage` instances.
    /// `None` contaminates on both sides.
    pub fn sum_strict(&mut self, other: &Usage) {
        self.input_tokens = match (self.input_tokens, other.input_tokens) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };

        self.output_tokens = match (self.output_tokens, other.output_tokens) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };

        self.input_tokens_details =
            sum_input_details_strict(self.input_tokens_details, other.input_tokens_details);
        self.output_tokens_details =
            sum_output_details_strict(self.output_tokens_details, other.output_tokens_details);
    }
}

/// Sum two `InputTokensDetails` instances.
/// `None` contaminates on both sides.
fn sum_input_details_strict(
    acc: Option<InputTokensDetails>,
    other: Option<InputTokensDetails>,
) -> Option<InputTokensDetails> {
    match (acc, other) {
        (Some(a), Some(b)) => Some(InputTokensDetails {
            cached_tokens: match (a.cached_tokens, b.cached_tokens) {
                (Some(x), Some(y)) => Some(x + y),
                _ => None,
            },
            audio_tokens: match (a.audio_tokens, b.audio_tokens) {
                (Some(x), Some(y)) => Some(x + y),
                _ => None,
            },
        }),
        _ => None,
    }
}

/// Sum two `OutputTokensDetails` instances.
/// `None` contaminates on both sides.
fn sum_output_details_strict(
    acc: Option<OutputTokensDetails>,
    other: Option<OutputTokensDetails>,
) -> Option<OutputTokensDetails> {
    match (acc, other) {
        (Some(a), Some(b)) => Some(OutputTokensDetails {
            reasoning_tokens: match (a.reasoning_tokens, b.reasoning_tokens) {
                (Some(x), Some(y)) => Some(x + y),
                _ => None,
            },
            audio_tokens: match (a.audio_tokens, b.audio_tokens) {
                (Some(x), Some(y)) => Some(x + y),
                _ => None,
            },
            accepted_prediction_tokens: match (
                a.accepted_prediction_tokens,
                b.accepted_prediction_tokens,
            ) {
                (Some(x), Some(y)) => Some(x + y),
                _ => None,
            },
            rejected_prediction_tokens: match (
                a.rejected_prediction_tokens,
                b.rejected_prediction_tokens,
            ) {
                (Some(x), Some(y)) => Some(x + y),
                _ => None,
            },
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_sum_all_some() {
        let usages = vec![
            Usage {
                input_tokens: Some(10),
                output_tokens: Some(20),
                ..Default::default()
            },
            Usage {
                input_tokens: Some(5),
                output_tokens: Some(15),
                ..Default::default()
            },
            Usage {
                input_tokens: Some(3),
                output_tokens: Some(7),
                ..Default::default()
            },
        ];

        let sum = Usage::sum_iter_strict(usages.into_iter());
        assert_eq!(sum.input_tokens, Some(18));
        assert_eq!(sum.output_tokens, Some(42));
    }

    #[test]
    fn test_usage_sum_with_none() {
        // If any usage has None for a field, the sum for that field becomes None
        let usages = vec![
            Usage {
                input_tokens: Some(10),
                output_tokens: Some(20),
                ..Default::default()
            },
            Usage {
                input_tokens: None,
                output_tokens: Some(15),
                ..Default::default()
            },
            Usage {
                input_tokens: Some(3),
                output_tokens: Some(7),
                ..Default::default()
            },
        ];

        let sum = Usage::sum_iter_strict(usages.into_iter());
        assert_eq!(sum.input_tokens, None); // None because one usage had None
        assert_eq!(sum.output_tokens, Some(42)); // All had Some, so sum is Some
    }

    #[test]
    fn test_usage_sum_both_fields_none() {
        let usages = vec![
            Usage {
                input_tokens: Some(10),
                output_tokens: None,
                ..Default::default()
            },
            Usage {
                input_tokens: None,
                output_tokens: Some(15),
                ..Default::default()
            },
        ];

        let sum = Usage::sum_iter_strict(usages.into_iter());
        assert_eq!(sum.input_tokens, None);
        assert_eq!(sum.output_tokens, None);
    }

    #[test]
    fn test_usage_sum_empty() {
        let usages: Vec<Usage> = vec![];
        let sum = Usage::sum_iter_strict(usages.into_iter());
        // Empty sum should return Some(0) for both fields since we start with Some(0)
        assert_eq!(sum.input_tokens, Some(0));
        assert_eq!(sum.output_tokens, Some(0));
    }

    #[test]
    fn test_usage_sum_single() {
        let usages = vec![Usage {
            input_tokens: Some(10),
            output_tokens: Some(20),
            input_tokens_details: None,
            output_tokens_details: None,
        }];

        let sum = Usage::sum_iter_strict(usages.into_iter());
        assert_eq!(sum.input_tokens, Some(10));
        assert_eq!(sum.output_tokens, Some(20));
    }

    #[test]
    fn test_usage_sum_with_details() {
        let usages = vec![
            Usage {
                input_tokens: Some(10),
                output_tokens: Some(20),
                input_tokens_details: Some(InputTokensDetails {
                    cached_tokens: Some(5),
                    audio_tokens: Some(2),
                }),
                output_tokens_details: Some(OutputTokensDetails {
                    reasoning_tokens: Some(10),
                    audio_tokens: Some(3),
                    accepted_prediction_tokens: Some(4),
                    rejected_prediction_tokens: Some(1),
                }),
            },
            Usage {
                input_tokens: Some(15),
                output_tokens: Some(30),
                input_tokens_details: Some(InputTokensDetails {
                    cached_tokens: Some(8),
                    audio_tokens: Some(3),
                }),
                output_tokens_details: Some(OutputTokensDetails {
                    reasoning_tokens: Some(15),
                    audio_tokens: Some(5),
                    accepted_prediction_tokens: Some(6),
                    rejected_prediction_tokens: Some(2),
                }),
            },
        ];

        let sum = Usage::sum_iter_strict(usages.into_iter());
        assert_eq!(
            sum.input_tokens,
            Some(25),
            "Input tokens should sum correctly"
        );
        assert_eq!(
            sum.output_tokens,
            Some(50),
            "Output tokens should sum correctly"
        );

        let input_details = sum
            .input_tokens_details
            .expect("Input details should be present");
        assert_eq!(
            input_details.cached_tokens,
            Some(13),
            "Cached tokens should sum correctly"
        );
        assert_eq!(
            input_details.audio_tokens,
            Some(5),
            "Audio tokens should sum correctly"
        );

        let output_details = sum
            .output_tokens_details
            .expect("Output details should be present");
        assert_eq!(
            output_details.reasoning_tokens,
            Some(25),
            "Reasoning tokens should sum correctly"
        );
        assert_eq!(
            output_details.audio_tokens,
            Some(8),
            "Audio tokens should sum correctly"
        );
        assert_eq!(
            output_details.accepted_prediction_tokens,
            Some(10),
            "Accepted prediction tokens should sum correctly"
        );
        assert_eq!(
            output_details.rejected_prediction_tokens,
            Some(3),
            "Rejected prediction tokens should sum correctly"
        );
    }

    #[test]
    fn test_usage_sum_details_none_contamination() {
        let usages = vec![
            Usage {
                input_tokens: Some(10),
                output_tokens: Some(20),
                input_tokens_details: Some(InputTokensDetails {
                    cached_tokens: Some(5),
                    audio_tokens: Some(2),
                }),
                output_tokens_details: Some(OutputTokensDetails {
                    reasoning_tokens: Some(10),
                    audio_tokens: None,
                    accepted_prediction_tokens: Some(4),
                    rejected_prediction_tokens: Some(1),
                }),
            },
            Usage {
                input_tokens: Some(15),
                output_tokens: Some(30),
                input_tokens_details: None, // This should contaminate
                output_tokens_details: Some(OutputTokensDetails {
                    reasoning_tokens: Some(15),
                    audio_tokens: Some(5),
                    accepted_prediction_tokens: Some(6),
                    rejected_prediction_tokens: Some(2),
                }),
            },
        ];

        let sum = Usage::sum_iter_strict(usages.into_iter());
        assert_eq!(
            sum.input_tokens_details, None,
            "Input details should be None due to contamination"
        );

        let output_details = sum
            .output_tokens_details
            .expect("Output details should be present");
        assert_eq!(
            output_details.audio_tokens, None,
            "Audio tokens should be None due to contamination"
        );
        assert_eq!(
            output_details.reasoning_tokens,
            Some(25),
            "Reasoning tokens should sum correctly"
        );
    }
}
