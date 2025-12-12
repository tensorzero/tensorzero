//! Usage tracking types for OpenAI-compatible API.
//!
//! This module provides types for token usage reporting in OpenAI-compatible responses,
//! including prompt tokens, completion tokens, and total token counts.

use serde::Serialize;

use crate::inference::types::Usage;
use crate::inference::types::usage::{InputTokensDetails, OutputTokensDetails};

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct PromptTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct CompletionTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accepted_prediction_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejected_prediction_tokens: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct OpenAICompatibleUsage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

impl OpenAICompatibleUsage {
    pub fn zero() -> Self {
        Self {
            prompt_tokens: Some(0),
            completion_tokens: Some(0),
            total_tokens: Some(0),
            prompt_tokens_details: None,
            completion_tokens_details: None,
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

        // Sum detail fields using strict contamination
        self.prompt_tokens_details =
            Self::sum_prompt_details_strict(self.prompt_tokens_details, other.input_tokens_details);
        self.completion_tokens_details = Self::sum_completion_details_strict(
            self.completion_tokens_details,
            other.output_tokens_details,
        );
    }

    fn sum_prompt_details_strict(
        a: Option<PromptTokensDetails>,
        b: Option<InputTokensDetails>,
    ) -> Option<PromptTokensDetails> {
        match (a, b) {
            (Some(a_details), Some(b_details)) => Some(PromptTokensDetails {
                cached_tokens: match (a_details.cached_tokens, b_details.cached_tokens) {
                    (Some(a_val), Some(b_val)) => Some(a_val + b_val),
                    _ => None,
                },
                audio_tokens: match (a_details.audio_tokens, b_details.audio_tokens) {
                    (Some(a_val), Some(b_val)) => Some(a_val + b_val),
                    _ => None,
                },
            }),
            _ => None,
        }
    }

    fn sum_completion_details_strict(
        a: Option<CompletionTokensDetails>,
        b: Option<OutputTokensDetails>,
    ) -> Option<CompletionTokensDetails> {
        match (a, b) {
            (Some(a_details), Some(b_details)) => Some(CompletionTokensDetails {
                reasoning_tokens: match (a_details.reasoning_tokens, b_details.reasoning_tokens) {
                    (Some(a_val), Some(b_val)) => Some(a_val + b_val),
                    _ => None,
                },
                audio_tokens: match (a_details.audio_tokens, b_details.audio_tokens) {
                    (Some(a_val), Some(b_val)) => Some(a_val + b_val),
                    _ => None,
                },
                accepted_prediction_tokens: match (
                    a_details.accepted_prediction_tokens,
                    b_details.accepted_prediction_tokens,
                ) {
                    (Some(a_val), Some(b_val)) => Some(a_val + b_val),
                    _ => None,
                },
                rejected_prediction_tokens: match (
                    a_details.rejected_prediction_tokens,
                    b_details.rejected_prediction_tokens,
                ) {
                    (Some(a_val), Some(b_val)) => Some(a_val + b_val),
                    _ => None,
                },
            }),
            _ => None,
        }
    }
}

impl From<Usage> for OpenAICompatibleUsage {
    fn from(usage: Usage) -> Self {
        let prompt_tokens_details = usage
            .input_tokens_details
            .map(|details| PromptTokensDetails {
                cached_tokens: details.cached_tokens,
                audio_tokens: details.audio_tokens,
            });

        let completion_tokens_details =
            usage
                .output_tokens_details
                .map(|details| CompletionTokensDetails {
                    reasoning_tokens: details.reasoning_tokens,
                    audio_tokens: details.audio_tokens,
                    accepted_prediction_tokens: details.accepted_prediction_tokens,
                    rejected_prediction_tokens: details.rejected_prediction_tokens,
                });

        OpenAICompatibleUsage {
            prompt_tokens: usage.input_tokens,
            completion_tokens: usage.output_tokens,
            total_tokens: usage.total_tokens(),
            prompt_tokens_details,
            completion_tokens_details,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::types::usage::{InputTokensDetails, OutputTokensDetails, Usage};

    #[test]
    fn test_from_usage_with_details() {
        let usage = Usage {
            input_tokens: Some(10),
            output_tokens: Some(20),
            input_tokens_details: Some(InputTokensDetails {
                cached_tokens: Some(5),
                audio_tokens: Some(2),
            }),
            output_tokens_details: Some(OutputTokensDetails {
                reasoning_tokens: Some(8),
                audio_tokens: Some(3),
                accepted_prediction_tokens: Some(4),
                rejected_prediction_tokens: Some(1),
            }),
        };

        let openai_usage: OpenAICompatibleUsage = usage.into();
        
        assert_eq!(openai_usage.prompt_tokens, Some(10), "Prompt tokens should match");
        assert_eq!(openai_usage.completion_tokens, Some(20), "Completion tokens should match");
        assert_eq!(openai_usage.total_tokens, Some(30), "Total tokens should be sum");

        let prompt_details = openai_usage.prompt_tokens_details.expect("Prompt details should exist");
        assert_eq!(prompt_details.cached_tokens, Some(5), "Cached tokens should match");
        assert_eq!(prompt_details.audio_tokens, Some(2), "Audio tokens should match");

        let completion_details = openai_usage.completion_tokens_details.expect("Completion details should exist");
        assert_eq!(completion_details.reasoning_tokens, Some(8), "Reasoning tokens should match");
        assert_eq!(completion_details.audio_tokens, Some(3), "Audio tokens should match");
        assert_eq!(completion_details.accepted_prediction_tokens, Some(4), "Accepted prediction tokens should match");
        assert_eq!(completion_details.rejected_prediction_tokens, Some(1), "Rejected prediction tokens should match");
    }

    #[test]
    fn test_from_usage_without_details() {
        let usage = Usage {
            input_tokens: Some(15),
            output_tokens: Some(25),
            input_tokens_details: None,
            output_tokens_details: None,
        };

        let openai_usage: OpenAICompatibleUsage = usage.into();
        
        assert_eq!(openai_usage.prompt_tokens, Some(15), "Prompt tokens should match");
        assert_eq!(openai_usage.completion_tokens, Some(25), "Completion tokens should match");
        assert_eq!(openai_usage.total_tokens, Some(40), "Total tokens should be sum");
        assert!(openai_usage.prompt_tokens_details.is_none(), "Prompt details should be None");
        assert!(openai_usage.completion_tokens_details.is_none(), "Completion details should be None");
    }

    #[test]
    fn test_json_serialization_omits_none_fields() {
        let usage = OpenAICompatibleUsage {
            prompt_tokens: Some(10),
            completion_tokens: Some(20),
            total_tokens: Some(30),
            prompt_tokens_details: None,
            completion_tokens_details: None,
        };

        let json = serde_json::to_value(&usage).expect("Should serialize");
        let obj = json.as_object().expect("Should be an object");
        
        assert!(obj.contains_key("prompt_tokens"), "Should have prompt_tokens");
        assert!(obj.contains_key("completion_tokens"), "Should have completion_tokens");
        assert!(obj.contains_key("total_tokens"), "Should have total_tokens");
        assert!(!obj.contains_key("prompt_tokens_details"), "Should not have None prompt_tokens_details");
        assert!(!obj.contains_key("completion_tokens_details"), "Should not have None completion_tokens_details");
    }

    #[test]
    fn test_sum_usage_strict_with_details() {
        let mut usage1 = OpenAICompatibleUsage {
            prompt_tokens: Some(10),
            completion_tokens: Some(20),
            total_tokens: Some(30),
            prompt_tokens_details: Some(PromptTokensDetails {
                cached_tokens: Some(5),
                audio_tokens: Some(2),
            }),
            completion_tokens_details: Some(CompletionTokensDetails {
                reasoning_tokens: Some(8),
                audio_tokens: Some(3),
                accepted_prediction_tokens: Some(4),
                rejected_prediction_tokens: Some(1),
            }),
        };

        let usage2 = Usage {
            input_tokens: Some(15),
            output_tokens: Some(25),
            input_tokens_details: Some(InputTokensDetails {
                cached_tokens: Some(7),
                audio_tokens: Some(3),
            }),
            output_tokens_details: Some(OutputTokensDetails {
                reasoning_tokens: Some(10),
                audio_tokens: Some(5),
                accepted_prediction_tokens: Some(6),
                rejected_prediction_tokens: Some(2),
            }),
        };

        usage1.sum_usage_strict(&usage2);
        
        assert_eq!(usage1.prompt_tokens, Some(25), "Prompt tokens should sum");
        assert_eq!(usage1.completion_tokens, Some(45), "Completion tokens should sum");
        assert_eq!(usage1.total_tokens, Some(70), "Total tokens should sum");

        let prompt_details = usage1.prompt_tokens_details.expect("Should have prompt details");
        assert_eq!(prompt_details.cached_tokens, Some(12), "Cached tokens should sum");
        assert_eq!(prompt_details.audio_tokens, Some(5), "Audio tokens should sum");

        let completion_details = usage1.completion_tokens_details.expect("Should have completion details");
        assert_eq!(completion_details.reasoning_tokens, Some(18), "Reasoning tokens should sum");
        assert_eq!(completion_details.audio_tokens, Some(8), "Audio tokens should sum");
        assert_eq!(completion_details.accepted_prediction_tokens, Some(10), "Accepted prediction tokens should sum");
        assert_eq!(completion_details.rejected_prediction_tokens, Some(3), "Rejected prediction tokens should sum");
    }

    #[test]
    fn test_sum_usage_strict_none_contamination() {
        let mut usage1 = OpenAICompatibleUsage {
            prompt_tokens: Some(10),
            completion_tokens: Some(20),
            total_tokens: Some(30),
            prompt_tokens_details: Some(PromptTokensDetails {
                cached_tokens: Some(5),
                audio_tokens: None, // This will contaminate
            }),
            completion_tokens_details: Some(CompletionTokensDetails {
                reasoning_tokens: Some(8),
                audio_tokens: Some(3),
                accepted_prediction_tokens: Some(4),
                rejected_prediction_tokens: Some(1),
            }),
        };

        let usage2 = Usage {
            input_tokens: Some(15),
            output_tokens: Some(25),
            input_tokens_details: None, // This will contaminate prompt details
            output_tokens_details: Some(OutputTokensDetails {
                reasoning_tokens: Some(10),
                audio_tokens: Some(5),
                accepted_prediction_tokens: Some(6),
                rejected_prediction_tokens: Some(2),
            }),
        };

        usage1.sum_usage_strict(&usage2);
        
        assert_eq!(usage1.prompt_tokens, Some(25), "Prompt tokens should still sum");
        assert_eq!(usage1.completion_tokens, Some(45), "Completion tokens should still sum");
        assert!(usage1.prompt_tokens_details.is_none(), "Prompt details should be None due to contamination");
        
        let completion_details = usage1.completion_tokens_details.expect("Should have completion details");
        assert_eq!(completion_details.reasoning_tokens, Some(18), "Reasoning tokens should sum");
        assert_eq!(completion_details.audio_tokens, Some(8), "Audio tokens should sum");
    }
}
