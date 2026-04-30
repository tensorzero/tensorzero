//! `From` implementations converting provider-specific types into TensorZero core types.

use tensorzero_types::{FinishReason, Usage};

use crate::deepseek::DeepSeekUsage;
use crate::fireworks::FireworksFinishReason;
use crate::openai::OpenAIFinishReason;
use crate::together::TogetherFinishReason;
use crate::xai::XAIUsage;

impl From<OpenAIFinishReason> for FinishReason {
    fn from(reason: OpenAIFinishReason) -> Self {
        match reason {
            OpenAIFinishReason::Stop => FinishReason::Stop,
            OpenAIFinishReason::Length => FinishReason::Length,
            OpenAIFinishReason::ContentFilter => FinishReason::ContentFilter,
            OpenAIFinishReason::ToolCalls => FinishReason::ToolCall,
            OpenAIFinishReason::FunctionCall => FinishReason::ToolCall,
            OpenAIFinishReason::Unknown => FinishReason::Unknown,
        }
    }
}

impl From<FireworksFinishReason> for FinishReason {
    fn from(reason: FireworksFinishReason) -> Self {
        match reason {
            FireworksFinishReason::Stop => FinishReason::Stop,
            FireworksFinishReason::Length => FinishReason::Length,
            FireworksFinishReason::ToolCalls => FinishReason::ToolCall,
            FireworksFinishReason::ContentFilter => FinishReason::ContentFilter,
            FireworksFinishReason::Unknown => FinishReason::Unknown,
        }
    }
}

impl From<TogetherFinishReason> for FinishReason {
    fn from(reason: TogetherFinishReason) -> Self {
        match reason {
            TogetherFinishReason::Stop => FinishReason::Stop,
            TogetherFinishReason::Eos => FinishReason::Stop,
            TogetherFinishReason::Length => FinishReason::Length,
            TogetherFinishReason::ToolCalls => FinishReason::ToolCall,
            TogetherFinishReason::FunctionCall => FinishReason::ToolCall,
            TogetherFinishReason::Unknown => FinishReason::Unknown,
        }
    }
}

impl From<XAIUsage> for Usage {
    fn from(usage: XAIUsage) -> Self {
        let output_tokens = match (usage.completion_tokens, usage.completion_tokens_details) {
            (Some(completion), Some(details)) => {
                Some(completion + details.reasoning_tokens.unwrap_or(0))
            }
            (Some(completion), None) => Some(completion),
            (None, Some(details)) => details.reasoning_tokens,
            (None, None) => None,
        };
        Usage {
            input_tokens: usage.prompt_tokens,
            output_tokens,
            provider_cache_read_input_tokens: usage
                .prompt_tokens_details
                .and_then(|d| d.cached_tokens),
            provider_cache_write_input_tokens: None,
            cost: None,
        }
    }
}

impl From<DeepSeekUsage> for Usage {
    fn from(usage: DeepSeekUsage) -> Self {
        Usage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            provider_cache_read_input_tokens: usage.prompt_cache_hit_tokens,
            provider_cache_write_input_tokens: usage.prompt_cache_miss_tokens,
            cost: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::OpenAIPromptTokensDetails;
    use crate::xai::XAICompletionTokensDetails;

    #[test]
    fn test_usage_from_xai_usage_without_cached_tokens() {
        // No `prompt_tokens_details` -> `cache_read` should be None.
        let xai_usage = XAIUsage {
            prompt_tokens: Some(1000),
            completion_tokens: Some(50),
            completion_tokens_details: None,
            prompt_tokens_details: None,
        };
        let usage: Usage = xai_usage.into();
        assert_eq!(usage.input_tokens, Some(1000));
        assert_eq!(usage.output_tokens, Some(50));
        assert_eq!(
            usage.provider_cache_read_input_tokens, None,
            "cache_read should be None when xAI omits prompt_tokens_details",
        );
        assert_eq!(usage.provider_cache_write_input_tokens, None);
    }

    #[test]
    fn test_usage_from_xai_usage_with_cached_tokens() {
        // xAI reports cache hits in `prompt_tokens_details.cached_tokens`
        // on non-streaming chat completion responses (Grok 4.1 Fast and
        // newer). Verify the From impl propagates them into Usage.
        let xai_usage = XAIUsage {
            prompt_tokens: Some(42_500),
            completion_tokens: Some(710),
            completion_tokens_details: None,
            prompt_tokens_details: Some(OpenAIPromptTokensDetails {
                cached_tokens: Some(22_058),
            }),
        };
        let usage: Usage = xai_usage.into();
        assert_eq!(usage.input_tokens, Some(42_500));
        assert_eq!(usage.output_tokens, Some(710));
        assert_eq!(
            usage.provider_cache_read_input_tokens,
            Some(22_058),
            "cache_read should come from prompt_tokens_details.cached_tokens",
        );
        assert_eq!(
            usage.provider_cache_write_input_tokens, None,
            "xAI doesn't report cache_write",
        );
    }

    #[test]
    fn test_usage_from_xai_usage_with_empty_prompt_tokens_details() {
        // `prompt_tokens_details` present but `cached_tokens` absent -> None.
        let xai_usage = XAIUsage {
            prompt_tokens: Some(100),
            completion_tokens: Some(20),
            completion_tokens_details: None,
            prompt_tokens_details: Some(OpenAIPromptTokensDetails {
                cached_tokens: None,
            }),
        };
        let usage: Usage = xai_usage.into();
        assert_eq!(usage.provider_cache_read_input_tokens, None);
    }

    #[test]
    fn test_usage_from_xai_usage_combines_cached_and_reasoning() {
        // Reasoning tokens must still be added to output_tokens AND
        // cached_tokens must still propagate into cache_read. The two
        // features are orthogonal.
        let xai_usage = XAIUsage {
            prompt_tokens: Some(5_000),
            completion_tokens: Some(300),
            completion_tokens_details: Some(XAICompletionTokensDetails {
                reasoning_tokens: Some(150),
            }),
            prompt_tokens_details: Some(OpenAIPromptTokensDetails {
                cached_tokens: Some(4_500),
            }),
        };
        let usage: Usage = xai_usage.into();
        assert_eq!(usage.input_tokens, Some(5_000));
        assert_eq!(
            usage.output_tokens,
            Some(450),
            "output_tokens should include reasoning_tokens",
        );
        assert_eq!(usage.provider_cache_read_input_tokens, Some(4_500));
    }

    #[test]
    fn test_xai_usage_deserialization_with_cached_tokens() {
        // Real-world xAI non-streaming usage block with caching.
        let json = r#"{
            "prompt_tokens": 5000,
            "completion_tokens": 100,
            "prompt_tokens_details": {
                "cached_tokens": 4500
            }
        }"#;
        let xai_usage: XAIUsage =
            serde_json::from_str(json).expect("should deserialize XAIUsage with cached_tokens");
        let usage: Usage = xai_usage.into();
        assert_eq!(usage.input_tokens, Some(5000));
        assert_eq!(usage.provider_cache_read_input_tokens, Some(4500));
        assert_eq!(usage.provider_cache_write_input_tokens, None);
    }
}
