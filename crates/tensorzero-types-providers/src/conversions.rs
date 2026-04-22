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
