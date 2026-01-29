//! Usage types for Anthropic-compatible API.

use serde::Serialize;

/// Usage information for Anthropic-compatible responses
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Usage information for Anthropic-compatible streaming responses
/// Some fields may be omitted in intermediate chunks
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AnthropicStreamingUsage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
}

impl From<AnthropicUsage> for AnthropicStreamingUsage {
    fn from(usage: AnthropicUsage) -> Self {
        Self {
            input_tokens: Some(usage.input_tokens),
            output_tokens: Some(usage.output_tokens),
        }
    }
}
