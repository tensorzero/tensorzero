//! Serde types for xAI (Grok) API
//!
//! These types handle xAI-specific response structures, particularly for
//! reasoning models that report tokens differently than standard OpenAI format.

use serde::{Deserialize, Serialize};

// =============================================================================
// Usage Types
// =============================================================================

/// xAI-specific usage struct that includes `reasoning_tokens` in `output_tokens`.
/// xAI reports reasoning tokens separately in `completion_tokens_details.reasoning_tokens`,
/// so we need to add them to get the true output token count.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct XAIUsage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub completion_tokens_details: Option<XAICompletionTokensDetails>,
}

/// Details about completion tokens, including reasoning tokens
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct XAICompletionTokensDetails {
    pub reasoning_tokens: Option<u32>,
}

// =============================================================================
// Response Types
// =============================================================================

/// xAI-specific response struct that uses XAIUsage instead of OpenAIUsage.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct XAIResponse<C> {
    pub choices: Vec<C>,
    pub usage: XAIUsage,
}
