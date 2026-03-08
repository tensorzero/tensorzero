//! Serde types for xAI (Grok) API
//!
//! These types handle xAI-specific request/response structures, particularly for
//! reasoning models that support `reasoning_content` in assistant messages.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::borrow::Cow;

// =============================================================================
// Request Message Types
// =============================================================================

/// xAI request message enum.
/// Uses the same structure as OpenAI but with a custom assistant message
/// that supports `reasoning_content` for multi-turn reasoning conversations.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
pub enum XAIRequestMessage<'a> {
    System(XAISystemRequestMessage<'a>),
    User(XAIUserRequestMessage<'a>),
    Assistant(XAIAssistantRequestMessage<'a>),
    Tool(XAIToolRequestMessage<'a>),
}

/// System message for xAI requests
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct XAISystemRequestMessage<'a> {
    pub content: Cow<'a, str>,
}

/// User message for xAI requests.
/// Content is serialized as-is (array of content blocks in OpenAI format).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct XAIUserRequestMessage<'a> {
    /// Content blocks (text, image_url, etc.) - serialized as JSON array
    pub content: Cow<'a, [Value]>,
}

/// Assistant message for xAI requests with support for `reasoning_content`.
/// This is the key difference from OpenAI - xAI accepts reasoning_content
/// in assistant messages for multi-turn reasoning conversations.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct XAIAssistantRequestMessage<'a> {
    /// Content blocks - serialized as JSON array or null
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Cow<'a, [Value]>>,
    /// Tool calls made by the assistant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Cow<'a, [Value]>>,
    /// Reasoning content from a previous response (for multi-turn reasoning).
    /// Only include for Thought blocks with provider_type = "xai" or None.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<Cow<'a, str>>,
}

/// Tool message for xAI requests
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct XAIToolRequestMessage<'a> {
    pub content: Cow<'a, str>,
    pub tool_call_id: Cow<'a, str>,
}

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
