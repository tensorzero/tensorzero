//! Serde types for OpenRouter API
//!
//! These types mirror the OpenRouter API request/response structures
//! for reasoning tokens and related features.
//!
//! See: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens

use serde::{Deserialize, Serialize};

// =============================================================================
// Request Types
// =============================================================================

/// OpenRouter reasoning configuration for request.
/// See: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ReasoningConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
}

// =============================================================================
// Response Types
// =============================================================================

/// OpenRouter reasoning detail types as returned in responses.
/// See: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum ReasoningDetail {
    /// Text reasoning with optional signature for multi-turn support.
    /// Note: `text` is optional because OpenRouter may return signature-only
    /// entries for multi-turn conversations.
    #[serde(rename = "reasoning.text")]
    Text {
        #[serde(skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        format: Option<String>,
        /// Stable index for grouping streaming chunks. If present, should be used
        /// instead of array position for ID generation.
        #[serde(skip_serializing_if = "Option::is_none")]
        index: Option<u32>,
    },
    /// Summary of reasoning (may be provided by some models).
    #[serde(rename = "reasoning.summary")]
    Summary {
        summary: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        format: Option<String>,
        /// Stable index for grouping streaming chunks. If present, should be used
        /// instead of array position for ID generation.
        #[serde(skip_serializing_if = "Option::is_none")]
        index: Option<u32>,
    },
    /// Encrypted reasoning content (for models that don't expose reasoning).
    #[serde(rename = "reasoning.encrypted")]
    Encrypted {
        data: String,
        format: String,
        /// Stable index for grouping streaming chunks. If present, should be used
        /// instead of array position for ID generation.
        #[serde(skip_serializing_if = "Option::is_none")]
        index: Option<u32>,
    },
}
