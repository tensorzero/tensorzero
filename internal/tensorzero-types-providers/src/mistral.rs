//! Serde types for Mistral API
//!
//! These types handle Mistral-specific response structures for both
//! non-streaming and streaming responses.

use serde::{Deserialize, Serialize};

/// Sub-chunks inside a `Thinking` content chunk.
/// The `thinking` field in Magistral responses is an array of these.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MistralThinkingSubChunk {
    Text { text: String },
}

/// Represents a typed content chunk in Mistral reasoning responses.
/// Magistral models return content as an array of these chunks instead of a plain string.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MistralContentChunk {
    Text {
        text: String,
    },
    Thinking {
        thinking: Vec<MistralThinkingSubChunk>,
    },
}

/// Mistral's `content` field can be either a plain string (non-reasoning models)
/// or an array of typed chunks (Magistral reasoning models).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum MistralContent {
    String(String),
    Chunks(Vec<MistralContentChunk>),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
pub enum MistralResponseFormat {
    JsonObject,
    #[default]
    Text,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct MistralUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
pub struct MistralResponseFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
pub struct MistralResponseToolCall {
    pub id: String,
    pub function: MistralResponseFunctionCall,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MistralResponseMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<MistralContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<MistralResponseToolCall>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MistralFinishReason {
    Stop,
    Length,
    ModelLength,
    Error,
    ToolCalls,
    #[serde(other)]
    Unknown,
}

// Leaving out logprobs and finish_reason for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct MistralResponseChoice {
    pub index: u8,
    pub message: MistralResponseMessage,
    pub finish_reason: MistralFinishReason,
}

// Leaving out id, created, model, service_tier, system_fingerprint, object for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct MistralResponse {
    pub choices: Vec<MistralResponseChoice>,
    pub usage: MistralUsage,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MistralFunctionCallChunk {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MistralToolCallChunk {
    pub id: String,
    // NOTE: these are externally tagged enums, for now we're gonna just keep this hardcoded as there's only one option
    // If we were to do this better, we would need to check the `type` field
    pub function: MistralFunctionCallChunk,
}

// This doesn't include role
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MistralDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<MistralContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<MistralToolCallChunk>>,
}

// This doesn't include logprobs, finish_reason, and index
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MistralChatChunkChoice {
    pub delta: MistralDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<MistralFinishReason>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MistralChatChunk {
    pub choices: Vec<MistralChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<MistralUsage>,
}
