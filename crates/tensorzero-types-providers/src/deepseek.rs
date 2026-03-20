//! Serde types for DeepSeek API
//!
//! These types handle DeepSeek-specific request/response structures, including
//! support for `reasoning_content` in responses.

use serde::{Deserialize, Serialize};

use crate::openai::{OpenAIFinishReason, OpenAIResponseToolCall, OpenAIUsage};

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
pub enum DeepSeekResponseFormat {
    #[default]
    Text,
    JsonObject,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DeepSeekFunctionCallChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DeepSeekToolCallChunk {
    pub index: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    // NOTE: these are externally tagged enums, for now we're gonna just keep this hardcoded as there's only one option
    // If we were to do this better, we would need to check the `type` field
    pub function: DeepSeekFunctionCallChunk,
}

// This doesn't include role
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DeepSeekDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<DeepSeekToolCallChunk>>,
}

// This doesn't include logprobs, finish_reason, and index
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DeepSeekChatChunkChoice {
    pub delta: DeepSeekDelta,
    pub finish_reason: Option<OpenAIFinishReason>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DeepSeekChatChunk {
    pub choices: Vec<DeepSeekChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenAIUsage>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DeepSeekResponseMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIResponseToolCall>>,
}

// Leaving out logprobs and finish_reason for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DeepSeekResponseChoice {
    pub index: u8,
    pub message: DeepSeekResponseMessage,
    pub finish_reason: Option<OpenAIFinishReason>,
}

// Leaving out id, created, model, service_tier, system_fingerprint, object for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DeepSeekResponse {
    pub choices: Vec<DeepSeekResponseChoice>,
    pub usage: OpenAIUsage,
}
