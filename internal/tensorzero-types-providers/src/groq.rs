//! Serde types for Groq API
//!
//! These types handle Groq-specific response structures for both
//! non-streaming and streaming responses.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::serde_util::empty_string_as_none;

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
pub enum GroqResponseFormat {
    #[default]
    Text,
    JsonObject,
    JsonSchema {
        json_schema: Value,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum GroqToolType {
    Function,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GroqUsage {
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
pub struct GroqResponseFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
pub struct GroqResponseToolCall {
    pub id: String,
    pub r#type: GroqToolType,
    pub function: GroqResponseFunctionCall,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GroqResponseMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<GroqResponseToolCall>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum GroqFinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    FunctionCall,
    #[serde(other)]
    Unknown,
}

// Leaving out logprobs and finish_reason for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GroqResponseChoice {
    pub index: u8,
    pub message: GroqResponseMessage,
    pub finish_reason: GroqFinishReason,
}

// Leaving out id, created, model, service_tier, system_fingerprint, object for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GroqResponse {
    pub choices: Vec<GroqResponseChoice>,
    pub usage: GroqUsage,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GroqFunctionCallChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GroqToolCallChunk {
    pub index: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    // NOTE: these are externally tagged enums, for now we're gonna just keep this hardcoded as there's only one option
    // If we were to do this better, we would need to check the `type` field
    pub function: GroqFunctionCallChunk,
}

// This doesn't include role
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GroqDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<GroqToolCallChunk>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GroqChatChunkChoice {
    pub delta: GroqDelta,
    #[serde(default)]
    #[serde(deserialize_with = "empty_string_as_none")]
    pub finish_reason: Option<GroqFinishReason>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GroqChatChunk {
    pub choices: Vec<GroqChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<GroqUsage>,
}
