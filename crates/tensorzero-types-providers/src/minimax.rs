//! Serde types for MiniMax API
//!
//! MiniMax uses an OpenAI-compatible API, so most types are reused from the
//! OpenAI module. This module provides MiniMax-specific response format types.

use serde::{Deserialize, Serialize};

use crate::openai::{OpenAIFinishReason, OpenAIResponseToolCall, OpenAIUsage};

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
pub enum MiniMaxResponseFormat {
    #[default]
    Text,
    JsonObject,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MiniMaxFunctionCallChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MiniMaxToolCallChunk {
    pub index: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub function: MiniMaxFunctionCallChunk,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MiniMaxDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<MiniMaxToolCallChunk>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MiniMaxChatChunkChoice {
    pub delta: MiniMaxDelta,
    pub finish_reason: Option<OpenAIFinishReason>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MiniMaxChatChunk {
    pub choices: Vec<MiniMaxChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenAIUsage>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MiniMaxResponseMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIResponseToolCall>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct MiniMaxResponseChoice {
    pub index: u8,
    pub message: MiniMaxResponseMessage,
    pub finish_reason: Option<OpenAIFinishReason>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct MiniMaxResponse {
    pub choices: Vec<MiniMaxResponseChoice>,
    pub usage: OpenAIUsage,
}
