//! Serde types for Fireworks API
//!
//! These types handle Fireworks-specific response structures for both
//! non-streaming and streaming responses.

use serde::{Deserialize, Serialize};

use crate::openai::OpenAIUsage;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FireworksToolType {
    Function,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
pub struct FireworksResponseFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
pub struct FireworksResponseToolCall {
    pub id: String,
    pub r#type: FireworksToolType,
    pub function: FireworksResponseFunctionCall,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FireworksResponseMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<FireworksResponseToolCall>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FireworksFinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    #[serde(other)]
    Unknown,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct FireworksResponseChoice {
    pub index: u8,
    pub message: FireworksResponseMessage,
    pub finish_reason: Option<FireworksFinishReason>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct FireworksResponse {
    pub choices: Vec<FireworksResponseChoice>,
    pub usage: OpenAIUsage,
}

// Streaming-specific structs
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FireworksFunctionCallChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FireworksToolCallChunk {
    pub index: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub function: FireworksFunctionCallChunk,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FireworksDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<FireworksToolCallChunk>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FireworksChatChunkChoice {
    pub delta: FireworksDelta,
    #[serde(default)]
    pub finish_reason: Option<FireworksFinishReason>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FireworksChatChunk {
    pub choices: Vec<FireworksChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenAIUsage>,
}
