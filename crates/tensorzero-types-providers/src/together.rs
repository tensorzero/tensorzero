//! Serde types for Together API

use serde::{Deserialize, Serialize};

use crate::openai::OpenAIUsage;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TogetherToolType {
    Function,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
pub struct TogetherResponseFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
pub struct TogetherResponseToolCall {
    pub id: String,
    pub r#type: TogetherToolType,
    pub function: TogetherResponseFunctionCall,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TogetherResponseMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<TogetherResponseToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(alias = "reasoning_content")]
    pub reasoning: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TogetherFinishReason {
    Stop,
    Eos,
    Length,
    ToolCalls,
    FunctionCall,
    #[serde(other)]
    Unknown,
}

// Leaving out logprobs and finish_reason for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TogetherResponseChoice {
    pub index: u8,
    pub message: TogetherResponseMessage,
    pub finish_reason: Option<TogetherFinishReason>,
}

// Leaving out id, created, model, service_tier, system_fingerprint, object for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TogetherResponse {
    pub choices: Vec<TogetherResponseChoice>,
    pub usage: OpenAIUsage,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TogetherFunctionCallChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TogetherToolCallChunk {
    pub index: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    // NOTE: these are externally tagged enums, for now we're gonna just keep this hardcoded as there's only one option
    // If we were to do this better, we would need to check the type field
    pub function: TogetherFunctionCallChunk,
}

// This doesn't include role
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TogetherDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(alias = "reasoning_content")]
    pub reasoning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<TogetherToolCallChunk>>,
}

// This doesn't include logprobs, finish_reason, and index
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TogetherChatChunkChoice {
    pub delta: TogetherDelta,
    pub finish_reason: Option<TogetherFinishReason>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TogetherChatChunk {
    pub choices: Vec<TogetherChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenAIUsage>,
}
