//! Serde types shared across OpenAI-compatible providers.
//!
//! These types are used by OpenAI, DeepSeek, xAI, and other providers that
//! follow the OpenAI chat completions response format.

use serde::{Deserialize, Serialize};
use tensorzero_derive::TensorZeroDeserialize;

// Re-export usage types that were moved to tensorzero-types
pub use tensorzero_types::{OpenAIPromptTokensDetails, OpenAIUsage};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum OpenAIFinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    FunctionCall,
    #[serde(other)]
    Unknown,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
pub struct OpenAIResponseFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
pub struct OpenAIResponseCustomCall {
    pub name: String,
    pub input: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, TensorZeroDeserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum OpenAIResponseToolCall {
    Function {
        id: String,
        function: OpenAIResponseFunctionCall,
    },
    Custom {
        id: String,
        custom: OpenAIResponseCustomCall,
    },
}
