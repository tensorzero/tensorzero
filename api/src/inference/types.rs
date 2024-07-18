use derive_builder::Builder;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    pin::Pin,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use uuid::Uuid;

use crate::error::Error;

/// Top-level TensorZero type for an inference request to a particular model.
/// This should contain all the information required to make a valid inference request
/// for a provider, except for information about what model to actually request.
/// and to convert it back to the appropriate response format.
/// An example of the latter is that we might have prepared a request with Tools available
/// but the client actually just wants.
#[derive(Debug, PartialEq, Builder, Default, Clone)]
#[builder(setter(into, strip_option), default)]
pub struct ModelInferenceRequest {
    pub messages: Vec<InferenceRequestMessage>,
    pub tools_available: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    pub parallel_tool_calls: Option<bool>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub stream: bool,
    pub json_mode: bool,
    pub function_type: FunctionType,
    pub output_schema: Option<Value>,
}

/// FunctionType denotes whether the request is a chat or tool call.
/// By keeping this enum separately from whether tools are available, we
/// allow the request to use tools to enforce an output schema without necessarily
/// exposing that to the client (unless they requested a tool call themselves).
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
#[repr(u8)]
pub enum FunctionType {
    Chat = 1,
    Tool = 2,
}

/// The default FunctionType is Chat
impl Default for FunctionType {
    fn default() -> Self {
        FunctionType::Chat
    }
}

/// Most inference providers allow the user to force a tool to be used
/// and even specify which tool to be used.
///
/// This enum is used to denote this tool choice.
#[derive(Debug, PartialEq, Clone)]
pub enum ToolChoice {
    #[allow(dead_code)] // TODO: remove
    None,
    #[allow(dead_code)] // TODO: remove
    Auto,
    #[allow(dead_code)] // TODO: remove
    Required,
    #[allow(dead_code)] // TODO: remove
    Tool(String),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Role {
    #[allow(dead_code)] // TODO: remove
    User,
    #[allow(dead_code)] // TODO: remove
    Assistant,
    #[allow(dead_code)] // TODO: remove
    System,
    #[allow(dead_code)] // TODO: remove
    Tool,
}

#[derive(Debug, PartialEq, Clone)]
pub enum ToolType {
    #[allow(dead_code)] // TODO: remove
    Function,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Tool {
    pub r#type: ToolType,
    pub description: Option<String>,
    pub name: String,
    pub parameters: Value,
}

// We do not support passing tool results in as a message content at the moment
// If we did, we would need to add a field for tool name here.
// Since TensorZero has a typed interface for function calling we'll need to put some thought into
// how that might be supported or whether it would be required.
#[derive(Debug, PartialEq, Clone)]
pub struct InferenceRequestMessage {
    pub role: Role,
    pub content: String,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

// TODO: use this and write to DB somehow
#[derive(Debug, PartialEq, Clone)]
pub enum Latency {
    #[allow(dead_code)] // TODO: remove
    Streaming { ttft: Duration, ttd: Duration },
    #[allow(dead_code)] // TODO: remove
    NonStreaming { ttd: Duration },
}

#[derive(Debug, PartialEq)]
pub struct ModelInferenceResponse {
    pub inference_id: Uuid,
    pub created: u64,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub raw: Value,
    pub usage: Usage,
}

impl ModelInferenceResponse {
    pub fn new(
        content: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
        raw: Value,
        usage: Usage,
    ) -> Self {
        Self {
            inference_id: Uuid::now_v7(),
            created: current_timestamp(),
            content,
            tool_calls,
            raw,
            usage,
        }
    }
}

// Function to get the current timestamp in seconds
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs()
}

#[derive(Serialize, Debug, PartialEq, Clone)]
pub struct ToolCall {
    pub name: String,
    pub arguments: String,
    pub tool_call_id: String,
}

#[derive(Debug)]
pub struct InferenceResponseChunk {
    pub inference_id: Uuid,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCallChunk>>,
    pub created: u64,
    pub usage: Option<Usage>,
}

impl InferenceResponseChunk {
    #[allow(dead_code)] // TODO: remove
    pub fn new(
        inference_id: Uuid,
        content: Option<String>,
        tool_calls: Option<Vec<ToolCallChunk>>,
        usage: Option<Usage>,
    ) -> Self {
        Self {
            inference_id,
            content,
            tool_calls,
            created: current_timestamp(),
            usage,
        }
    }
}

#[derive(Serialize, Debug, PartialEq, Clone)]
pub struct ToolCallChunk {
    pub id: Option<String>,
    pub name: Option<String>,
    pub arguments: Option<String>,
}

#[allow(dead_code)] // TODO: remove
pub type InferenceResponseStream =
    Pin<Box<dyn Stream<Item = Result<InferenceResponseChunk, Error>> + Send>>;
