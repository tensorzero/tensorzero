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
/// for a provider, except for information about what model to actually request,
/// and to convert it back to the appropriate response format.
/// An example of the latter is that we might have prepared a request with Tools available
/// but the client actually just wants a chat response.
#[derive(Builder, Clone, Debug, Default, PartialEq)]
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
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub enum FunctionType {
    Chat,
    Tool,
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
#[derive(Clone, Debug, PartialEq)]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    Tool(String), // Forces the LLM to call a particular tool, the String is the name of the Tool
}

#[derive(Clone, Debug, PartialEq)]
pub enum ToolType {
    Function,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tool {
    pub r#type: ToolType,
    pub description: Option<String>,
    pub name: String,
    pub parameters: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UserInferenceRequestMessage {
    pub content: String, // TODO: for now, we don't support image input. This would be the place to start.
}

#[derive(Clone, Debug, PartialEq)]
pub struct SystemInferenceRequestMessage {
    pub content: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AssistantInferenceRequestMessage {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolInferenceRequestMessage {
    pub tool_call_id: String,
    pub content: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum InferenceRequestMessage {
    User(UserInferenceRequestMessage),
    System(SystemInferenceRequestMessage),
    Assistant(AssistantInferenceRequestMessage),
    Tool(ToolInferenceRequestMessage),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

// TODO: use this and write to DB somehow
#[derive(Clone, Debug, PartialEq)]
pub enum Latency {
    Streaming { ttft: Duration, ttd: Duration },
    NonStreaming { ttd: Duration },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelInferenceResponse {
    pub inference_id: Uuid,
    pub created: u64,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub raw: String,
    pub usage: Usage,
}

impl ModelInferenceResponse {
    pub fn new(
        content: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
        raw: String,
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

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: String,
    pub id: String,
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

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToolCallChunk {
    pub id: Option<String>,
    pub name: Option<String>,
    pub arguments: Option<String>,
}

pub type InferenceResponseStream =
    Pin<Box<dyn Stream<Item = Result<InferenceResponseChunk, Error>> + Send>>;
