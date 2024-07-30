use derive_builder::Builder;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    fmt,
    pin::Pin,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use uuid::Uuid;

use crate::{error::Error, function::FunctionConfig};

/// TODO(Viraj): write a substantial docstring describing how data flows through the system as a series of
/// transformations between types.

/// InputMessage and InputMessageRole are our representation of the input sent by the client
/// prior to any processing into LLM representations below.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InputMessageRole {
    System,
    User,
    Assistant,
    // TODO: add Tool
}

impl fmt::Display for InputMessageRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", serde_json::to_string(self).unwrap_or_default())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InputMessage {
    pub role: InputMessageRole,
    pub content: serde_json::Value,
}

/// Top-level TensorZero type for an inference request to a particular model.
/// This should contain all the information required to make a valid inference request
/// for a provider, except for information about what model to actually request,
/// and to convert it back to the appropriate response format.
/// An example of the latter is that we might have prepared a request with Tools available
/// but the client actually just wants a chat response.
#[derive(Builder, Clone, Debug, Default, PartialEq)]
#[builder(setter(into, strip_option), default)]
pub struct ModelInferenceRequest<'a> {
    pub messages: Vec<InferenceRequestMessage>,
    pub tools_available: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    pub parallel_tool_calls: Option<bool>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub stream: bool,
    pub json_mode: bool,
    pub function_type: FunctionType,
    pub output_schema: Option<&'a Value>,
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

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
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

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelInferenceResponse {
    pub id: Uuid,
    pub created: u64,
    pub content: Option<String>,
    #[serde(skip)]
    pub tool_calls: Option<Vec<ToolCall>>,
    pub raw: String,
    pub usage: Usage,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelInference {
    pub id: Uuid,
    pub inference_id: Uuid,
    pub input: String,
    pub output: String,
    pub raw_response: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl ModelInference {
    pub fn new(response: ModelInferenceResponse, input: String, inference_id: Uuid) -> Self {
        // TODO: deal with tools
        Self {
            id: Uuid::now_v7(),
            inference_id,
            input,
            output: response.content.unwrap_or_default(),
            raw_response: response.raw,
            input_tokens: response.usage.prompt_tokens,
            output_tokens: response.usage.completion_tokens,
        }
    }
}

impl ModelInferenceResponse {
    pub fn new(
        content: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
        raw: String,
        usage: Usage,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            created: current_timestamp(),
            content,
            tool_calls,
            raw,
            usage,
        }
    }
}

#[derive(Serialize, Debug)]
pub struct ChatInferenceResponse {
    pub inference_id: Uuid,
    pub created: u64,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub usage: Usage,
    pub model_inference_responses: Vec<ModelInferenceResponse>,
}

#[derive(Serialize, Debug)]
#[serde(tag = "type")]
pub enum InferenceResponse {
    Chat(ChatInferenceResponse),
}

impl InferenceResponse {
    pub fn parse_output(&self, function: &FunctionConfig) -> Result<Value, Error> {
        // TODO(Viraj): factor this out into a trait of the ChatInferenceResponse, write a test.
        let content = match self {
            InferenceResponse::Chat(chat_response) => chat_response.content.as_ref(),
        };
        let content = content.ok_or(Error::OutputParsing {
            raw_output: "".to_string(),
            message: "".to_string(),
        })?;
        let output_value = serde_json::from_str(content).map_err(|e| Error::OutputParsing {
            raw_output: content.clone(),
            message: e.to_string(),
        })?;
        match function.output_schema() {
            Some(schema) => {
                schema
                    .validate(&output_value)
                    .map_err(|e| Error::OutputValidation {
                        source: Box::new(e),
                    })?;
                Ok(output_value)
            }
            // TODO(Viraj): check how a raw string is handled here and make sure it's sensible
            None => Ok(output_value),
        }
    }
}

// TODO: handle references and stuff
#[derive(Serialize, Debug)]
pub struct Inference {
    pub id: Uuid,
    pub function_name: String,
    pub variant_name: String,
    pub episode_id: Uuid,
    pub input: String,
    pub output: Option<Value>,
    pub raw_output: String,
}

impl Inference {
    pub fn new(
        inference_response: InferenceResponse,
        parsed_output: Option<Value>,
        input: String,
        episode_id: Uuid,
        function_name: String,
        variant_name: String,
    ) -> Self {
        match inference_response {
            InferenceResponse::Chat(chat_response) => Self {
                id: chat_response.inference_id,
                function_name,
                variant_name,
                episode_id,
                input,
                raw_output: chat_response.content.unwrap_or_default(),
                output: parsed_output,
            },
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

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChatInferenceResponseChunk {
    pub inference_id: Uuid,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCallChunk>>,
    pub created: u64,
    pub usage: Option<Usage>,
}

impl From<ModelInferenceResponseChunk> for ChatInferenceResponseChunk {
    fn from(chunk: ModelInferenceResponseChunk) -> Self {
        Self {
            inference_id: chunk.inference_id,
            content: chunk.content,
            tool_calls: chunk.tool_calls,
            created: chunk.created,
            usage: chunk.usage,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum InferenceResponseChunk {
    Chat(ChatInferenceResponseChunk),
}

#[derive(Debug, Clone)]
pub struct ModelInferenceResponseChunk {
    pub inference_id: Uuid,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCallChunk>>,
    pub created: u64,
    pub usage: Option<Usage>,
    pub raw: String,
}

impl ModelInferenceResponseChunk {
    pub fn new(
        inference_id: Uuid,
        content: Option<String>,
        tool_calls: Option<Vec<ToolCallChunk>>,
        usage: Option<Usage>,
        raw: String,
    ) -> Self {
        Self {
            inference_id,
            content,
            tool_calls,
            created: current_timestamp(),
            usage,
            raw,
        }
    }
}

impl TryFrom<Vec<ModelInferenceResponseChunk>> for InferenceResponse {
    type Error = Error;

    fn try_from(value: Vec<ModelInferenceResponseChunk>) -> Result<Self, Self::Error> {
        // TODO(Viraj): we need this to be per-inference-response-type
        // and sensitive to the type of variant and function being called.

        // TODO(Viraj): test extensively
        let inference_id = value
            .first()
            .ok_or(Error::TypeConversion {
                message:
                    "Attempted to create an InferenceResponse from an empty response chunk vector"
                        .to_string(),
            })?
            .inference_id;
        let created = value
            .first()
            .ok_or(Error::TypeConversion {
                message:
                    "Attempted to create an InferenceResponse from an empty response chunk vector"
                        .to_string(),
            })?
            .created;
        let mut content: Option<String> = None;
        let mut tool_calls: Option<Vec<ToolCall>> = None;
        let raw: String = value
            .iter()
            .map(|chunk| chunk.raw.as_str())
            .collect::<Vec<&str>>()
            .join("\n");
        let mut usage: Usage = Usage::default();
        for chunk in value {
            content = match content {
                Some(c) => Some(c + &chunk.content.unwrap_or_default()),
                None => chunk.content,
            };
            tool_calls = match tool_calls {
                Some(mut t) => {
                    for (j, tool_call) in chunk.tool_calls.unwrap_or_default().iter().enumerate() {
                        if let Some(existing_tool_call) = t.get_mut(j) {
                            existing_tool_call
                                .arguments
                                .push_str(tool_call.arguments.as_deref().unwrap_or_default());
                        }
                    }
                    Some(t)
                }
                None => chunk.tool_calls.map(|tool_calls| {
                    tool_calls
                        .into_iter()
                        .map(|tool_call| tool_call.into())
                        .collect()
                }),
            };
            if let Some(chunk_usage) = chunk.usage {
                usage.prompt_tokens = usage
                    .prompt_tokens
                    .saturating_add(chunk_usage.prompt_tokens);
                usage.completion_tokens = usage
                    .completion_tokens
                    .saturating_add(chunk_usage.completion_tokens);
            }
        }
        let model_response =
            ModelInferenceResponse::new(content.clone(), tool_calls.clone(), raw, usage.clone());
        Ok(InferenceResponse::Chat(ChatInferenceResponse {
            inference_id,
            created,
            content,
            tool_calls,
            usage,
            model_inference_responses: vec![model_response],
        }))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToolCallChunk {
    pub id: Option<String>,
    pub name: Option<String>,
    pub arguments: Option<String>,
}

impl From<ToolCallChunk> for ToolCall {
    fn from(tool_call: ToolCallChunk) -> Self {
        // TODO: explicitly handle tools both for streaming and non-streaming
        // as well as for Chat and Tool-style Functions
        Self {
            id: tool_call.id.unwrap_or_default(),
            name: tool_call.name.unwrap_or_default(),
            arguments: tool_call.arguments.unwrap_or_default(),
        }
    }
}

pub type InferenceResponseStream =
    Pin<Box<dyn Stream<Item = Result<ModelInferenceResponseChunk, Error>> + Send>>;
