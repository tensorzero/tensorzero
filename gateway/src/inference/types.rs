use derive_builder::Builder;
use futures::Stream;
use serde::{ser::SerializeStruct, Deserialize, Serialize, Serializer};
use serde_json::Value;
use std::{
    collections::HashMap,
    fmt,
    path::PathBuf,
    pin::Pin,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use uuid::Uuid;

use crate::tool::{ToolCall, ToolCallChunk, ToolCallConfig, ToolChoice, ToolResult};
use crate::{error::Error, jsonschema_util::JSONSchemaFromPath};
/// Data flow in TensorZero
///
/// The flow of an inference request through TensorZero can be viewed as a series of transformations between types.
/// Most of them are defined below.

/// A request is made that contains an Input
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Input {
    pub system: Option<Value>,
    pub messages: Vec<InputMessage>,
}

/// InputMessage and Role are our representation of the input sent by the client
/// prior to any processing into LLM representations below.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InputMessage {
    pub role: Role,
    pub content: Vec<InputMessageContent>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum InputMessageContent {
    Text(Value),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    // We may extend this in the future to include other types of content
}

#[cfg(test)]
impl From<String> for InputMessageContent {
    fn from(text: String) -> Self {
        InputMessageContent::Text(Value::String(text))
    }
}

impl From<Value> for InputMessageContent {
    fn from(value: Value) -> Self {
        InputMessageContent::Text(value)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
}

/// InputMessages are validated against the input schema of the Function
/// and then templated and transformed into RequestMessages for a particular Variant.
/// They might contain tool calls or tool results along with text.
/// The abstraction we use to represent this is ContentBlock, which is a union of Text, ToolCall, and ToolResult.
/// ContentBlocks are collected into RequestMessages.
/// These RequestMessages are collected into a ModelInferenceRequest,
/// which should contain all information needed by a ModelProvider to perform the
/// inference that is called for.

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Text {
    pub text: String,
}

/// Core representation of the types of content that could go in or out of a model
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ContentBlock {
    Text(Text),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
}

/// A RequestMessage is a message sent to a model
#[derive(Clone, Debug, PartialEq)]
pub struct RequestMessage {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub enum FunctionType {
    #[default]
    Chat,
    Json,
}

#[derive(Clone, Default, Debug, PartialEq)]
pub enum JSONMode {
    #[default]
    Off,
    On,
    Strict,
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
    pub messages: Vec<RequestMessage>,
    pub system: Option<String>,
    pub tool_config: Option<&'a ToolCallConfig<'a>>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub stream: bool,
    pub json_mode: JSONMode,
    pub function_type: FunctionType,
    pub output_schema: Option<&'a Value>,
}

/// Each provider transforms a ModelInferenceRequest into a provider-specific (private) inference request type
/// that is suitable for serialization directly into a request to the provider.
///
/// In both non-streaming and streaming inference, each ModelProvider recieves data from the provider in a
/// a (private) provider-specific format that is then transformed into a ModelInferenceResponse (non-streaming)
/// or a stream of ModelInferenceResponseChunks (streaming).

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelInferenceResponse {
    pub id: Uuid,
    pub created: u64,
    pub content: Vec<ContentBlock>,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Latency {
    Streaming {
        ttft: Duration,
        response_time: Duration,
    },
    NonStreaming {
        response_time: Duration,
    },
}

/// As a Variant might make use of multiple model inferences, we then combine
/// one or more ModelInferenceResponses into a single InferenceResponse (but we keep the original ModelInferenceResponses around).
/// In the non-streaming case, this InferenceResponse is serialized into the TensorZero response format.
/// See below for streaming case.

#[derive(Debug, Clone)]
pub enum InferenceResponse {
    Chat(ChatInferenceResponse),
}

// Determines the return type of Inference API for Chat-type functions (which is all of them right now)
/// See InferenceResponseWithOutputSchema below for details
#[derive(Debug, Clone)]
pub struct ChatInferenceResponse {
    inference_id: Uuid,
    created: u64,
    pub parsed_output: Option<Value>,
    pub content_blocks: Vec<ContentBlock>,
    pub usage: Usage,
    pub model_inference_responses: Vec<ModelInferenceResponse>,
}

/// In the streaming case we convert ModelInferenceResponseChunks into serialized InferenceResponseChunks to the client.
/// We then collect all the InferenceResponseChunks into an InferenceResponse for validation and storage after the fact.

#[derive(Debug, Clone)]
pub struct ModelInferenceResponseChunk {
    pub inference_id: Uuid,
    pub content: Vec<ContentBlockChunk>,
    pub created: u64,
    pub usage: Option<Usage>,
    pub raw_response: String,
    pub latency: Duration,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ContentBlockChunk {
    Text(TextChunk),
    ToolCall(ToolCallChunk),
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TextChunk {
    pub id: String,
    pub text: String,
}

// This determines what gets serialized in streaming Chat functions
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChatInferenceResponseChunk {
    pub inference_id: Uuid,
    pub content: Vec<ContentBlockChunk>,
    pub created: u64,
    pub usage: Option<Usage>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum InferenceResponseChunk {
    Chat(ChatInferenceResponseChunk),
}

/// Alongside the response, we also store information about what happened during the request.
/// For this we convert the InferenceResponse into an Inference and ModelInferences,
/// which are written to ClickHouse tables of the same name asynchronously.

#[derive(Serialize, Debug)]
pub struct Inference {
    pub id: Uuid,
    pub function_name: String,
    pub variant_name: String,
    pub episode_id: Uuid,
    pub input: String,
    pub output: String,
    pub dynamic_tool_config: String,
    pub processing_time_ms: u32,
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
    pub response_time_ms: u32,
    pub ttft_ms: Option<u32>,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
        }
    }
}

impl From<String> for ContentBlock {
    fn from(text: String) -> Self {
        ContentBlock::Text(Text { text })
    }
}

impl ModelInference {
    pub fn new(response: ModelInferenceResponse, input: String, inference_id: Uuid) -> Self {
        // TODO (#30): deal with tools
        let (latency_ms, ttft_ms) = match response.latency {
            Latency::Streaming {
                ttft,
                response_time,
            } => (
                response_time.as_millis() as u32,
                Some(ttft.as_millis() as u32),
            ),
            Latency::NonStreaming { response_time } => (response_time.as_millis() as u32, None),
        };
        Self {
            id: Uuid::now_v7(),
            inference_id,
            input,
            // We write the serialized JSON form of the ContentBlocks output by the model
            output: serde_json::to_string(&response.content).unwrap_or_default(),
            raw_response: response.raw_response,
            input_tokens: response.usage.prompt_tokens,
            output_tokens: response.usage.completion_tokens,
            response_time_ms: latency_ms,
            ttft_ms,
        }
    }
}

impl ModelInferenceResponse {
    pub fn new(content: Vec<ContentBlock>, raw: String, usage: Usage, latency: Latency) -> Self {
        Self {
            id: Uuid::now_v7(),
            created: current_timestamp(),
            content,
            raw_response: raw,
            usage,
            latency,
        }
    }
}

impl ChatInferenceResponse {
    pub fn new(
        inference_id: Uuid,
        raw_content: Vec<ContentBlock>,
        usage: Usage,
        model_inference_responses: Vec<ModelInferenceResponse>,
        output_schema: Option<&JSONSchemaFromPath>,
    ) -> Self {
        #[allow(clippy::expect_used)]
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_secs();
        let parsed_output = match output_schema {
            // We write None for parsed_output if parsing fails
            Some(schema) => Self::parse_output(&raw_content, schema).ok(),
            // We also write None if there is no output schema
            None => None,
        };
        Self {
            inference_id,
            created,
            parsed_output,
            content_blocks: raw_content,
            usage,
            model_inference_responses,
        }
    }

    fn parse_output(
        content: &[ContentBlock],
        output_schema: &JSONSchemaFromPath,
    ) -> Result<Value, Error> {
        if content.is_empty() {
            return Err(Error::OutputParsing {
                raw_output: "".to_string(),
                message: "Output parsing failed due to empty content".to_string(),
            });
        }
        // let tool_choice = match tool_config {
        //     Some(config) => config.tool_choice,
        //     None => &ToolChoice::None,
        // };
        // match tool_choice {
        //     ToolChoice::Implicit => {
        //         // TODO(#30): handle tool calls and fully think this through
        //         for content in content.iter().rev() {
        //             let arguments = match content {
        //                 ContentBlock::ToolCall(tool_call) => &tool_call.arguments,
        //                 _ => continue,
        //             };
        //             let parsed_arguments =
        //                 serde_json::from_str(arguments).map_err(|e| Error::OutputParsing {
        //                     raw_output: arguments.to_string(),
        //                     message: e.to_string(),
        //                 })?;
        //             output_schema.validate(&parsed_arguments).map_err(|e| {
        //                 Error::OutputValidation {
        //                     source: Box::new(e),
        //                 }
        //             })?;
        //             return Ok(parsed_arguments);
        //         }
        //         Err(Error::OutputParsing {
        //             raw_output: "".to_string(),
        //             message: "Output parsing failed due to no tool calls".to_string(),
        //         })
        //     }
        //     _ => {
        // Grab the last text content block, parse it, and return
        for content in content.iter().rev() {
            let text = match content {
                ContentBlock::Text(Text { text }) => text,
                _ => continue,
            };
            let parsed_text = serde_json::from_str(text).map_err(|e| Error::OutputParsing {
                raw_output: text.to_string(),
                message: e.to_string(),
            })?;
            output_schema
                .validate(&parsed_text)
                .map_err(|e| Error::OutputValidation {
                    source: Box::new(e),
                })?;
            return Ok(parsed_text);
        }
        Err(Error::OutputParsing {
            raw_output: "".to_string(),
            message: "Output parsing failed due to no text content".to_string(),
        })
    }

    pub fn get_serialized_model_inferences(&self, input: &str) -> Vec<serde_json::Value> {
        self.model_inference_responses
            .iter()
            .map(|r| {
                let model_inference =
                    ModelInference::new(r.clone(), input.to_string(), self.inference_id);
                serde_json::to_value(model_inference).unwrap_or_default()
            })
            .collect()
    }
}

/// Helper type for output serialization of non-streaming responses.
/// This is required since we want to only return the `parsed_output` field
/// if there is an output schema for the Function being called.
pub struct InferenceResponseWithOutputSchema<'a> {
    pub inference_response: InferenceResponse,
    pub output_schema: Option<&'a JSONSchemaFromPath>,
}

/// Custom serializer is required since we want to only return the `parsed_output` field
/// if there is an output schema for the Function being called (null-valued if output parsing fails).
impl<'a> Serialize for InferenceResponseWithOutputSchema<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match &self.inference_response {
            InferenceResponse::Chat(chat_response) => {
                // Count the number of fields we'll be serializing
                let mut field_count = 5; // type, inference_id, created, content_blocks, usage
                if self.output_schema.is_some() {
                    field_count += 1; // We'll include parsed_output
                }

                let mut state = serializer.serialize_struct("InferenceResponse", field_count)?;

                state.serialize_field("type", "chat")?;
                state.serialize_field("inference_id", &chat_response.inference_id)?;
                state.serialize_field("created", &chat_response.created)?;
                state.serialize_field("content_blocks", &chat_response.content_blocks)?;
                state.serialize_field("usage", &chat_response.usage)?;

                if self.output_schema.is_some() {
                    state.serialize_field("parsed_output", &chat_response.parsed_output)?;
                }

                state.end()
            }
        }
    }
}

impl Inference {
    pub fn new(
        inference_response: InferenceResponse,
        input: String,
        episode_id: Uuid,
        function_name: String,
        variant_name: String,
        dynamic_tool_config: String,
        processing_time: Duration,
    ) -> Self {
        let processing_time_ms = processing_time.as_millis() as u32;
        match inference_response {
            InferenceResponse::Chat(chat_response) => Self {
                id: chat_response.inference_id,
                function_name,
                variant_name,
                episode_id,
                input,
                dynamic_tool_config,
                output: serde_json::to_string(&chat_response.content_blocks).unwrap_or_default(),
                processing_time_ms,
            },
        }
    }
}

// Function to get the current timestamp in seconds
fn current_timestamp() -> u64 {
    #[allow(clippy::expect_used)]
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs()
}

impl ModelInferenceResponseChunk {
    pub fn new(
        inference_id: Uuid,
        content: Vec<ContentBlockChunk>,
        usage: Option<Usage>,
        raw_response: String,
        latency: Duration,
    ) -> Self {
        Self {
            inference_id,
            content,
            created: current_timestamp(),
            usage,
            raw_response,
            latency,
        }
    }
}

impl From<ModelInferenceResponseChunk> for ChatInferenceResponseChunk {
    fn from(chunk: ModelInferenceResponseChunk) -> Self {
        Self {
            inference_id: chunk.inference_id,
            content: chunk.content,
            created: chunk.created,
            usage: chunk.usage,
        }
    }
}

pub fn collect_chunks(
    value: Vec<ModelInferenceResponseChunk>,
    output_schema: Option<&JSONSchemaFromPath>,
) -> Result<InferenceResponse, Error> {
    // NOTE: We will eventually need this to be per-inference-response-type and sensitive to the type of variant and function being called.

    let inference_id = value
        .first()
        .ok_or(Error::TypeConversion {
            message: "Attempted to create an InferenceResponse from an empty response chunk vector"
                .to_string(),
        })?
        .inference_id;
    let mut tool_call_blocks: HashMap<String, ContentBlock> = HashMap::new();
    let mut text_blocks: HashMap<String, ContentBlock> = HashMap::new();
    let raw: String = value
        .iter()
        .map(|chunk| chunk.raw_response.as_str())
        .collect::<Vec<&str>>()
        .join("\n");
    let mut usage: Usage = Usage::default();
    let mut ttft: Option<Duration> = None;
    let response_time = value
        .last()
        .ok_or(Error::TypeConversion {
            message: "Attempted to create an InferenceResponse from an empty response chunk vector"
                .to_string(),
        })?
        .latency;
    for chunk in value {
        for content in chunk.content {
            match content {
                ContentBlockChunk::Text(text) => {
                    match text_blocks.get_mut(&text.id) {
                        // If there is already a text block, append to it
                        Some(ContentBlock::Text(Text {
                            text: existing_text,
                        })) => {
                            existing_text.push_str(&text.text);
                        }
                        // If there is no text block, create one
                        _ => {
                            // We put this here and below rather than in the loop start because we
                            // only want to set TTFT if there is some real content
                            if ttft.is_none() {
                                ttft = Some(chunk.latency);
                            }
                            text_blocks.insert(text.id, text.text.into());
                        }
                    }
                }
                ContentBlockChunk::ToolCall(tool_call) => {
                    match tool_call_blocks.get_mut(&tool_call.id) {
                        // If there is already a tool call block with this id, append to it
                        Some(ContentBlock::ToolCall(existing_tool_call)) => {
                            // We assume that the name and ID are present and complete in the first chunk
                            existing_tool_call.arguments.push_str(&tool_call.arguments);
                        }
                        // If there is no tool call block, create one
                        _ => {
                            if ttft.is_none() {
                                ttft = Some(chunk.latency);
                            }
                            tool_call_blocks.insert(
                                tool_call.id.clone(),
                                ContentBlock::ToolCall(tool_call.into()),
                            );
                        }
                    }
                }
            };
        }
        if let Some(chunk_usage) = chunk.usage {
            usage.prompt_tokens = usage
                .prompt_tokens
                .saturating_add(chunk_usage.prompt_tokens);
            usage.completion_tokens = usage
                .completion_tokens
                .saturating_add(chunk_usage.completion_tokens);
        }
    }

    let ttft = ttft.ok_or(Error::TypeConversion {
        message: "Never got TTFT because there was never content in the response.".to_string(),
    })?;
    let latency = Latency::Streaming {
        ttft,
        response_time,
    };
    let mut content_blocks: Vec<ContentBlock> = tool_call_blocks.into_values().collect();
    content_blocks.extend(text_blocks.into_values());
    let model_response = ModelInferenceResponse::new(
        content_blocks.clone(),
        raw.clone(),
        usage.clone(),
        latency.clone(),
    );
    Ok(InferenceResponse::Chat(ChatInferenceResponse::new(
        inference_id,
        content_blocks,
        usage,
        vec![model_response],
        output_schema,
    )))
}

impl From<ToolCallChunk> for ToolCall {
    fn from(tool_call: ToolCallChunk) -> Self {
        // TODO (#30): explicitly handle tools both for streaming and non-streaming
        // as well as for Chat and Tool-style Functions
        Self {
            id: tool_call.id,
            name: tool_call.name,
            arguments: tool_call.arguments,
        }
    }
}

pub type InferenceResponseStream =
    Pin<Box<dyn Stream<Item = Result<ModelInferenceResponseChunk, Error>> + Send>>;

/// Types that go with `function.rs`
/// See that file for impls
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
#[serde(deny_unknown_fields)]
pub enum FunctionConfig {
    Chat(FunctionConfigChat),
    Json(FunctionConfigJson),
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionConfigChat {
    pub variants: HashMap<String, VariantConfig>, // variant name => variant config
    pub system_schema: Option<JSONSchemaFromPath>,
    pub user_schema: Option<JSONSchemaFromPath>,
    pub assistant_schema: Option<JSONSchemaFromPath>,
    #[serde(default)]
    pub tools: Vec<String>, // tool names
    #[serde(default)]
    pub tool_choice: ToolChoice,
    #[serde(default)] // defaults to false
    pub parallel_tool_calls: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionConfigJson {
    pub variants: HashMap<String, VariantConfig>, // variant name => variant config
    pub system_schema: Option<JSONSchemaFromPath>,
    pub user_schema: Option<JSONSchemaFromPath>,
    pub assistant_schema: Option<JSONSchemaFromPath>,
    pub output_schema: JSONSchemaFromPath, // schema is mandatory for JSON functions
}

/// Types that go with `variant.rs`
/// See that file for impls
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub enum VariantConfig {
    ChatCompletion(ChatCompletionConfig),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionConfig {
    pub weight: f64,
    pub model: String, // TODO (#85): validate that this model exists in the model config
    pub system_template: Option<PathBuf>,
    pub user_template: Option<PathBuf>,
    pub assistant_template: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_chat_inference_response() {
        // TODO (#30): handle the tool call case here. For now, we will always set those values to None.
        // Case 1: No output schema
        let inference_id = Uuid::now_v7();
        let content = vec!["Hello, world!".to_string().into()];
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
        };
        let model_inference_responses = vec![ModelInferenceResponse::new(
            content.clone(),
            "".to_string(),
            usage.clone(),
            Latency::NonStreaming {
                response_time: Duration::default(),
            },
        )];
        let chat_inference_response = ChatInferenceResponse::new(
            inference_id,
            content.clone(),
            usage.clone(),
            model_inference_responses,
            None,
        );
        assert_eq!(chat_inference_response.parsed_output, None);
        assert_eq!(chat_inference_response.content_blocks, content);
        assert_eq!(chat_inference_response.usage, usage);

        // Case 2: a JSON string that passes validation
        let inference_id = Uuid::now_v7();
        let created = current_timestamp();
        let json_content = r#"{"name": "John", "age": 30}"#.to_string();
        let content = vec![json_content.clone().into()];
        let usage = Usage {
            prompt_tokens: 15,
            completion_tokens: 25,
        };

        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(400),
        };
        let model_inference_responses = vec![ModelInferenceResponse::new(
            content.clone(),
            "".to_string(),
            usage.clone(),
            latency.clone(),
        )];

        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });
        let output_schema = JSONSchemaFromPath::from_value(&schema);

        let chat_inference_response = ChatInferenceResponse::new(
            inference_id,
            content.clone(),
            usage.clone(),
            model_inference_responses,
            Some(&output_schema),
        );

        assert_eq!(chat_inference_response.inference_id, inference_id);
        // Content will be the parsed Value if the JSON passes validation
        assert_eq!(chat_inference_response.content_blocks, content);
        assert_eq!(
            chat_inference_response.parsed_output,
            Some(serde_json::from_str(&json_content).unwrap())
        );
        assert_eq!(chat_inference_response.usage, usage);

        // TODO (#87): assert that the appropriate errors were logged in the next two test cases
        // Case 3: a JSON string that fails validation
        let invalid_json = r#"{"name": "John", "age": "thirty"}"#.to_string();
        let invalid_json_content = vec![invalid_json.into()];
        let model_inference_responses = vec![ModelInferenceResponse::new(
            invalid_json_content.clone(),
            "".to_string(),
            usage.clone(),
            Latency::NonStreaming {
                response_time: Duration::from_millis(300),
            },
        )];

        let chat_inference_response = ChatInferenceResponse::new(
            inference_id,
            invalid_json_content.clone(),
            usage.clone(),
            model_inference_responses,
            Some(&output_schema),
        );

        assert_eq!(chat_inference_response.inference_id, inference_id);
        // Content will be None if the validation fails
        assert_eq!(chat_inference_response.content_blocks, invalid_json_content);
        assert_eq!(chat_inference_response.parsed_output, None);
        assert_eq!(chat_inference_response.usage, usage);

        // Case 4: a malformed JSON
        let malformed_json = r#"{"name": "John", "age": 30,"#.to_string();
        let model_inference_responses = vec![ModelInferenceResponse::new(
            vec![malformed_json.clone().into()],
            "".to_string(),
            usage.clone(),
            Latency::NonStreaming {
                response_time: Duration::from_millis(310),
            },
        )];

        let chat_inference_response = ChatInferenceResponse::new(
            inference_id,
            vec![malformed_json.clone().into()],
            usage.clone(),
            model_inference_responses,
            Some(&output_schema),
        );

        assert_eq!(chat_inference_response.inference_id, inference_id);
        assert_eq!(chat_inference_response.created, created);
        // Content will be None if the JSON is malformed
        assert_eq!(
            chat_inference_response.content_blocks,
            vec![malformed_json.clone().into()]
        );
        assert_eq!(chat_inference_response.parsed_output, None);
        assert_eq!(chat_inference_response.usage, usage);
    }

    #[test]
    fn test_collect_chunks() {
        // Test case 1: empty chunks (should error)
        let chunks = vec![];
        let output_schema = None;
        let result = collect_chunks(chunks, output_schema);
        assert_eq!(
            result.unwrap_err(),
            Error::TypeConversion {
                message:
                    "Attempted to create an InferenceResponse from an empty response chunk vector"
                        .to_string(),
            }
        );

        // Test case 2: non-empty chunks with no tool calls but content exists
        let inference_id = Uuid::now_v7();
        let created = current_timestamp();
        let content = vec![ContentBlockChunk::Text(TextChunk {
            text: "Hello,".to_string(),
            id: "0".to_string(),
        })];
        let latency = Duration::from_millis(150);
        let chunks = vec![
            ModelInferenceResponseChunk {
                inference_id,
                content,
                created,
                usage: None,
                raw_response: "{\"message\": \"Hello}".to_string(),
                latency,
            },
            ModelInferenceResponseChunk {
                inference_id,
                content: vec![ContentBlockChunk::Text(TextChunk {
                    text: " world!".to_string(),
                    id: "0".to_string(),
                })],
                created,
                usage: Some(Usage {
                    prompt_tokens: 2,
                    completion_tokens: 4,
                }),
                raw_response: ", world!\"}".to_string(),
                latency: Duration::from_millis(250),
            },
        ];
        let response = collect_chunks(chunks, None).unwrap();
        let InferenceResponse::Chat(chat_response) = response;
        assert_eq!(chat_response.inference_id, inference_id);
        assert_eq!(chat_response.created, created);
        assert_eq!(
            chat_response.content_blocks,
            vec!["Hello, world!".to_string().into()]
        );
        assert_eq!(chat_response.parsed_output, None);
        assert_eq!(
            chat_response.usage,
            Usage {
                prompt_tokens: 2,
                completion_tokens: 4,
            }
        );

        // Test Case 3: a JSON string that passes validation and also include usage in each chunk
        let inference_id = Uuid::now_v7();
        let schema = JSONSchemaFromPath::from_value(&serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        }));
        let usage1 = Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
        };
        let usage2 = Usage {
            prompt_tokens: 5,
            completion_tokens: 10,
        };
        let chunks = vec![
            ModelInferenceResponseChunk {
                inference_id,
                content: vec![ContentBlockChunk::Text(TextChunk {
                    text: "{\"name\":".to_string(),
                    id: "0".to_string(),
                })],
                created,
                usage: Some(usage1.clone()),
                raw_response: "{\"name\":".to_string(),
                latency: Duration::from_millis(150),
            },
            ModelInferenceResponseChunk {
                inference_id,
                content: vec![ContentBlockChunk::Text(TextChunk {
                    text: "\"John\",\"age\":30}".to_string(),
                    id: "0".to_string(),
                })],
                created,
                usage: Some(usage2.clone()),
                raw_response: "\"John\",\"age\":30}".to_string(),
                latency: Duration::from_millis(250),
            },
        ];
        let response = collect_chunks(chunks, Some(&schema)).unwrap();
        let InferenceResponse::Chat(chat_response) = response;
        assert_eq!(chat_response.inference_id, inference_id);
        assert_eq!(
            chat_response.parsed_output,
            Some(serde_json::json!({"name": "John", "age": 30}))
        );
        assert_eq!(
            chat_response.content_blocks,
            vec!["{\"name\":\"John\",\"age\":30}".to_string().into()]
        );
        assert_eq!(
            chat_response.usage,
            Usage {
                prompt_tokens: 15,
                completion_tokens: 15,
            }
        );

        // Test Case 4: a JSON string that fails validation and usage only in last chunk
        let inference_id = Uuid::now_v7();
        let created = current_timestamp();
        let schema = JSONSchemaFromPath::from_value(&serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        }));
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
        };
        let chunks = vec![
            ModelInferenceResponseChunk {
                inference_id,
                content: vec![ContentBlockChunk::Text(TextChunk {
                    text: "{\"name\":".to_string(),
                    id: "0".to_string(),
                })],
                created,
                usage: Some(usage.clone()),
                raw_response: "{\"name\":".to_string(),
                latency: Duration::from_millis(100),
            },
            ModelInferenceResponseChunk {
                inference_id,
                content: vec![ContentBlockChunk::Text(TextChunk {
                    text: "\"John\"}".to_string(),
                    id: "0".to_string(),
                })],
                created,
                usage: None,
                raw_response: "\"John\"}".to_string(),
                latency: Duration::from_millis(200),
            },
        ];
        let result = collect_chunks(chunks, Some(&schema));
        assert!(result.is_ok());
        if let Ok(InferenceResponse::Chat(chat_response)) = result {
            assert_eq!(chat_response.inference_id, inference_id);
            assert_eq!(chat_response.created, created);
            // Content is None when we fail validation
            assert_eq!(chat_response.parsed_output, None);
            assert_eq!(
                chat_response.content_blocks,
                vec!["{\"name\":\"John\"}".to_string().into()]
            );
            assert_eq!(chat_response.usage, usage);
        } else {
            unreachable!("Expected Ok(InferenceResponse::Chat), got {:?}", result);
        }

        // Test case 5: chunks with some None content
        let inference_id = Uuid::now_v7();
        let created = current_timestamp();
        let schema = JSONSchemaFromPath::from_value(&serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        }));
        let usage = Usage {
            prompt_tokens: 15,
            completion_tokens: 10,
        };
        let chunks = vec![
            ModelInferenceResponseChunk {
                inference_id,
                content: vec![ContentBlockChunk::Text(TextChunk {
                    text: "{\"name\":\"John\",".to_string(),
                    id: "0".to_string(),
                })],
                created,
                usage: Some(usage.clone()),
                raw_response: "{\"name\":\"John\",".to_string(),
                latency: Duration::from_millis(100),
            },
            ModelInferenceResponseChunk {
                inference_id,
                content: vec![],
                created,
                usage: None,
                raw_response: "".to_string(),
                latency: Duration::from_millis(200),
            },
            ModelInferenceResponseChunk {
                inference_id,
                content: vec![ContentBlockChunk::Text(TextChunk {
                    text: "\"age\":30}".to_string(),
                    id: "0".to_string(),
                })],
                created,
                usage: None,
                raw_response: "\"age\":30}".to_string(),
                latency: Duration::from_millis(300),
            },
        ];
        let result = collect_chunks(chunks, Some(&schema));
        assert!(result.is_ok());
        if let Ok(InferenceResponse::Chat(chat_response)) = result {
            assert_eq!(chat_response.inference_id, inference_id);
            assert_eq!(chat_response.created, created);
            assert_eq!(
                chat_response.parsed_output,
                Some(serde_json::json!({"name": "John", "age": 30}))
            );
            assert_eq!(
                chat_response.content_blocks,
                vec!["{\"name\":\"John\",\"age\":30}".to_string().into()]
            );
            assert_eq!(chat_response.usage, usage);
        } else {
            unreachable!("Expected Ok(InferenceResponse::Chat), got {:?}", result);
        }
    }
}
