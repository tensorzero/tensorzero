use derive_builder::Builder;
use futures::Stream;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use std::{
    borrow::Cow,
    collections::HashMap,
    fmt,
    pin::Pin,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use uuid::Uuid;

use crate::endpoints::inference::InferenceParams;
use crate::function::FunctionConfig;
use crate::tool::ToolCallConfigDatabaseInsert;
use crate::tool::{ToolCall, ToolCallChunk, ToolCallConfig, ToolCallOutput, ToolResult};
use crate::{endpoints::inference::InferenceDatabaseInsertMetadata, variant::InferenceConfig};
use crate::{error::Error, variant::JsonMode};

/// Data flow in TensorZero
///
/// The flow of an inference request through TensorZero can be viewed as a series of transformations between types.
/// Most of them are defined below.

/// A request is made that contains an Input
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Input {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Value>,
    #[serde(default)]
    pub messages: Vec<InputMessage>,
}

/// InputMessage and Role are our representation of the input sent by the client
/// prior to any processing into LLM representations below.
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct InputMessage {
    pub role: Role,
    pub content: Vec<InputMessageContent>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputMessageContent {
    Text { value: Value },
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    // We may extend this in the future to include other types of content
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

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Text {
    pub text: String,
}

/// Core representation of the types of content that could go in or out of a model
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text(Text),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockOutput {
    Text(Text),
    ToolCall(ToolCallOutput),
}

/// A RequestMessage is a message sent to a model
#[derive(Clone, Debug, PartialEq, Serialize)]
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

#[derive(Clone, Default, Debug, PartialEq, Serialize)]
pub enum ModelInferenceRequestJsonMode {
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
#[derive(Builder, Clone, Debug, Default)]
#[builder(setter(into, strip_option), default)]
pub struct ModelInferenceRequest<'a> {
    pub messages: Vec<RequestMessage>,
    pub system: Option<String>,
    pub tool_config: Option<Cow<'a, ToolCallConfig>>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub seed: Option<u32>,
    pub stream: bool,
    pub json_mode: ModelInferenceRequestJsonMode,
    pub function_type: FunctionType,
    pub output_schema: Option<&'a Value>,
}

/// Each provider transforms a ModelInferenceRequest into a provider-specific (private) inference request type
/// that is suitable for serialization directly into a request to the provider.
///
/// In both non-streaming and streaming inference, each ModelProvider recieves data from the provider in a
/// a (private) provider-specific format that is then transformed into a ProviderInferenceResponse (non-streaming)
/// or a stream of ProviderInferenceResponseChunks (streaming).

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ProviderInferenceResponse {
    pub id: Uuid,
    pub created: u64,
    pub content: Vec<ContentBlock>,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
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

/// After a ProviderInferenceResponse is returned to the Model,
/// it is converted into a ModelInferenceResponse that includes additional metadata (such as the model provider name).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelInferenceResponse<'a> {
    pub id: Uuid,
    pub created: u64,
    pub content: Vec<ContentBlock>,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
    pub model_provider_name: &'a str,
}

/// Finally, in the Variant we convert the ModelInferenceResponse into a ModelInferenceResponseWithMetadata
/// that includes additional metadata (such as the model name).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelInferenceResponseWithMetadata<'a> {
    pub id: Uuid,
    pub created: u64,
    pub content: Vec<ContentBlock>,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
    pub model_provider_name: &'a str,
    pub model_name: &'a str,
}

/// As a Variant might make use of multiple model inferences, we then combine
/// one or more ModelInferenceResults into a single InferenceResult (but we keep the original ModelInferenceResults around for storage).
/// In the non-streaming case, this InferenceResult is converted into an InferenceResponse and sent to the client.
/// See below for streaming case.

/// This type contains the result of running a variant of a function
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceResult<'a> {
    Chat(ChatInferenceResult<'a>),
    Json(JsonInferenceResult<'a>),
}

#[derive(Clone, Debug, Serialize)]
pub struct ChatInferenceResult<'a> {
    pub inference_id: Uuid,
    created: u64,
    pub content: Vec<ContentBlockOutput>,
    pub usage: Usage,
    pub model_inference_results: Vec<ModelInferenceResponseWithMetadata<'a>>,
    pub inference_params: InferenceParams,
}

#[derive(Clone, Debug, Serialize)]
pub struct JsonInferenceResult<'a> {
    pub inference_id: Uuid,
    created: u64,
    pub output: JsonInferenceOutput,
    pub usage: Usage,
    pub model_inference_results: Vec<ModelInferenceResponseWithMetadata<'a>>,
    pub output_schema: Value,
    pub inference_params: InferenceParams,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct JsonInferenceOutput {
    pub raw: String,
    pub parsed: Option<Value>,
}

/// In the streaming case we convert ProviderInferenceResponseChunks into a InferenceResultChunk, which is then
/// converted into an InferenceResponseChunk and sent to the client.
/// We then collect all the InferenceResultChunks into an InferenceResult for validation and storage after the fact.

#[derive(Debug, Clone, PartialEq)]
pub struct ProviderInferenceResponseChunk {
    pub inference_id: Uuid,
    pub content: Vec<ContentBlockChunk>,
    pub created: u64,
    pub usage: Option<Usage>,
    pub raw_response: String,
    pub latency: Duration,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockChunk {
    Text(TextChunk),
    ToolCall(ToolCallChunk),
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TextChunk {
    pub id: String,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChatInferenceResultChunk {
    pub inference_id: Uuid,
    pub content: Vec<ContentBlockChunk>,
    pub created: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    pub latency: Duration,
    pub raw_response: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct JsonInferenceResultChunk {
    pub inference_id: Uuid,
    pub raw: String,
    pub created: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    pub latency: Duration,
    pub raw_response: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum InferenceResultChunk {
    Chat(ChatInferenceResultChunk),
    Json(JsonInferenceResultChunk),
}

/// Alongside the response, we also store information about what happened during the request.
/// For this we convert the InferenceResult into a ChatInferenceDatabaseInsert or JsonInferenceDatabaseInsert and ModelInferenceDatabaseInserts,
/// which are written to ClickHouse tables of the same name asynchronously.

#[derive(Debug, Serialize)]
pub struct ChatInferenceDatabaseInsert {
    pub id: Uuid,
    pub function_name: String,
    pub variant_name: String,
    pub episode_id: Uuid,
    pub input: String,
    pub output: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_params: Option<ToolCallConfigDatabaseInsert>,
    pub inference_params: InferenceParams,
    pub processing_time_ms: u32,
}

#[derive(Debug, Serialize)]
pub struct JsonInferenceDatabaseInsert {
    pub id: Uuid,
    pub function_name: String,
    pub variant_name: String,
    pub episode_id: Uuid,
    pub input: String,
    pub output: String,
    pub inference_params: InferenceParams,
    pub processing_time_ms: u32,
    pub output_schema: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelInferenceDatabaseInsert {
    pub id: Uuid,
    pub inference_id: Uuid,
    pub raw_request: String,
    pub raw_response: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub response_time_ms: u32,
    pub model_name: String,
    pub model_provider_name: String,
    pub ttft_ms: Option<u32>,
}

#[cfg(test)]
impl From<String> for InputMessageContent {
    fn from(text: String) -> Self {
        InputMessageContent::Text {
            value: Value::String(text),
        }
    }
}

#[cfg(any(test, feature = "e2e_tests"))]
impl From<String> for ContentBlockOutput {
    fn from(text: String) -> Self {
        ContentBlockOutput::Text(Text { text })
    }
}

impl From<Value> for InputMessageContent {
    fn from(value: Value) -> Self {
        InputMessageContent::Text { value }
    }
}

impl<'de> Deserialize<'de> for InputMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            role: Role,
            content: ContentHelper,
        }

        #[derive(Deserialize)]
        #[serde(untagged)]
        enum ContentHelper {
            Single(String),
            Object(serde_json::Map<String, Value>),
            Multiple(Vec<InputMessageContent>),
        }

        let helper = Helper::deserialize(deserializer)?;

        let content = match helper.content {
            ContentHelper::Single(text) => {
                vec![InputMessageContent::Text {
                    value: Value::String(text),
                }]
            }
            ContentHelper::Object(object) => {
                vec![InputMessageContent::Text {
                    value: Value::Object(object),
                }]
            }
            ContentHelper::Multiple(content) => content,
        };

        Ok(InputMessage {
            role: helper.role,
            content,
        })
    }
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

impl<'a> ModelInferenceResponse<'a> {
    pub fn new(
        provider_inference_response: ProviderInferenceResponse,
        model_provider_name: &'a str,
    ) -> Self {
        Self {
            id: provider_inference_response.id,
            created: provider_inference_response.created,
            content: provider_inference_response.content,
            raw_request: provider_inference_response.raw_request,
            raw_response: provider_inference_response.raw_response,
            usage: provider_inference_response.usage,
            latency: provider_inference_response.latency,
            model_provider_name,
        }
    }
}

impl<'a> ModelInferenceResponseWithMetadata<'a> {
    pub fn new(model_inference_response: ModelInferenceResponse<'a>, model_name: &'a str) -> Self {
        Self {
            id: model_inference_response.id,
            created: model_inference_response.created,
            content: model_inference_response.content,
            raw_request: model_inference_response.raw_request,
            raw_response: model_inference_response.raw_response,
            usage: model_inference_response.usage,
            latency: model_inference_response.latency,
            model_provider_name: model_inference_response.model_provider_name,
            model_name,
        }
    }
}

impl ModelInferenceDatabaseInsert {
    pub fn new(result: ModelInferenceResponseWithMetadata, inference_id: Uuid) -> Self {
        let (latency_ms, ttft_ms) = match result.latency {
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
            raw_request: result.raw_request,
            raw_response: result.raw_response,
            input_tokens: result.usage.input_tokens,
            output_tokens: result.usage.output_tokens,
            response_time_ms: latency_ms,
            ttft_ms,
            model_provider_name: result.model_provider_name.to_string(),
            model_name: result.model_name.to_string(),
        }
    }
}

impl ProviderInferenceResponse {
    pub fn new(
        content: Vec<ContentBlock>,
        raw_request: String,
        raw_response: String,
        usage: Usage,
        latency: Latency,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            created: current_timestamp(),
            content,
            raw_request,
            raw_response,
            usage,
            latency,
        }
    }
}

impl<'a> InferenceResult<'a> {
    pub fn get_serialized_model_inferences(&self) -> Vec<serde_json::Value> {
        let model_inference_responses = match self {
            InferenceResult::Chat(chat_result) => &chat_result.model_inference_results,
            InferenceResult::Json(json_result) => &json_result.model_inference_results,
        };
        let inference_id = match self {
            InferenceResult::Chat(chat_result) => chat_result.inference_id,
            InferenceResult::Json(json_result) => json_result.inference_id,
        };
        model_inference_responses
            .iter()
            .map(|r| {
                let model_inference = ModelInferenceDatabaseInsert::new(r.clone(), inference_id);
                serde_json::to_value(model_inference).unwrap_or_default()
            })
            .collect()
    }

    pub fn usage(&self) -> &Usage {
        match self {
            InferenceResult::Chat(chat_result) => &chat_result.usage,
            InferenceResult::Json(json_result) => &json_result.usage,
        }
    }

    pub fn set_usage(&mut self, usage: Usage) {
        match self {
            InferenceResult::Chat(chat_result) => chat_result.usage = usage,
            InferenceResult::Json(json_result) => json_result.usage = usage,
        }
    }

    pub fn mut_model_inference_results(
        &mut self,
    ) -> &mut Vec<ModelInferenceResponseWithMetadata<'a>> {
        match self {
            InferenceResult::Chat(chat_result) => &mut chat_result.model_inference_results,
            InferenceResult::Json(json_result) => &mut json_result.model_inference_results,
        }
    }

    pub fn owned_model_inference_results(self) -> Vec<ModelInferenceResponseWithMetadata<'a>> {
        match self {
            InferenceResult::Chat(chat_result) => chat_result.model_inference_results,
            InferenceResult::Json(json_result) => json_result.model_inference_results,
        }
    }
}

impl<'a> JsonInferenceResult<'a> {
    pub fn new(
        inference_id: Uuid,
        raw: String,
        parsed: Option<Value>,
        usage: Usage,
        model_inference_results: Vec<ModelInferenceResponseWithMetadata<'a>>,
        output_schema: Value,
        inference_params: InferenceParams,
    ) -> Self {
        let output = JsonInferenceOutput { raw, parsed };
        Self {
            inference_id,
            created: current_timestamp(),
            output,
            usage,
            model_inference_results,
            output_schema,
            inference_params,
        }
    }
}

impl<'a> ChatInferenceResult<'a> {
    pub async fn new(
        inference_id: Uuid,
        raw_content: Vec<ContentBlock>,
        usage: Usage,
        model_inference_results: Vec<ModelInferenceResponseWithMetadata<'a>>,
        tool_config: Option<&ToolCallConfig>,
        inference_params: InferenceParams,
    ) -> Self {
        #[allow(clippy::expect_used)]
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_secs();
        let content = parse_chat_output(raw_content, tool_config).await;
        Self {
            inference_id,
            created,
            content,
            usage,
            model_inference_results,
            inference_params,
        }
    }
}

pub async fn parse_chat_output(
    content: Vec<ContentBlock>,
    tool_config: Option<&ToolCallConfig>,
) -> Vec<ContentBlockOutput> {
    if content.is_empty() {
        Error::Inference {
            message: "No content blocks in inference result".to_string(),
        }
        .log();
    }

    let mut output = Vec::new();
    for content in content.into_iter() {
        match content {
            ContentBlock::Text(text) => {
                output.push(ContentBlockOutput::Text(text));
            }
            ContentBlock::ToolCall(tool_call) => {
                // Parse the tool call arguments
                let tool_call_output = ToolCallOutput::new(tool_call, tool_config).await;
                output.push(ContentBlockOutput::ToolCall(tool_call_output));
            }
            ContentBlock::ToolResult(tool_result) => {
                Error::OutputParsing {
                    message: "Tool results are not supported in output for Chat functions"
                        .to_string(),
                    raw_output: serde_json::to_string(&tool_result).unwrap_or_default(),
                }
                .log();
            }
        }
    }
    output
}

impl ChatInferenceDatabaseInsert {
    pub fn new(
        chat_result: ChatInferenceResult,
        input: String,
        metadata: InferenceDatabaseInsertMetadata,
    ) -> Self {
        let processing_time_ms = metadata.processing_time.as_millis() as u32;

        let tool_params = metadata.tool_params.map(ToolCallConfigDatabaseInsert::from);
        let inference_params = chat_result.inference_params;
        let output = serde_json::to_string(&chat_result.content)
            .map_err(|e| Error::Serialization {
                message: format!("Failed to serialize output: {}", e),
            })
            .unwrap_or_else(|e| {
                e.log();
                String::new()
            });

        Self {
            id: chat_result.inference_id,
            function_name: metadata.function_name,
            variant_name: metadata.variant_name,
            episode_id: metadata.episode_id,
            input,
            tool_params,
            inference_params,
            output,
            processing_time_ms,
        }
    }
}

impl JsonInferenceDatabaseInsert {
    pub fn new(
        json_result: JsonInferenceResult,
        input: String,
        metadata: InferenceDatabaseInsertMetadata,
    ) -> Self {
        let processing_time_ms = metadata.processing_time.as_millis() as u32;

        let inference_params = json_result.inference_params;

        let output = serde_json::to_string(&json_result.output)
            .map_err(|e| Error::Serialization {
                message: format!("Failed to serialize output: {}", e),
            })
            .unwrap_or_else(|e| {
                e.log();
                String::new()
            });
        let output_schema = serde_json::to_string(&json_result.output_schema)
            .map_err(|e| Error::Serialization {
                message: format!("Failed to serialize output schema: {}", e),
            })
            .unwrap_or_else(|e| {
                e.log();
                String::new()
            });
        Self {
            id: json_result.inference_id,
            function_name: metadata.function_name,
            variant_name: metadata.variant_name,
            episode_id: metadata.episode_id,
            input,
            inference_params,
            output,
            processing_time_ms,
            output_schema,
        }
    }
}

// Function to get the current timestamp in seconds
pub fn current_timestamp() -> u64 {
    #[allow(clippy::expect_used)]
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs()
}

impl ProviderInferenceResponseChunk {
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

impl InferenceResultChunk {
    pub fn inference_id(&self) -> Uuid {
        match self {
            InferenceResultChunk::Chat(chunk) => chunk.inference_id,
            InferenceResultChunk::Json(chunk) => chunk.inference_id,
        }
    }

    pub fn latency(&self) -> Duration {
        match self {
            InferenceResultChunk::Chat(chunk) => chunk.latency,
            InferenceResultChunk::Json(chunk) => chunk.latency,
        }
    }

    pub fn usage(&self) -> Option<&Usage> {
        match self {
            InferenceResultChunk::Chat(chunk) => chunk.usage.as_ref(),
            InferenceResultChunk::Json(chunk) => chunk.usage.as_ref(),
        }
    }

    pub fn raw_response(&self) -> &str {
        match self {
            InferenceResultChunk::Chat(chunk) => &chunk.raw_response,
            InferenceResultChunk::Json(chunk) => &chunk.raw_response,
        }
    }
}

impl InferenceResultChunk {
    pub fn new(chunk: ProviderInferenceResponseChunk, function: &FunctionConfig) -> Self {
        match function {
            FunctionConfig::Chat(_) => Self::Chat(chunk.into()),
            FunctionConfig::Json(_) => Self::Json(chunk.into()),
        }
    }
}

impl From<ProviderInferenceResponseChunk> for ChatInferenceResultChunk {
    fn from(chunk: ProviderInferenceResponseChunk) -> Self {
        Self {
            inference_id: chunk.inference_id,
            content: chunk.content,
            created: chunk.created,
            usage: chunk.usage,
            latency: chunk.latency,
            raw_response: chunk.raw_response,
        }
    }
}

impl From<ToolCallOutput> for ToolCall {
    fn from(output: ToolCallOutput) -> Self {
        Self {
            id: output.id,
            name: output.raw_name,
            arguments: output.raw_arguments,
        }
    }
}

impl From<ContentBlockOutput> for ContentBlock {
    fn from(output: ContentBlockOutput) -> Self {
        match output {
            ContentBlockOutput::Text(text) => ContentBlock::Text(text),
            ContentBlockOutput::ToolCall(tool_call_output) => {
                ContentBlock::ToolCall(tool_call_output.into())
            }
        }
    }
}

/// We use best-effort to reconstruct the raw response for JSON functions
/// They might either return a ToolCallChunk or a TextChunk
/// We take the string from either of these (from the last block if there are multiple)
/// and use that as the raw response.
impl From<ProviderInferenceResponseChunk> for JsonInferenceResultChunk {
    fn from(mut chunk: ProviderInferenceResponseChunk) -> Self {
        let raw = match chunk.content.pop() {
            Some(ContentBlockChunk::ToolCall(tool_call)) => tool_call.raw_arguments.to_owned(),
            Some(ContentBlockChunk::Text(text)) => text.text.to_owned(),
            None => String::new(),
        };

        Self {
            inference_id: chunk.inference_id,
            raw,
            created: chunk.created,
            usage: chunk.usage,
            latency: chunk.latency,
            raw_response: chunk.raw_response,
        }
    }
}

pub async fn collect_chunks<'a>(
    value: Vec<InferenceResultChunk>,
    function: &FunctionConfig,
    inference_config: &InferenceConfig<'a>,
    model_name: &'a str,
    model_provider_name: &'a str,
    raw_request: String,
    inference_params: InferenceParams,
) -> Result<InferenceResult<'a>, Error> {
    // NOTE: We will eventually need this to be per-inference-response-type and sensitive to the type of variant and function being called.

    let inference_id = value
        .first()
        .ok_or(Error::TypeConversion {
            message: "Attempted to create an InferenceResult from an empty response chunk vector"
                .to_string(),
        })?
        .inference_id();
    let mut tool_call_blocks: HashMap<String, ContentBlock> = HashMap::new();
    let mut text_blocks: HashMap<String, ContentBlock> = HashMap::new();
    let raw_response: String = value
        .iter()
        .map(|chunk| chunk.raw_response())
        .collect::<Vec<&str>>()
        .join("\n");
    let mut usage: Usage = Usage::default();
    let mut ttft: Option<Duration> = None;
    let response_time = value
        .last()
        .ok_or(Error::TypeConversion {
            message: "Attempted to create an InferenceResult from an empty response chunk vector"
                .to_string(),
        })?
        .latency();
    for chunk in value {
        if let Some(chunk_usage) = chunk.usage() {
            usage.input_tokens = usage.input_tokens.saturating_add(chunk_usage.input_tokens);
            usage.output_tokens = usage
                .output_tokens
                .saturating_add(chunk_usage.output_tokens);
        }
        match chunk {
            InferenceResultChunk::Chat(chunk) => {
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
                                    existing_tool_call
                                        .arguments
                                        .push_str(&tool_call.raw_arguments);
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
                    }
                }
            }
            InferenceResultChunk::Json(chunk) => {
                match text_blocks.get_mut("") {
                    // If there is already a text block, append to it
                    Some(ContentBlock::Text(Text {
                        text: existing_text,
                    })) => {
                        existing_text.push_str(&chunk.raw);
                    }
                    // If there is no text block, create one
                    _ => {
                        // We put this here and below rather than in the loop start because we
                        // only want to set TTFT if there is some real content
                        if ttft.is_none() {
                            ttft = Some(chunk.latency);
                        }
                        text_blocks.insert("".to_string(), chunk.raw.into());
                    }
                }
            }
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
    let model_response = ProviderInferenceResponse::new(
        content_blocks.clone(),
        raw_request,
        raw_response,
        usage.clone(),
        latency.clone(),
    );
    let model_inference_response = ModelInferenceResponse::new(model_response, model_provider_name);
    let model_inference_result =
        ModelInferenceResponseWithMetadata::new(model_inference_response, model_name);
    function
        .prepare_response(
            inference_id,
            content_blocks,
            usage,
            vec![model_inference_result],
            inference_config,
            inference_params,
        )
        .await
}

impl From<ToolCallChunk> for ToolCall {
    fn from(tool_call: ToolCallChunk) -> Self {
        Self {
            id: tool_call.id,
            name: tool_call.raw_name,
            arguments: tool_call.raw_arguments,
        }
    }
}

pub type ProviderInferenceResponseStream =
    Pin<Box<dyn Stream<Item = Result<ProviderInferenceResponseChunk, Error>> + Send>>;

pub type InferenceResultStream =
    Pin<Box<dyn Stream<Item = Result<InferenceResultChunk, Error>> + Send>>;

impl From<&JsonMode> for ModelInferenceRequestJsonMode {
    fn from(json_enforcement: &JsonMode) -> Self {
        match json_enforcement {
            JsonMode::On => ModelInferenceRequestJsonMode::On,
            JsonMode::Strict => ModelInferenceRequestJsonMode::Strict,
            JsonMode::ImplicitTool => ModelInferenceRequestJsonMode::Off,
            JsonMode::Off => ModelInferenceRequestJsonMode::Off,
        }
    }
}

impl<'a> std::iter::Sum<&'a Usage> for Usage {
    fn sum<I: Iterator<Item = &'a Usage>>(iter: I) -> Self {
        iter.fold(Usage::default(), |mut acc, u| {
            acc.input_tokens = acc.input_tokens.saturating_add(u.input_tokens);
            acc.output_tokens = acc.output_tokens.saturating_add(u.output_tokens);
            acc
        })
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use serde_json::json;

    use crate::function::{FunctionConfigChat, FunctionConfigJson};
    use crate::inference::providers::common::get_temperature_tool_config;
    use crate::jsonschema_util::{DynamicJSONSchema, JSONSchemaFromPath};
    use crate::minijinja_util::TemplateConfig;
    use crate::tool::ToolChoice;

    use super::*;

    #[tokio::test]
    async fn test_create_chat_inference_response() {
        // TODO (#30): handle the tool call case here. For now, we will always set those values to None.
        // Case 1: No output schema
        let inference_id = Uuid::now_v7();
        let content = vec!["Hello, world!".to_string().into()];
        let usage = Usage {
            input_tokens: 10,
            output_tokens: 20,
        };
        let raw_request = "raw request".to_string();
        let model_inference_responses = vec![ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            content: content.clone(),
            raw_request: raw_request.clone(),
            raw_response: "".to_string(),
            usage: usage.clone(),
            latency: Latency::NonStreaming {
                response_time: Duration::default(),
            },
            model_provider_name: "test_provider",
            model_name: "test_model",
        }];
        let chat_inference_response = ChatInferenceResult::new(
            inference_id,
            content.clone(),
            usage.clone(),
            model_inference_responses,
            None,
            InferenceParams::default(),
        )
        .await;
        let output_content = ["Hello, world!".to_string().into()];
        assert_eq!(chat_inference_response.content, output_content);
        assert_eq!(chat_inference_response.usage, usage);
        assert_eq!(chat_inference_response.model_inference_results.len(), 1);
        let model_inference_result = chat_inference_response
            .model_inference_results
            .first()
            .unwrap();
        assert_eq!(model_inference_result.model_name, "test_model");
        assert_eq!(model_inference_result.model_provider_name, "test_provider");
        assert_eq!(model_inference_result.raw_request, raw_request);
        // Case 2: A tool call that fails argument validation
        let inference_id = Uuid::now_v7();
        let content = vec![ContentBlock::ToolCall(ToolCall {
            name: "get_temperature".to_string(),
            arguments: r#"{"where": "the moon"}"#.to_string(),
            id: "0".to_string(),
        })];
        let model_inference_responses = vec![ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            content: content.clone(),
            raw_request: raw_request.clone(),
            raw_response: "".to_string(),
            usage: usage.clone(),
            latency: Latency::NonStreaming {
                response_time: Duration::default(),
            },
            model_provider_name: "test_provider",
            model_name: "test_model",
        }];

        let weather_tool_config = get_temperature_tool_config();
        let chat_inference_response = ChatInferenceResult::new(
            inference_id,
            content,
            usage.clone(),
            model_inference_responses,
            Some(&weather_tool_config),
            InferenceParams::default(),
        )
        .await;
        assert_eq!(chat_inference_response.content.len(), 1);
        let tool_call_block = chat_inference_response.content.first().unwrap();
        match tool_call_block {
            ContentBlockOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "get_temperature");
                assert_eq!(tool_call.raw_arguments, r#"{"where": "the moon"}"#);
                assert_eq!(tool_call.id, "0");
                assert_eq!(tool_call.name, Some("get_temperature".to_string()));
                assert_eq!(tool_call.arguments, None);
            }
            _ => panic!("Expected a tool call block"),
        }

        // Case 3: A tool call that fails name validation
        let inference_id = Uuid::now_v7();
        let content = vec![ContentBlock::ToolCall(ToolCall {
            name: "bad name".to_string(),
            arguments: r#"{"where": "the moon"}"#.to_string(),
            id: "0".to_string(),
        })];
        let model_inference_responses = vec![ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            content: content.clone(),
            raw_request: raw_request.clone(),
            raw_response: "".to_string(),
            usage: usage.clone(),
            latency: Latency::NonStreaming {
                response_time: Duration::default(),
            },
            model_provider_name: "test_provider",
            model_name: "test_model",
        }];

        let chat_inference_response = ChatInferenceResult::new(
            inference_id,
            content,
            usage.clone(),
            model_inference_responses,
            Some(&weather_tool_config),
            InferenceParams::default(),
        )
        .await;
        assert_eq!(chat_inference_response.content.len(), 1);
        let tool_call_block = chat_inference_response.content.first().unwrap();
        match tool_call_block {
            ContentBlockOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "bad name");
                assert_eq!(tool_call.raw_arguments, r#"{"where": "the moon"}"#);
                assert_eq!(tool_call.id, "0");
                assert_eq!(tool_call.name, None);
                assert_eq!(tool_call.arguments, None);
            }
            _ => panic!("Expected a tool call block"),
        }

        // Case 4: A tool call that passes validation
        let inference_id = Uuid::now_v7();
        let content = vec![ContentBlock::ToolCall(ToolCall {
            name: "get_temperature".to_string(),
            arguments: r#"{"location": "the moon", "units": "celsius"}"#.to_string(),
            id: "0".to_string(),
        })];
        let model_inference_responses = vec![ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            content: content.clone(),
            raw_request: raw_request.clone(),
            raw_response: "".to_string(),
            usage: usage.clone(),
            latency: Latency::NonStreaming {
                response_time: Duration::default(),
            },
            model_provider_name: "test_provider",
            model_name: "test_model",
        }];

        let chat_inference_response = ChatInferenceResult::new(
            inference_id,
            content,
            usage.clone(),
            model_inference_responses,
            Some(&weather_tool_config),
            InferenceParams::default(),
        )
        .await;
        assert_eq!(chat_inference_response.content.len(), 1);
        let tool_call_block = chat_inference_response.content.first().unwrap();
        match tool_call_block {
            ContentBlockOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "get_temperature");
                assert_eq!(
                    tool_call.raw_arguments,
                    r#"{"location": "the moon", "units": "celsius"}"#
                );
                assert_eq!(tool_call.id, "0");
                assert_eq!(tool_call.name, Some("get_temperature".to_string()));
                assert_eq!(
                    tool_call.arguments,
                    Some(
                        serde_json::from_str(r#"{"location": "the moon", "units": "celsius"}"#)
                            .unwrap()
                    )
                );
            }
            _ => panic!("Expected a tool call block"),
        }
    }

    #[tokio::test]
    async fn test_collect_chunks() {
        // Test case 1: empty chunks (should error)
        let tool_config = ToolCallConfig {
            tools_available: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: false,
        };
        let inference_config = InferenceConfig {
            tool_config: Some(tool_config),
            templates: &TemplateConfig::default(),
            function_name: "".to_string(),
            variant_name: "".to_string(),
            dynamic_output_schema: None,
        };
        let chunks = vec![];
        let function = FunctionConfig::Chat(FunctionConfigChat::default());
        let model_name = "test_model";
        let model_provider_name = "test_provider";
        let raw_request = "raw request".to_string();
        let result = collect_chunks(
            chunks,
            &function,
            &inference_config,
            model_name,
            model_provider_name,
            raw_request.clone(),
            InferenceParams::default(),
        )
        .await;
        assert_eq!(
            result.unwrap_err(),
            Error::TypeConversion {
                message:
                    "Attempted to create an InferenceResult from an empty response chunk vector"
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
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                inference_id,
                content,
                created,
                usage: None,
                raw_response: "{\"message\": \"Hello}".to_string(),
                latency,
            }),
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                inference_id,
                content: vec![ContentBlockChunk::Text(TextChunk {
                    text: " world!".to_string(),
                    id: "0".to_string(),
                })],
                created,
                usage: Some(Usage {
                    input_tokens: 2,
                    output_tokens: 4,
                }),
                raw_response: ", world!\"}".to_string(),
                latency: Duration::from_millis(250),
            }),
        ];
        let result = collect_chunks(
            chunks,
            &function,
            &inference_config,
            model_name,
            model_provider_name,
            raw_request.clone(),
            InferenceParams::default(),
        )
        .await
        .unwrap();
        let chat_result = match result {
            InferenceResult::Chat(chat_result) => chat_result,
            _ => panic!("Expected Chat inference response"),
        };
        assert_eq!(chat_result.inference_id, inference_id);
        assert_eq!(chat_result.created, created);
        assert_eq!(
            chat_result.content,
            vec!["Hello, world!".to_string().into()]
        );
        assert_eq!(
            chat_result.usage,
            Usage {
                input_tokens: 2,
                output_tokens: 4,
            }
        );
        assert_eq!(chat_result.model_inference_results.len(), 1);
        let model_inference_result = chat_result.model_inference_results.first().unwrap();
        assert_eq!(model_inference_result.model_name, model_name);
        assert_eq!(
            model_inference_result.model_provider_name,
            model_provider_name
        );
        assert_eq!(model_inference_result.raw_request, raw_request);
        // Test Case 3: a JSON string that passes validation and also include usage in each chunk
        let inference_id = Uuid::now_v7();
        let output_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        });
        let implicit_tool_call_config = ToolCallConfig::implicit_from_value(&output_schema);
        let output_schema = JSONSchemaFromPath::from_value(&output_schema).unwrap();
        let function_config = FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            implicit_tool_call_config,
            output_schema,
        });
        let usage1 = Usage {
            input_tokens: 10,
            output_tokens: 5,
        };
        let usage2 = Usage {
            input_tokens: 5,
            output_tokens: 10,
        };
        let chunks = vec![
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                inference_id,
                raw: "{\"name\":".to_string(),
                created,
                usage: Some(usage1.clone()),
                raw_response: "{\"name\":".to_string(),
                latency: Duration::from_millis(150),
            }),
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                inference_id,
                raw: "\"John\",\"age\":30}".to_string(),
                created,
                usage: Some(usage2.clone()),
                raw_response: "\"John\",\"age\":30}".to_string(),
                latency: Duration::from_millis(250),
            }),
        ];
        let response = collect_chunks(
            chunks,
            &function_config,
            &inference_config,
            model_name,
            model_provider_name,
            raw_request.clone(),
            InferenceParams::default(),
        )
        .await
        .unwrap();
        match response {
            InferenceResult::Json(json_result) => {
                assert_eq!(json_result.inference_id, inference_id);
                assert_eq!(
                    json_result.output.parsed,
                    Some(serde_json::json!({"name": "John", "age": 30}))
                );
                assert_eq!(
                    json_result.output.raw,
                    "{\"name\":\"John\",\"age\":30}".to_string()
                );
                assert_eq!(
                    json_result.usage,
                    Usage {
                        input_tokens: 15,
                        output_tokens: 15,
                    }
                );
                assert_eq!(json_result.model_inference_results.len(), 1);
                let model_inference_result = json_result.model_inference_results.first().unwrap();
                assert_eq!(model_inference_result.model_name, model_name);
                assert_eq!(
                    model_inference_result.model_provider_name,
                    model_provider_name
                );
                assert_eq!(model_inference_result.raw_request, raw_request);
            }
            _ => panic!("Expected Json inference response"),
        }

        // Test Case 4: a JSON string that fails validation and usage only in last chunk
        let inference_id = Uuid::now_v7();
        let created = current_timestamp();
        let usage = Usage {
            input_tokens: 10,
            output_tokens: 5,
        };
        let chunks = vec![
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                inference_id,
                raw: "{\"name\":".to_string(),
                created,
                usage: Some(usage.clone()),
                raw_response: "{\"name\":".to_string(),
                latency: Duration::from_millis(100),
            }),
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                inference_id,
                raw: "\"John\"}".to_string(),
                created,
                usage: None,
                raw_response: "\"John\"}".to_string(),
                latency: Duration::from_millis(200),
            }),
        ];
        let result = collect_chunks(
            chunks,
            &function_config,
            &inference_config,
            model_name,
            model_provider_name,
            raw_request.clone(),
            InferenceParams::default(),
        )
        .await;
        assert!(result.is_ok());
        match result {
            Ok(InferenceResult::Json(json_result)) => {
                assert_eq!(json_result.inference_id, inference_id);
                assert_eq!(json_result.created, created);
                assert_eq!(json_result.usage, usage);
                assert_eq!(json_result.output.parsed, None);
                assert_eq!(json_result.output.raw, "{\"name\":\"John\"}".to_string());
                assert_eq!(json_result.model_inference_results.len(), 1);
                let model_inference_result = json_result.model_inference_results.first().unwrap();
                assert_eq!(model_inference_result.model_name, model_name);
                assert_eq!(
                    model_inference_result.model_provider_name,
                    model_provider_name
                );
                assert_eq!(model_inference_result.raw_request, raw_request);
            }
            _ => panic!("Expected Json inference response"),
        }

        // Test case 5: chunks with some None content
        let inference_id = Uuid::now_v7();
        let created = current_timestamp();
        let usage = Usage {
            input_tokens: 15,
            output_tokens: 10,
        };
        let chunks = vec![
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                inference_id,
                raw: "{\"name\":\"John\",".to_string(),
                created,
                usage: Some(usage.clone()),
                raw_response: "{\"name\":\"John\",".to_string(),
                latency: Duration::from_millis(100),
            }),
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                inference_id,
                raw: "".to_string(),
                created,
                usage: None,
                raw_response: "".to_string(),
                latency: Duration::from_millis(200),
            }),
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                inference_id,
                raw: "\"age\":30}".to_string(),
                created,
                usage: None,
                raw_response: "\"age\":30}".to_string(),
                latency: Duration::from_millis(300),
            }),
        ];
        let result = collect_chunks(
            chunks,
            &function,
            &inference_config,
            model_name,
            model_provider_name,
            raw_request.clone(),
            InferenceParams::default(),
        )
        .await;
        if let Ok(InferenceResult::Chat(chat_response)) = result {
            assert_eq!(chat_response.inference_id, inference_id);
            assert_eq!(chat_response.created, created);
            assert_eq!(
                chat_response.content,
                vec!["{\"name\":\"John\",\"age\":30}".to_string().into()]
            );
            assert_eq!(chat_response.usage, usage);
            assert_eq!(chat_response.model_inference_results.len(), 1);
            let model_inference_result = chat_response.model_inference_results.first().unwrap();
            assert_eq!(model_inference_result.model_name, model_name);
            assert_eq!(
                model_inference_result.model_provider_name,
                model_provider_name
            );
            assert_eq!(model_inference_result.raw_request, raw_request);
        } else {
            panic!("Expected Ok(InferenceResult::Chat), got {:?}", result);
        }

        // Test Case 6: a JSON function with implicit tool call config
        let inference_id = Uuid::now_v7();
        let created = current_timestamp();
        let output_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        });
        let implicit_tool_call_config = ToolCallConfig::implicit_from_value(&output_schema);
        let output_schema = JSONSchemaFromPath::from_value(&output_schema).unwrap();
        let function_config = FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            implicit_tool_call_config,
            output_schema,
        });
        let usage1 = Usage {
            input_tokens: 10,
            output_tokens: 5,
        };
        let usage2 = Usage {
            input_tokens: 5,
            output_tokens: 10,
        };
        let chunks = vec![
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                inference_id,
                raw: "{\"name\":".to_string(),
                created,
                usage: Some(usage1.clone()),
                raw_response: "{\"name\":".to_string(),
                latency: Duration::from_millis(150),
            }),
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                inference_id,
                raw: "\"John\",\"age\":30}".to_string(),
                created,
                usage: Some(usage2.clone()),
                raw_response: "\"John\",\"age\":30}".to_string(),
                latency: Duration::from_millis(250),
            }),
        ];
        let response = collect_chunks(
            chunks,
            &function_config,
            &inference_config,
            model_name,
            model_provider_name,
            raw_request.clone(),
            InferenceParams::default(),
        )
        .await
        .unwrap();
        match response {
            InferenceResult::Json(json_result) => {
                assert_eq!(json_result.inference_id, inference_id);
                assert_eq!(
                    json_result.output.parsed,
                    Some(serde_json::json!({"name": "John", "age": 30}))
                );
                assert_eq!(
                    json_result.output.raw,
                    "{\"name\":\"John\",\"age\":30}".to_string()
                );
                assert_eq!(
                    json_result.usage,
                    Usage {
                        input_tokens: 15,
                        output_tokens: 15,
                    }
                );
                assert_eq!(json_result.model_inference_results.len(), 1);
                let model_inference_result = json_result.model_inference_results.first().unwrap();
                assert_eq!(model_inference_result.model_name, model_name);
                assert_eq!(
                    model_inference_result.model_provider_name,
                    model_provider_name
                );
                assert_eq!(model_inference_result.raw_request, raw_request);
            }
            _ => panic!("Expected Json inference response"),
        }
        // Test Case 7: a JSON string with a dynamic schema that passes validation and also include usage in each chunk
        let inference_id = Uuid::now_v7();
        let static_output_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"]
        });
        let implicit_tool_call_config = ToolCallConfig::implicit_from_value(&static_output_schema);
        let output_schema = JSONSchemaFromPath::from_value(&static_output_schema).unwrap();
        let function_config = FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            implicit_tool_call_config,
            output_schema,
        });
        let usage1 = Usage {
            input_tokens: 10,
            output_tokens: 5,
        };
        let usage2 = Usage {
            input_tokens: 5,
            output_tokens: 10,
        };
        let dynamic_output_schema = DynamicJSONSchema::new(serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        }));
        let inference_config = InferenceConfig {
            tool_config: None,
            function_name: "".to_string(),
            variant_name: "".to_string(),
            templates: &TemplateConfig::default(),
            dynamic_output_schema: Some(dynamic_output_schema),
        };
        let chunks = vec![
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                inference_id,
                raw: "{\"name\":".to_string(),
                created,
                usage: Some(usage1.clone()),
                raw_response: "{\"name\":".to_string(),
                latency: Duration::from_millis(150),
            }),
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                inference_id,
                raw: "\"John\",\"age\":30}".to_string(),
                created,
                usage: Some(usage2.clone()),
                raw_response: "\"John\",\"age\":30}".to_string(),
                latency: Duration::from_millis(250),
            }),
        ];
        let response = collect_chunks(
            chunks,
            &function_config,
            &inference_config,
            model_name,
            model_provider_name,
            raw_request.clone(),
            InferenceParams::default(),
        )
        .await
        .unwrap();
        match response {
            InferenceResult::Json(json_result) => {
                assert_eq!(json_result.inference_id, inference_id);
                assert_eq!(
                    json_result.output.parsed,
                    Some(serde_json::json!({"name": "John", "age": 30}))
                );
                assert_eq!(
                    json_result.output.raw,
                    "{\"name\":\"John\",\"age\":30}".to_string()
                );
                assert_eq!(
                    json_result.usage,
                    Usage {
                        input_tokens: 15,
                        output_tokens: 15,
                    }
                );
                assert_eq!(json_result.model_inference_results.len(), 1);
                let model_inference_result = json_result.model_inference_results.first().unwrap();
                assert_eq!(model_inference_result.model_name, model_name);
                assert_eq!(
                    model_inference_result.model_provider_name,
                    model_provider_name
                );
                assert_eq!(model_inference_result.raw_request, raw_request);
            }
            _ => panic!("Expected Json inference response"),
        }
    }

    #[test]
    fn test_deserialize_input_message() {
        // Test case for single string content
        let input = json!({
            "role": "user",
            "content": "Hello, world!"
        });
        let message: InputMessage = serde_json::from_value(input).unwrap();
        assert_eq!(message.role, Role::User);
        assert_eq!(message.content.len(), 1);
        match &message.content[0] {
            InputMessageContent::Text { value } => assert_eq!(value, "Hello, world!"),
            _ => panic!("Expected Text content"),
        }

        // Test case for object content
        let input = json!({
            "role": "assistant",
            "content": {"key": "value"}
        });
        let message: InputMessage = serde_json::from_value(input).unwrap();
        assert_eq!(message.role, Role::Assistant);
        assert_eq!(message.content.len(), 1);
        match &message.content[0] {
            InputMessageContent::Text { value } => {
                assert_eq!(value, &json!({"key": "value"}))
            }
            _ => panic!("Expected Text content"),
        }

        // Test case for multiple content items
        let input = json!({
            "role": "user",
            "content": [
                {"type": "text", "value": "Hello"},
                {"type": "tool_call", "id": "123", "name": "test_tool", "arguments": "{}"}
            ]
        });
        let message: InputMessage = serde_json::from_value(input).unwrap();
        assert_eq!(message.role, Role::User);
        assert_eq!(message.content.len(), 2);
        match &message.content[0] {
            InputMessageContent::Text { value } => assert_eq!(value, "Hello"),
            _ => panic!("Expected Text content"),
        }
        match &message.content[1] {
            InputMessageContent::ToolCall(tool_call) => {
                assert_eq!(tool_call.id, "123");
                assert_eq!(tool_call.name, "test_tool");
                assert_eq!(tool_call.arguments, "{}");
            }
            _ => panic!("Expected ToolCall content"),
        }
        // Test case for multiple content items with JSON object in text block
        let input = json!({
            "role": "user",
            "content": [
                {"type": "text", "value": {"complex": "json", "with": ["nested", "array"]}},
                {"type": "tool_call", "id": "456", "name": "another_tool", "arguments": "{\"key\": \"value\"}"}
            ]
        });
        let message: InputMessage = serde_json::from_value(input).unwrap();
        assert_eq!(message.role, Role::User);
        assert_eq!(message.content.len(), 2);
        match &message.content[0] {
            InputMessageContent::Text { value } => {
                assert_eq!(
                    value,
                    &json!({"complex": "json", "with": ["nested", "array"]})
                )
            }
            _ => panic!("Expected Text content with JSON object"),
        }
        match &message.content[1] {
            InputMessageContent::ToolCall(tool_call) => {
                assert_eq!(tool_call.id, "456");
                assert_eq!(tool_call.name, "another_tool");
                assert_eq!(tool_call.arguments, "{\"key\": \"value\"}");
            }
            _ => panic!("Expected ToolCall content"),
        }

        // Test case for invalid role
        let input = json!({
            "role": "invalid_role",
            "content": "Hello"
        });
        assert!(serde_json::from_value::<InputMessage>(input).is_err());

        // Test case for missing role
        let input = json!({
            "content": "Hello"
        });
        assert!(serde_json::from_value::<InputMessage>(input).is_err());

        // Test case for missing content
        let input = json!({
            "role": "user"
        });
        assert!(serde_json::from_value::<InputMessage>(input).is_err());

        // Test case for empty content array
        let input = json!({
            "role": "user",
            "content": []
        });
        let message: InputMessage = serde_json::from_value(input).unwrap();
        assert_eq!(message.role, Role::User);
        assert_eq!(message.content.len(), 0);

        // Test case for invalid content type
        let input = json!({
            "role": "user",
            "content": [{"type": "invalid_type", "value": "test"}]
        });
        assert!(serde_json::from_value::<InputMessage>(input).is_err());
    }
}
