use crate::inference::types::batch::deserialize_json_string;
use crate::inference::types::batch::deserialize_optional_json_string;
use derive_builder::Builder;
use extra_body::FullExtraBodyConfig;
use extra_body::UnfilteredInferenceExtraBody;
use futures::stream::Peekable;
use futures::Stream;
use image::sanitize_raw_request;
pub use image::{Base64Image, Image, ImageKind};
use itertools::Itertools;
use resolved_input::ImageWithPath;
pub use resolved_input::{ResolvedInput, ResolvedInputMessage, ResolvedInputMessageContent};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};
use serde_untagged::UntaggedEnumVisitor;
use std::{
    borrow::Cow,
    collections::HashMap,
    fmt::{self},
    pin::Pin,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use uuid::Uuid;

use crate::cache::NonStreamingCacheData;
use crate::{cache::CacheData, config_parser::ObjectStoreInfo};
use crate::{endpoints::inference::InferenceParams, error::ErrorDetails};
use crate::{
    endpoints::inference::{InferenceDatabaseInsertMetadata, InferenceIds},
    variant::InferenceConfig,
};
use crate::{error::Error, variant::JsonMode};
use crate::{function::FunctionConfig, minijinja_util::TemplateConfig};
use crate::{
    function::FunctionConfigType,
    tool::{ToolCall, ToolCallChunk, ToolCallConfig, ToolCallOutput, ToolResult},
};
use crate::{jsonschema_util::DynamicJSONSchema, tool::ToolCallConfigDatabaseInsert};

pub mod batch;
pub mod extra_body;
pub mod image;
pub mod resolved_input;
pub mod storage;

/*
 * Data flow in TensorZero
 *
 * The flow of an inference request through TensorZero can be viewed as a series of transformations between types.
 * Most of them are defined below.
 */

/// A request is made that contains an Input
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct Input {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Value>,
    #[serde(default)]
    pub messages: Vec<InputMessage>,
}

pub struct FetchContext<'a> {
    pub client: &'a reqwest::Client,
    pub object_store_info: &'a Option<ObjectStoreInfo>,
}

impl Input {
    /// Resolves any nested network resources in the input.
    /// Currently, this resolves input image urls into base64-encoded images.
    pub async fn resolve(self, context: &FetchContext<'_>) -> Result<ResolvedInput, Error> {
        let messages = futures::future::try_join_all(
            self.messages
                .into_iter()
                .map(|message| message.resolve(context)),
        )
        .await?;
        Ok(ResolvedInput {
            system: self.system,
            messages,
        })
    }
}

impl InputMessage {
    pub async fn resolve(self, context: &FetchContext<'_>) -> Result<ResolvedInputMessage, Error> {
        let content = futures::future::try_join_all(
            self.content
                .into_iter()
                .map(|content| content.resolve(context)),
        )
        .await?;
        Ok(ResolvedInputMessage {
            role: self.role,
            content,
        })
    }
}

impl InputMessageContent {
    pub async fn resolve(
        self,
        context: &FetchContext<'_>,
    ) -> Result<ResolvedInputMessageContent, Error> {
        Ok(match self {
            InputMessageContent::Text(TextKind::Text { text }) => {
                ResolvedInputMessageContent::Text {
                    value: Value::String(text),
                }
            }
            InputMessageContent::Text(TextKind::Arguments { arguments }) => {
                ResolvedInputMessageContent::Text {
                    value: Value::Object(arguments),
                }
            }
            InputMessageContent::ToolCall(tool_call) => {
                ResolvedInputMessageContent::ToolCall(tool_call)
            }
            InputMessageContent::ToolResult(tool_result) => {
                ResolvedInputMessageContent::ToolResult(tool_result)
            }
            InputMessageContent::RawText { value } => {
                ResolvedInputMessageContent::RawText { value }
            }
            InputMessageContent::Thought(thought) => ResolvedInputMessageContent::Thought(thought),
            InputMessageContent::Text(TextKind::LegacyValue { value }) => {
                tracing::warn!(
                    r#"Deprecation warning: `{{"type": "text", "value", ...}}` is deprecated. Please use `{{"type": "text", "text": "String input"}}` or `{{"type": "text", "arguments": {{..}}}} ` instead."#
                );
                ResolvedInputMessageContent::Text { value }
            }
            InputMessageContent::Image(image) => {
                let storage_kind = context
                    .object_store_info
                    .as_ref()
                    .ok_or_else(|| {
                        Error::new(ErrorDetails::ObjectStoreUnconfigured {
                            block_type: "image".to_string(),
                        })
                    })?
                    .kind
                    .clone();
                let image = image.take_or_fetch(context.client).await?;
                let path = storage_kind.image_path(&image)?;
                ResolvedInputMessageContent::Image(ImageWithPath {
                    image,
                    storage_path: path,
                })
            }
            InputMessageContent::Unknown {
                data,
                model_provider_name,
            } => ResolvedInputMessageContent::Unknown {
                data,
                model_provider_name,
            },
        })
    }
}

/// InputMessage and Role are our representation of the input sent by the client
/// prior to any processing into LLM representations below.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct InputMessage {
    pub role: Role,
    #[serde(deserialize_with = "deserialize_content")]
    pub content: Vec<InputMessageContent>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputMessageContent {
    Text(TextKind),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    RawText {
        value: String,
    },
    Thought(Thought),
    Image(Image),
    /// An unknown content block type, used to allow passing provider-specific
    /// content blocks (e.g. Anthropic's "redacted_thinking") in and out
    /// of TensorZero.
    /// The 'data' field hold the original content block from the provider,
    /// without any validation or transformation by TensorZero.
    Unknown {
        data: Value,
        model_provider_name: Option<String>,
    },
    // We may extend this in the future to include other types of content
}

#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(untagged, deny_unknown_fields)]
pub enum TextKind {
    Text { text: String },
    Arguments { arguments: Map<String, Value> },
    LegacyValue { value: Value },
}

impl<'de> Deserialize<'de> for TextKind {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let object: Map<String, Value> = Map::deserialize(de)?;
        // Expect exactly one key
        if object.keys().len() != 1 {
            return Err(serde::de::Error::custom(format!(
                "Expected exactly one other key in text content, found {} other keys",
                object.keys().len()
            )));
        }
        let (key, value) = object.into_iter().next().ok_or_else(|| {
            serde::de::Error::custom(
                "Internal error: Failed to get key/value after checking length",
            )
        })?;
        match key.as_str() {
            "text" => Ok(TextKind::Text {
                text: serde_json::from_value(value).map_err(|e| {
                    serde::de::Error::custom(format!("Error deserializing 'text': {e}"))
                })?,
            }),
            "arguments" => Ok(TextKind::Arguments {
                arguments: serde_json::from_value(value).map_err(|e| {
                    serde::de::Error::custom(format!("Error deserializing 'arguments': {e}"))
                })?,
            }),
            "value" => Ok(TextKind::LegacyValue { value }),
            _ => Err(serde::de::Error::custom(format!(
                "Unknown key '{}' in text content",
                key
            ))),
        }
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

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Text {
    pub text: String,
}

/// Struct that represents Chain of Thought reasoning
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Thought {
    pub text: String,
    /// An optional signature - currently, this is only used with Anthropic,
    /// and is ignored by other providers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

/// Core representation of the types of content that could go into a model provider
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text(Text),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    Image(ImageWithPath),
    Thought(Thought),
    /// Represents an unknown provider-specific content block.
    /// We pass this along as-is without any validation or transformation.
    Unknown {
        /// The underlying content block to be passed to the model provider.
        data: Value,
        /// A fully-qualified name specifying when this content block should
        /// be included in the model provider input.
        /// E.g `tensorzero::model_name::claude-3-7-sonnet-20250219-thinking::provider_name::anthropic-extra-body`
        ///
        /// If set to `Some`, this is compared against the output of `fully_qualified_name` before invoking
        /// a model provider, and stripped from the input if it doesn't match.
        /// If set to `None, then this is passed to all model providers.
        /// Individual model provider implementation never need to check this field themselves -
        /// they only need to produce it with the proper `fully_qualified_name` set.
        model_provider_name: Option<String>,
    },
}

/// A helper type for dealing with `ContentBlock::Unknown` in model providers.
/// This flattens the wrapped `Value` when serializing and deserializing.
///
/// During deserialization, we'll first attempt to deserialize a `T`
/// (e.g. `AnthropicContentBlock`), and fall back to `Unknown` with the raw
/// json `Value` if that fails.
///
/// During serialization, a `FlattenUnknown::Unknown` will have the wrapped
/// `Value` serialized, allowing us to send an arbitrary json value to
/// a provider.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum FlattenUnknown<'a, T> {
    Normal(T),
    Unknown(Cow<'a, Value>),
}

/// Defines the types of content block that can come out of a model provider
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockOutput {
    Text(Text),
    ToolCall(ToolCall),
    Thought(Thought),
    Unknown {
        data: Value,
        model_provider_name: Option<String>,
    },
}

/// Defines the types of content block that can come from a `chat` function
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockChatOutput {
    Text(Text),
    ToolCall(ToolCallOutput),
    Thought(Thought),
    Unknown {
        data: Value,
        model_provider_name: Option<String>,
    },
}

/// A RequestMessage is a message sent to a model
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
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

#[derive(Clone, Copy, Default, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
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
#[derive(Builder, Clone, Debug, Default, PartialEq, Serialize)]
#[builder(setter(into, strip_option), default)]
pub struct ModelInferenceRequest<'a> {
    pub inference_id: Uuid,
    pub messages: Vec<RequestMessage>,
    pub system: Option<String>,
    pub tool_config: Option<Cow<'a, ToolCallConfig>>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<u32>,
    pub stream: bool,
    pub json_mode: ModelInferenceRequestJsonMode,
    pub function_type: FunctionType,
    pub output_schema: Option<&'a Value>,
    pub extra_body: FullExtraBodyConfig,
    /// Optional arbitrary data, only used when constructing the cache key.
    /// This is used by best_of_n/mixture_of_n to force different sub-variants
    /// to have different cache keys.
    pub extra_cache_key: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCall,
    ContentFilter,
    Unknown,
}

/// Each provider transforms a ModelInferenceRequest into a provider-specific (private) inference request type
/// that is suitable for serialization directly into a request to the provider.
///
/// In both non-streaming and streaming inference, each ModelProvider receives data from the provider in a
/// a (private) provider-specific format that is then transformed into a ProviderInferenceResponse (non-streaming)
/// or a stream of ProviderInferenceResponseChunks (streaming).

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ProviderInferenceResponse {
    pub id: Uuid,
    pub created: u64,
    pub output: Vec<ContentBlockOutput>,
    pub system: Option<String>,
    pub input_messages: Vec<RequestMessage>,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
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
    Batch,
}

/// After a ProviderInferenceResponse is returned to the Model,
/// it is converted into a ModelInferenceResponse that includes additional metadata (such as the model provider name).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelInferenceResponse {
    pub id: Uuid,
    pub created: u64,
    pub output: Vec<ContentBlockOutput>,
    pub system: Option<String>,
    pub input_messages: Vec<RequestMessage>,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
    pub model_provider_name: Arc<str>,
    pub cached: bool,
    pub finish_reason: Option<FinishReason>,
}

/// Finally, in the Variant we convert the ModelInferenceResponse into a ModelInferenceResponseWithMetadata
/// that includes additional metadata (such as the model name).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelInferenceResponseWithMetadata {
    pub id: Uuid,
    pub created: u64,
    pub output: Vec<ContentBlockOutput>,
    pub system: Option<String>,
    pub input_messages: Vec<RequestMessage>,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
    pub model_provider_name: Arc<str>,
    pub model_name: Arc<str>,
    pub cached: bool,
    pub finish_reason: Option<FinishReason>,
}

impl ModelInferenceResponseWithMetadata {
    /// We return the actual usage (meaning the number of tokens the user would be billed for)
    /// in the HTTP response.
    /// However, we store the number of tokens that would have been used in the database.
    /// So we need this function to compute the actual usage in order to send it in the HTTP response.
    pub fn actual_usage(&self) -> Usage {
        if self.cached {
            Usage {
                input_tokens: 0,
                output_tokens: 0,
            }
        } else {
            self.usage.clone()
        }
    }
}

/* As a Variant might make use of multiple model inferences, we then combine
 * one or more ModelInferenceResults into a single InferenceResult (but we keep the original ModelInferenceResults around for storage).
 * In the non-streaming case, this InferenceResult is converted into an InferenceResponse and sent to the client.
 * See below for streaming case.
 */

/// This type contains the result of running a variant of a function
#[derive(Clone, Debug)]
//#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceResult {
    Chat(ChatInferenceResult),
    Json(JsonInferenceResult),
}

#[derive(Clone, Debug)]
pub struct ChatInferenceResult {
    pub inference_id: Uuid,
    #[allow(dead_code)]
    created: u64,
    pub content: Vec<ContentBlockChatOutput>,
    pub usage: Usage,
    pub model_inference_results: Vec<ModelInferenceResponseWithMetadata>,
    pub inference_params: InferenceParams,
    pub original_response: Option<String>,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Clone, Debug)]
pub struct JsonInferenceResult {
    pub inference_id: Uuid,
    #[allow(dead_code)]
    created: u64,
    pub output: JsonInferenceOutput,
    pub usage: Usage,
    pub model_inference_results: Vec<ModelInferenceResponseWithMetadata>,
    pub output_schema: Value,
    pub inference_params: InferenceParams,
    pub original_response: Option<String>,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct JsonInferenceOutput {
    pub raw: String,
    pub parsed: Option<Value>,
}

/// In the streaming case we convert ProviderInferenceResponseChunks into a InferenceResultChunk, which is then
/// converted into an InferenceResponseChunk and sent to the client.
/// We then collect all the InferenceResultChunks into an InferenceResult for validation and storage after the fact.

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ProviderInferenceResponseChunk {
    pub content: Vec<ContentBlockChunk>,
    pub created: u64,
    pub usage: Option<Usage>,
    pub raw_response: String,
    pub latency: Duration,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockChunk {
    Text(TextChunk),
    ToolCall(ToolCallChunk),
    Thought(ThoughtChunk),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextChunk {
    pub id: String,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThoughtChunk {
    pub id: String,
    pub text: Option<String>,
    pub signature: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatInferenceResultChunk {
    pub content: Vec<ContentBlockChunk>,
    pub created: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    pub latency: Duration,
    pub raw_response: String,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct JsonInferenceResultChunk {
    pub raw: Option<String>,
    pub thought: Option<String>,
    pub created: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    pub latency: Duration,
    pub raw_response: String,
    pub finish_reason: Option<FinishReason>,
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

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatInferenceDatabaseInsert {
    pub id: Uuid,
    pub function_name: String,
    pub variant_name: String,
    pub episode_id: Uuid,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub input: ResolvedInput,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub output: Vec<ContentBlockChatOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(deserialize_with = "deserialize_optional_json_string")]
    pub tool_params: Option<ToolCallConfigDatabaseInsert>,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub inference_params: InferenceParams,
    pub processing_time_ms: Option<u32>,
    pub tags: HashMap<String, String>,
    #[serde(default)]
    pub extra_body: UnfilteredInferenceExtraBody,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct JsonInferenceDatabaseInsert {
    pub id: Uuid,
    pub function_name: String,
    pub variant_name: String,
    pub episode_id: Uuid,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub input: ResolvedInput,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub output: JsonInferenceOutput,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub inference_params: InferenceParams,
    pub processing_time_ms: Option<u32>,
    pub output_schema: Value,
    pub tags: HashMap<String, String>,
    #[serde(default)]
    pub extra_body: UnfilteredInferenceExtraBody,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum InferenceDatabaseInsert {
    Chat(ChatInferenceDatabaseInsert),
    Json(JsonInferenceDatabaseInsert),
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelInferenceDatabaseInsert {
    pub id: Uuid,
    pub inference_id: Uuid,
    pub raw_request: String,
    pub raw_response: String,
    pub system: Option<String>,
    pub input_messages: String,
    pub output: String,
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    pub response_time_ms: Option<u32>,
    pub model_name: String,
    pub model_provider_name: String,
    pub ttft_ms: Option<u32>,
    pub cached: bool,
    pub finish_reason: Option<FinishReason>,
}

#[cfg(test)]
impl From<String> for InputMessageContent {
    fn from(text: String) -> Self {
        InputMessageContent::Text(TextKind::Text { text })
    }
}

#[cfg(test)]
impl From<String> for ResolvedInputMessageContent {
    fn from(text: String) -> Self {
        ResolvedInputMessageContent::Text {
            value: Value::String(text),
        }
    }
}

#[cfg(any(test, feature = "e2e_tests"))]
impl From<String> for ContentBlockChatOutput {
    fn from(text: String) -> Self {
        ContentBlockChatOutput::Text(Text { text })
    }
}

impl From<Value> for ResolvedInputMessageContent {
    fn from(value: Value) -> Self {
        ResolvedInputMessageContent::Text { value }
    }
}

fn deserialize_content<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<InputMessageContent>, D::Error> {
    UntaggedEnumVisitor::new()
        .string(|text| {
            Ok(vec![InputMessageContent::Text(TextKind::Text {
                text: text.to_string(),
            })])
        })
        .map(|object| {
            tracing::warn!("Deprecation warning - passing in an object for `content` is deprecated. Please use an array of content blocks instead.");
            Ok(vec![InputMessageContent::Text(TextKind::Arguments {
                arguments: object.deserialize()?,
            })])
        })
        .seq(|seq| seq.deserialize())
        .deserialize(deserializer)
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

impl From<String> for ContentBlockOutput {
    fn from(text: String) -> Self {
        ContentBlockOutput::Text(Text { text })
    }
}

impl ModelInferenceResponse {
    pub fn new(
        provider_inference_response: ProviderInferenceResponse,
        model_provider_name: Arc<str>,
        cached: bool,
    ) -> Self {
        Self {
            id: provider_inference_response.id,
            created: provider_inference_response.created,
            output: provider_inference_response.output,
            system: provider_inference_response.system,
            input_messages: provider_inference_response.input_messages,
            raw_request: provider_inference_response.raw_request,
            raw_response: provider_inference_response.raw_response,
            usage: provider_inference_response.usage,
            latency: provider_inference_response.latency,
            finish_reason: provider_inference_response.finish_reason,
            model_provider_name,
            cached,
        }
    }

    pub fn from_cache(
        cache_lookup: CacheData<NonStreamingCacheData>,
        request: &ModelInferenceRequest<'_>,
        model_provider_name: &str,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            created: current_timestamp(),
            output: cache_lookup.output.blocks,
            system: request.system.clone(),
            input_messages: request.messages.clone(), // maybe we can clean this up
            raw_request: cache_lookup.raw_request,
            raw_response: cache_lookup.raw_response,
            usage: Usage {
                input_tokens: cache_lookup.input_tokens,
                output_tokens: cache_lookup.output_tokens,
            },
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(0),
            },
            finish_reason: cache_lookup.finish_reason,
            model_provider_name: Arc::from(model_provider_name),
            cached: true,
        }
    }
}

impl ModelInferenceResponseWithMetadata {
    pub fn new(model_inference_response: ModelInferenceResponse, model_name: Arc<str>) -> Self {
        Self {
            id: model_inference_response.id,
            created: model_inference_response.created,
            output: model_inference_response.output,
            system: model_inference_response.system,
            input_messages: model_inference_response.input_messages,
            raw_request: model_inference_response.raw_request,
            raw_response: model_inference_response.raw_response,
            usage: model_inference_response.usage,
            latency: model_inference_response.latency,
            finish_reason: model_inference_response.finish_reason,
            model_provider_name: model_inference_response.model_provider_name,
            model_name,
            cached: model_inference_response.cached,
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
                Some(response_time.as_millis() as u32),
                Some(ttft.as_millis() as u32),
            ),
            Latency::NonStreaming { response_time } => {
                (Some(response_time.as_millis() as u32), None)
            }
            Latency::Batch => (None, None),
        };
        let serialized_input_messages = serialize_or_log(&result.input_messages);
        let serialized_output = serialize_or_log(&result.output);

        // A usage of 0 indicates that something went wrong, since a model
        // should always consume and produce at least one token.
        // We store this as `null` in ClickHouse, so that we can easily filter
        // out these values from aggregation queries.
        let input_tokens = if result.usage.input_tokens > 0 {
            Some(result.usage.input_tokens)
        } else {
            None
        };
        let output_tokens = if result.usage.output_tokens > 0 {
            Some(result.usage.output_tokens)
        } else {
            None
        };

        Self {
            id: Uuid::now_v7(),
            inference_id,
            raw_request: result.raw_request,
            raw_response: result.raw_response,
            system: result.system,
            input_messages: serialized_input_messages,
            output: serialized_output,
            input_tokens,
            output_tokens,
            response_time_ms: latency_ms,
            ttft_ms,
            model_provider_name: result.model_provider_name.to_string(),
            model_name: result.model_name.to_string(),
            cached: result.cached,
            finish_reason: result.finish_reason,
        }
    }
}

pub struct ProviderInferenceResponseArgs {
    pub output: Vec<ContentBlockOutput>,
    pub system: Option<String>,
    pub input_messages: Vec<RequestMessage>,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
    pub finish_reason: Option<FinishReason>,
}

impl ProviderInferenceResponse {
    pub fn new(args: ProviderInferenceResponseArgs) -> Self {
        let sanitized_raw_request = sanitize_raw_request(&args.input_messages, args.raw_request);
        Self {
            id: Uuid::now_v7(),
            created: current_timestamp(),
            output: args.output,
            system: args.system,
            input_messages: args.input_messages,
            raw_request: sanitized_raw_request,
            raw_response: args.raw_response,
            usage: args.usage,
            latency: args.latency,
            finish_reason: args.finish_reason,
        }
    }
}

impl InferenceResult {
    pub fn model_inference_results(&self) -> &Vec<ModelInferenceResponseWithMetadata> {
        match self {
            InferenceResult::Chat(chat_result) => &chat_result.model_inference_results,
            InferenceResult::Json(json_result) => &json_result.model_inference_results,
        }
    }

    pub fn get_serialized_model_inferences(&self) -> Vec<serde_json::Value> {
        let model_inference_responses = self.model_inference_results();
        let inference_id = match self {
            InferenceResult::Chat(chat_result) => chat_result.inference_id,
            InferenceResult::Json(json_result) => json_result.inference_id,
        };
        model_inference_responses
            .iter()
            .map(|r| {
                let model_inference = ModelInferenceDatabaseInsert::new(r.clone(), inference_id);
                match serde_json::to_value(model_inference) {
                    Ok(v) => v,
                    Err(e) => {
                        ErrorDetails::Serialization {
                            message: format!(
                                "Failed to serialize ModelInferenceDatabaseInsert: {e:?}"
                            ),
                        }
                        .log();
                        Default::default()
                    }
                }
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

    pub fn set_original_response(&mut self, original_response: Option<String>) {
        match self {
            InferenceResult::Chat(chat_result) => chat_result.original_response = original_response,
            InferenceResult::Json(json_result) => json_result.original_response = original_response,
        }
    }

    pub fn mut_model_inference_results(&mut self) -> &mut Vec<ModelInferenceResponseWithMetadata> {
        match self {
            InferenceResult::Chat(chat_result) => &mut chat_result.model_inference_results,
            InferenceResult::Json(json_result) => &mut json_result.model_inference_results,
        }
    }

    pub fn owned_model_inference_results(self) -> Vec<ModelInferenceResponseWithMetadata> {
        match self {
            InferenceResult::Chat(chat_result) => chat_result.model_inference_results,
            InferenceResult::Json(json_result) => json_result.model_inference_results,
        }
    }
}

impl JsonInferenceResult {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        inference_id: Uuid,
        raw: String,
        parsed: Option<Value>,
        usage: Usage,
        model_inference_results: Vec<ModelInferenceResponseWithMetadata>,
        output_schema: Value,
        inference_params: InferenceParams,
        original_response: Option<String>,
    ) -> Self {
        let output = JsonInferenceOutput { raw, parsed };
        let finish_reason = get_finish_reason(&model_inference_results);
        Self {
            inference_id,
            created: current_timestamp(),
            output,
            usage,
            model_inference_results,
            output_schema,
            inference_params,
            original_response,
            finish_reason,
        }
    }
}

impl ChatInferenceResult {
    pub async fn new(
        inference_id: Uuid,
        raw_content: Vec<ContentBlockOutput>,
        usage: Usage,
        model_inference_results: Vec<ModelInferenceResponseWithMetadata>,
        tool_config: Option<&ToolCallConfig>,
        inference_params: InferenceParams,
        original_response: Option<String>,
    ) -> Self {
        let created = current_timestamp();
        let content = parse_chat_output(raw_content, tool_config).await;
        let finish_reason = get_finish_reason(&model_inference_results);
        Self {
            inference_id,
            created,
            content,
            usage,
            model_inference_results,
            inference_params,
            original_response,
            finish_reason,
        }
    }
}

/// Get the finish reason from the last model inference result sorted by created time (or None if it is not present)
fn get_finish_reason(
    model_inference_results: &[ModelInferenceResponseWithMetadata],
) -> Option<FinishReason> {
    model_inference_results
        .iter()
        .sorted_by_key(|r| r.created)
        .next_back()
        .and_then(|r| r.finish_reason.clone())
}

pub async fn parse_chat_output(
    content: Vec<ContentBlockOutput>,
    tool_config: Option<&ToolCallConfig>,
) -> Vec<ContentBlockChatOutput> {
    if content.is_empty() {
        Error::new(ErrorDetails::Inference {
            message: "No content blocks in inference result".to_string(),
        });
    }

    let mut output = Vec::new();
    for content in content.into_iter() {
        match content {
            ContentBlockOutput::Text(text) => {
                output.push(ContentBlockChatOutput::Text(text));
            }
            ContentBlockOutput::ToolCall(tool_call) => {
                // Parse the tool call arguments
                let tool_call_output = ToolCallOutput::new(tool_call, tool_config).await;
                output.push(ContentBlockChatOutput::ToolCall(tool_call_output));
            }
            ContentBlockOutput::Thought(thought) => {
                output.push(ContentBlockChatOutput::Thought(thought));
            }
            ContentBlockOutput::Unknown {
                data,
                model_provider_name,
            } => {
                output.push(ContentBlockChatOutput::Unknown {
                    data,
                    model_provider_name,
                });
            }
        }
    }
    output
}

impl ChatInferenceDatabaseInsert {
    pub fn new(
        chat_result: ChatInferenceResult,
        input: ResolvedInput,
        metadata: InferenceDatabaseInsertMetadata,
    ) -> Self {
        let processing_time_ms = metadata
            .processing_time
            .map(|duration| duration.as_millis() as u32);

        let tool_params = metadata.tool_config.map(ToolCallConfigDatabaseInsert::from);
        let inference_params = chat_result.inference_params;

        Self {
            id: chat_result.inference_id,
            function_name: metadata.function_name,
            variant_name: metadata.variant_name,
            episode_id: metadata.episode_id,
            input,
            tool_params,
            inference_params,
            output: chat_result.content,
            processing_time_ms,
            tags: metadata.tags,
            extra_body: metadata.extra_body,
        }
    }
}

impl JsonInferenceDatabaseInsert {
    pub fn new(
        json_result: JsonInferenceResult,
        input: ResolvedInput,
        metadata: InferenceDatabaseInsertMetadata,
    ) -> Self {
        let processing_time_ms = metadata
            .processing_time
            .map(|duration| duration.as_millis() as u32);

        let inference_params = json_result.inference_params;

        Self {
            id: json_result.inference_id,
            function_name: metadata.function_name,
            variant_name: metadata.variant_name,
            episode_id: metadata.episode_id,
            input,
            inference_params,
            output: json_result.output,
            processing_time_ms,
            output_schema: json_result.output_schema,
            tags: metadata.tags,
            extra_body: metadata.extra_body,
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
        content: Vec<ContentBlockChunk>,
        usage: Option<Usage>,
        raw_response: String,
        latency: Duration,
        finish_reason: Option<FinishReason>,
    ) -> Self {
        Self {
            content,
            created: current_timestamp(),
            usage,
            raw_response,
            latency,
            finish_reason,
        }
    }
}

impl InferenceResultChunk {
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

    pub fn finish_reason(&self) -> Option<&FinishReason> {
        match self {
            InferenceResultChunk::Chat(chunk) => chunk.finish_reason.as_ref(),
            InferenceResultChunk::Json(chunk) => chunk.finish_reason.as_ref(),
        }
    }
}

impl InferenceResultChunk {
    pub fn new(chunk: ProviderInferenceResponseChunk, function: FunctionConfigType) -> Self {
        match function {
            FunctionConfigType::Chat => Self::Chat(chunk.into()),
            FunctionConfigType::Json => Self::Json(chunk.into()),
        }
    }
}

impl From<ProviderInferenceResponseChunk> for ChatInferenceResultChunk {
    fn from(chunk: ProviderInferenceResponseChunk) -> Self {
        Self {
            content: chunk.content,
            created: chunk.created,
            usage: chunk.usage,
            latency: chunk.latency,
            finish_reason: chunk.finish_reason,
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

impl From<ContentBlockChatOutput> for ContentBlock {
    fn from(output: ContentBlockChatOutput) -> Self {
        match output {
            ContentBlockChatOutput::Text(text) => ContentBlock::Text(text),
            ContentBlockChatOutput::ToolCall(tool_call_output) => {
                ContentBlock::ToolCall(tool_call_output.into())
            }
            ContentBlockChatOutput::Thought(thought) => ContentBlock::Thought(thought),
            ContentBlockChatOutput::Unknown {
                data,
                model_provider_name,
            } => ContentBlock::Unknown {
                data,
                model_provider_name,
            },
        }
    }
}

/// We use best-effort to reconstruct the raw response for JSON functions
/// They might either return a ToolCallChunk or a TextChunk
/// We take the string from either of these (from the last block if there are multiple)
/// and use that as the raw response.
impl From<ProviderInferenceResponseChunk> for JsonInferenceResultChunk {
    fn from(chunk: ProviderInferenceResponseChunk) -> Self {
        let mut raw = None;
        let mut thought = None;
        for content in chunk.content.into_iter() {
            match content {
                ContentBlockChunk::ToolCall(tool_call) => {
                    raw = Some(tool_call.raw_arguments.to_owned())
                }
                ContentBlockChunk::Text(text_chunk) => raw = Some(text_chunk.text.to_owned()),
                ContentBlockChunk::Thought(thought_chunk) => {
                    thought = thought_chunk.text;
                }
            }
        }
        Self {
            raw,
            thought,
            created: chunk.created,
            usage: chunk.usage,
            latency: chunk.latency,
            raw_response: chunk.raw_response,
            finish_reason: chunk.finish_reason,
        }
    }
}

// Define the CollectChunksArgs struct with existing and new fields
pub struct CollectChunksArgs<'a, 'b> {
    pub value: Vec<InferenceResultChunk>,
    pub inference_id: Uuid,
    pub episode_id: Uuid,
    pub function: Arc<FunctionConfig>,
    pub model_name: Arc<str>,
    pub model_provider_name: Arc<str>,
    pub raw_request: String,
    pub inference_params: InferenceParams,
    pub system: Option<String>,
    pub input_messages: Vec<RequestMessage>,
    pub function_name: &'b str,
    pub variant_name: &'b str,
    pub dynamic_output_schema: Option<DynamicJSONSchema>,
    pub templates: &'a TemplateConfig<'a>,
    pub tool_config: Option<&'b ToolCallConfig>,
    pub cached: bool,
    pub extra_body: UnfilteredInferenceExtraBody,
}

// Modify the collect_chunks function to accept CollectChunksArgs
// 'a ends up as static and 'b ends up as stack allocated in the caller (endpoints::inference::create_stream)
pub async fn collect_chunks(args: CollectChunksArgs<'_, '_>) -> Result<InferenceResult, Error> {
    let CollectChunksArgs {
        value,
        inference_id,
        episode_id,
        function,
        model_name,
        model_provider_name,
        raw_request,
        inference_params,
        system,
        input_messages,
        function_name,
        variant_name,
        dynamic_output_schema,
        templates,
        tool_config,
        cached,
        extra_body,
    } = args;

    // NOTE: We will eventually need this to be per-inference-response-type and sensitive to the type of variant and function being called.
    let mut tool_call_blocks: HashMap<String, ContentBlockOutput> = HashMap::new();
    let mut text_blocks: HashMap<String, ContentBlockOutput> = HashMap::new();
    let mut thought_blocks: HashMap<String, ContentBlockOutput> = HashMap::new();
    let raw_response: String = value
        .iter()
        .map(|chunk| chunk.raw_response())
        .collect::<Vec<&str>>()
        .join("\n");
    let mut usage: Usage = Usage::default();
    let mut ttft: Option<Duration> = None;
    let response_time = value
        .last()
        .ok_or_else(|| {
            Error::new(ErrorDetails::TypeConversion {
                message:
                    "Attempted to create an InferenceResult from an empty response chunk vector"
                        .to_string(),
            })
        })?
        .latency();
    // We'll take the finish reason from the last chunk
    let mut finish_reason: Option<FinishReason> = None;
    for chunk in value {
        if let Some(chunk_usage) = chunk.usage() {
            usage.input_tokens = usage.input_tokens.saturating_add(chunk_usage.input_tokens);
            usage.output_tokens = usage
                .output_tokens
                .saturating_add(chunk_usage.output_tokens);
        }
        match chunk {
            InferenceResultChunk::Chat(chunk) => {
                if let Some(chunk_finish_reason) = chunk.finish_reason {
                    finish_reason = Some(chunk_finish_reason);
                }
                for content in chunk.content {
                    match content {
                        ContentBlockChunk::Text(text) => {
                            handle_textual_content_block(
                                &mut text_blocks,
                                text.id,
                                text.text,
                                &mut ttft,
                                chunk.latency,
                                |text| text.into(),
                                |block, text| {
                                    if let ContentBlockOutput::Text(Text {
                                        text: existing_text,
                                    }) = block
                                    {
                                        existing_text.push_str(text);
                                    }
                                },
                            );
                        }
                        ContentBlockChunk::Thought(thought) => {
                            // We check for both 'text' and 'signature', in case a provider produces
                            // both in the same chunk.
                            // These two cases update different fields ('text' vs 'signature') on the
                            // thought with id 'thought.id' - this is how providers attach a signature
                            // to a thought.
                            if let Some(text) = thought.text {
                                handle_textual_content_block(
                                    &mut thought_blocks,
                                    thought.id.clone(),
                                    text,
                                    &mut ttft,
                                    chunk.latency,
                                    |text| {
                                        ContentBlockOutput::Thought(Thought {
                                            text,
                                            signature: None,
                                        })
                                    },
                                    |block, text| {
                                        if let ContentBlockOutput::Thought(thought) = block {
                                            thought.text.push_str(text);
                                        }
                                    },
                                );
                            }
                            if let Some(signature) = thought.signature {
                                handle_textual_content_block(
                                    &mut thought_blocks,
                                    thought.id,
                                    signature,
                                    &mut ttft,
                                    chunk.latency,
                                    |signature| {
                                        ContentBlockOutput::Thought(Thought {
                                            text: String::new(),
                                            signature: Some(signature),
                                        })
                                    },
                                    |block, signature| {
                                        if let ContentBlockOutput::Thought(thought) = block {
                                            match &mut thought.signature {
                                                Some(existing) => existing.push_str(signature),
                                                None => {
                                                    thought.signature = Some(signature.to_string());
                                                }
                                            }
                                        }
                                    },
                                );
                            }
                        }
                        ContentBlockChunk::ToolCall(tool_call) => {
                            match tool_call_blocks.get_mut(&tool_call.id) {
                                // If there is already a tool call block with this id, append to it
                                Some(ContentBlockOutput::ToolCall(existing_tool_call)) => {
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
                                        ContentBlockOutput::ToolCall(tool_call.into()),
                                    );
                                }
                            }
                        }
                    }
                }
            }
            InferenceResultChunk::Json(chunk) => {
                if let Some(chunk_finish_reason) = chunk.finish_reason {
                    finish_reason = Some(chunk_finish_reason);
                }
                match text_blocks.get_mut("") {
                    // If there is already a text block, append to it
                    Some(ContentBlockOutput::Text(Text {
                        text: existing_text,
                    })) => {
                        if let Some(raw) = chunk.raw {
                            existing_text.push_str(&raw);
                        }
                    }
                    // If there is no text block, create one
                    _ => {
                        // We put this here and below rather than in the loop start because we
                        // only want to set TTFT if there is some real content
                        if ttft.is_none() {
                            ttft = Some(chunk.latency);
                        }
                        if let Some(raw) = chunk.raw {
                            text_blocks.insert(String::new(), raw.into());
                        }
                    }
                }
                if let Some(thought) = chunk.thought {
                    match thought_blocks.get_mut("") {
                        // If there is already a thought block, append to it
                        Some(ContentBlockOutput::Thought(existing_thought)) => {
                            existing_thought.text.push_str(&thought);
                        }
                        // If there is no thought block, create one
                        _ => {
                            thought_blocks.insert(
                                String::new(),
                                ContentBlockOutput::Thought(Thought {
                                    text: thought,
                                    signature: None,
                                }),
                            );
                        }
                    }
                }
            }
        }
    }
    let ttft = ttft.ok_or_else(|| {
        Error::new(ErrorDetails::TypeConversion {
            message: "Never got TTFT because there was never content in the response.".to_string(),
        })
    })?;
    let latency = Latency::Streaming {
        ttft,
        response_time,
    };
    let mut content_blocks: Vec<ContentBlockOutput> = tool_call_blocks.into_values().collect();
    content_blocks.extend(thought_blocks.into_values());
    content_blocks.extend(text_blocks.into_values());
    let model_response = ProviderInferenceResponse::new(ProviderInferenceResponseArgs {
        output: content_blocks.clone(),
        system,
        input_messages,
        raw_request,
        raw_response,
        usage: usage.clone(),
        latency: latency.clone(),
        finish_reason,
    });
    let model_inference_response =
        ModelInferenceResponse::new(model_response, model_provider_name, cached);
    let original_response = model_inference_response.raw_response.clone();
    let model_inference_result =
        ModelInferenceResponseWithMetadata::new(model_inference_response, model_name);
    let inference_config = InferenceConfig {
        ids: InferenceIds {
            inference_id,
            episode_id,
        },
        function_name,
        variant_name: Some(variant_name),
        tool_config,
        templates,
        dynamic_output_schema: dynamic_output_schema.as_ref(),
        extra_body,
        extra_cache_key: None,
    };
    function
        .prepare_response(
            inference_id,
            content_blocks,
            usage,
            vec![model_inference_result],
            &inference_config,
            inference_params,
            Some(original_response),
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

// We use a very specific combination of `Pin` and `Peekable` here, due to a combination of several requirements:
// * Inside of a model provider (e.g. anthropic), we may want to peek and modify the first chunk
//   to fix the start of a JSON response.
// * Outside of a model provider, we always want to peek at the first chunk to make sure that the HTTP request
//   actually succeeded.
// * The model providers produce distinct stream types (arising from different `async_stream` calls), so we
//   need to use a trait object.
//
// Combining all of these requirements, we need to wrap the entire `Pin<Box<dyn Stream>>` in `Peekable`.
// The `Peekable` type needs to be 'visible' (not erased inside the trait object), so that we can
// check the first chunk with 'peek()' outside of a model provider implementation. While we could have
// two `Peekable` types (one erased inside the trait object, one visible outside), this would add
// additional runtime overhead, and make things more difficult to reason about.
//
// We cannot write `Peekable<dyn Stream>`, since `Peekable` does not support the special unsized coercions that standard
// library types support (e.g. `Box<MyStreamType>` -> `Box<dyn Stream>`)'
// We also cannot write `Pin<Peekable>`, since the argument to `Pin` needs to implement `DerefMut`.
// This gives us the particular combination of types below.
//
// We split this into an 'inner' type to make it easier to write `stream_<provider>` functions
// (e.g. `stream_anthropic`). These functions can return `ProviderInferenceResponseStreamInner`,
// which will cause the compiler to coerce `Pin<Box<SomeUnderlyingStreamType>>` into
// `Pin<Box<dyn Stream>>`. The caller than then write `stream_anthropic().peekable()` to produce
// a `PeekableProviderInferenceResponseStream`. If we attempted to directly return a `Peekable<Pin<Box<dyn Stream>>>`,
// the compiler would fail to coerce `Peekable<Pin<Box<SomeUnderlyingStreamType>>>` into `Peekable<Pin<Box<dyn Stream>>>`.
// (due to the fact that unsized coercions are not supported on `Peekable` or other user-defined types).
// This would require `stream_<provider>` functions to first introduce a local variable with the correct
// `Pin<Box<dyn Stream>>` type, and then call `.peekable()` on that.
pub type ProviderInferenceResponseStreamInner =
    Pin<Box<dyn Stream<Item = Result<ProviderInferenceResponseChunk, Error>> + Send>>;

pub type PeekableProviderInferenceResponseStream = Peekable<ProviderInferenceResponseStreamInner>;

pub type InferenceResultStream =
    Pin<Box<dyn Stream<Item = Result<InferenceResultChunk, Error>> + Send>>;

impl From<JsonMode> for ModelInferenceRequestJsonMode {
    fn from(json_enforcement: JsonMode) -> Self {
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

/// Serializes a value that implements `Serialize` into a JSON string.
/// If serialization fails, it logs the error and returns an empty string.
///
/// # Arguments
///
/// * `value` - A reference to the value to be serialized.
///
/// # Returns
///
/// A `String` containing the serialized JSON, or an empty string if serialization fails.
pub fn serialize_or_log<T: Serialize>(value: &T) -> String {
    match serde_json::to_string(value) {
        Ok(serialized) => serialized,
        Err(e) => {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize value: {}", e),
            });
            String::new()
        }
    }
}

/// Handles a textual content block (text or thought)
/// It checks if there is already a block with the given id, and if so, appends the text to it.
/// Otherwise, it creates a new block and inserts it into the map.
/// It also updates the TTFT if it hasn't been set
fn handle_textual_content_block<F, A>(
    blocks: &mut HashMap<String, ContentBlockOutput>,
    id: String,
    text: String,
    ttft: &mut Option<Duration>,
    chunk_latency: Duration,
    create_block: F,
    append_text: A,
) where
    F: FnOnce(String) -> ContentBlockOutput,
    A: FnOnce(&mut ContentBlockOutput, &str),
{
    match blocks.get_mut(&id) {
        // If there is already a block, append to it
        Some(existing_block) => append_text(existing_block, &text),
        // If there is no block, create one
        _ => {
            // We only want to set TTFT if there is some real content
            if ttft.is_none() {
                *ttft = Some(chunk_latency);
            }
            if !text.is_empty() {
                blocks.insert(id, create_block(text));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function::{FunctionConfigChat, FunctionConfigJson};
    use crate::inference::providers::test_helpers::get_temperature_tool_config;
    use crate::jsonschema_util::JSONSchemaFromPath;
    use crate::minijinja_util::TemplateConfig;
    use crate::tool::ToolConfig;
    use crate::tool::{DynamicToolConfig, ToolChoice};
    use serde_json::json;
    use tokio::time::Instant;

    #[tokio::test]
    async fn test_create_chat_inference_response() {
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
            system: None,
            input_messages: vec![],
            output: content.clone(),
            raw_request: raw_request.clone(),
            raw_response: "".to_string(),
            usage: usage.clone(),
            latency: Latency::NonStreaming {
                response_time: Duration::default(),
            },
            finish_reason: None,
            model_provider_name: "test_provider".into(),
            model_name: "test_model".into(),
            cached: false,
        }];
        let chat_inference_response = ChatInferenceResult::new(
            inference_id,
            content.clone(),
            usage.clone(),
            model_inference_responses,
            None,
            InferenceParams::default(),
            None,
        )
        .await;
        let output_content = ["Hello, world!".to_string().into()];
        assert_eq!(chat_inference_response.content, output_content);
        assert_eq!(chat_inference_response.usage, usage);
        assert_eq!(chat_inference_response.model_inference_results.len(), 1);
        assert_eq!(chat_inference_response.finish_reason, None);
        let model_inference_result = chat_inference_response
            .model_inference_results
            .first()
            .unwrap();
        assert_eq!(&*model_inference_result.model_name, "test_model");
        assert_eq!(
            &*model_inference_result.model_provider_name,
            "test_provider"
        );
        assert_eq!(model_inference_result.raw_request, raw_request);

        // Case 2: A tool call that fails argument validation
        let inference_id = Uuid::now_v7();
        let content = vec![ContentBlockOutput::ToolCall(ToolCall {
            name: "get_temperature".to_string(),
            arguments: r#"{"where": "the moon"}"#.to_string(),
            id: "0".to_string(),
        })];
        let model_inference_responses = vec![ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: vec![],
            output: content.clone(),
            raw_request: raw_request.clone(),
            raw_response: "".to_string(),
            usage: usage.clone(),
            latency: Latency::NonStreaming {
                response_time: Duration::default(),
            },
            finish_reason: Some(FinishReason::Stop),
            model_provider_name: "test_provider".into(),
            model_name: "test_model".into(),
            cached: false,
        }];

        let weather_tool_config = get_temperature_tool_config();
        let chat_inference_response = ChatInferenceResult::new(
            inference_id,
            content,
            usage.clone(),
            model_inference_responses,
            Some(&weather_tool_config),
            InferenceParams::default(),
            None,
        )
        .await;
        assert_eq!(chat_inference_response.content.len(), 1);
        let tool_call_block = chat_inference_response.content.first().unwrap();
        match tool_call_block {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "get_temperature");
                assert_eq!(tool_call.raw_arguments, r#"{"where": "the moon"}"#);
                assert_eq!(tool_call.id, "0");
                assert_eq!(tool_call.name, Some("get_temperature".to_string()));
                assert_eq!(tool_call.arguments, None);
            }
            _ => panic!("Expected a tool call block"),
        }
        assert_eq!(
            chat_inference_response.finish_reason,
            Some(FinishReason::Stop)
        );
        // Case 3: A tool call that fails name validation
        let inference_id = Uuid::now_v7();
        let content = vec![ContentBlockOutput::ToolCall(ToolCall {
            name: "bad name".to_string(),
            arguments: r#"{"where": "the moon"}"#.to_string(),
            id: "0".to_string(),
        })];
        let model_inference_responses = vec![ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: vec![],
            output: content.clone(),
            raw_request: raw_request.clone(),
            raw_response: "".to_string(),
            finish_reason: Some(FinishReason::Stop),
            usage: usage.clone(),
            latency: Latency::NonStreaming {
                response_time: Duration::default(),
            },
            model_provider_name: "test_provider".into(),
            model_name: "test_model".into(),
            cached: false,
        }];

        let chat_inference_response = ChatInferenceResult::new(
            inference_id,
            content,
            usage.clone(),
            model_inference_responses,
            Some(&weather_tool_config),
            InferenceParams::default(),
            None,
        )
        .await;
        assert_eq!(chat_inference_response.content.len(), 1);
        let tool_call_block = chat_inference_response.content.first().unwrap();
        match tool_call_block {
            ContentBlockChatOutput::ToolCall(tool_call) => {
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
        let content = vec![ContentBlockOutput::ToolCall(ToolCall {
            name: "get_temperature".to_string(),
            arguments: r#"{"location": "the moon", "units": "celsius"}"#.to_string(),
            id: "0".to_string(),
        })];
        let model_inference_responses = vec![ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: vec![],
            output: content.clone(),
            raw_request: raw_request.clone(),
            raw_response: "".to_string(),
            usage: usage.clone(),
            latency: Latency::NonStreaming {
                response_time: Duration::default(),
            },
            finish_reason: Some(FinishReason::ToolCall),
            model_provider_name: "test_provider".into(),
            model_name: "test_model".into(),
            cached: false,
        }];

        let chat_inference_response = ChatInferenceResult::new(
            inference_id,
            content,
            usage.clone(),
            model_inference_responses,
            Some(&weather_tool_config),
            InferenceParams::default(),
            None,
        )
        .await;
        assert_eq!(chat_inference_response.content.len(), 1);
        assert_eq!(
            chat_inference_response.finish_reason,
            Some(FinishReason::ToolCall)
        );
        let tool_call_block = chat_inference_response.content.first().unwrap();
        match tool_call_block {
            ContentBlockChatOutput::ToolCall(tool_call) => {
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

        // Case 5: Parallel tool calls
        let inference_id = Uuid::now_v7();
        let content = vec![
            ContentBlockOutput::ToolCall(ToolCall {
                name: "get_temperature".to_string(),
                arguments: r#"{"location": "moon", "units": "celsius"}"#.to_string(),
                id: "0".to_string(),
            }),
            ContentBlockOutput::ToolCall(ToolCall {
                name: "get_temperature".to_string(),
                arguments: r#"{"location": "mars", "units": "celsius"}"#.to_string(),
                id: "1".to_string(),
            }),
        ];
        let model_inference_responses = vec![ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: vec![],
            output: content.clone(),
            finish_reason: None,
            raw_request: raw_request.clone(),
            raw_response: "".to_string(),
            usage: usage.clone(),
            latency: Latency::NonStreaming {
                response_time: Duration::default(),
            },
            model_provider_name: "test_provider".into(),
            model_name: "test_model".into(),
            cached: false,
        }];

        let chat_inference_response = ChatInferenceResult::new(
            inference_id,
            content,
            usage.clone(),
            model_inference_responses,
            Some(&weather_tool_config),
            InferenceParams::default(),
            None,
        )
        .await;
        assert_eq!(chat_inference_response.content.len(), 2);

        // Verify first tool call
        match &chat_inference_response.content[0] {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "get_temperature");
                assert_eq!(
                    tool_call.raw_arguments,
                    r#"{"location": "moon", "units": "celsius"}"#
                );
                assert_eq!(tool_call.id, "0");
                assert_eq!(tool_call.name, Some("get_temperature".to_string()));
                assert_eq!(
                    tool_call.arguments,
                    Some(
                        serde_json::from_str(r#"{"location": "moon", "units": "celsius"}"#)
                            .unwrap()
                    )
                );
            }
            _ => panic!("Expected a tool call block"),
        }

        // Verify second tool call
        match &chat_inference_response.content[1] {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "get_temperature");
                assert_eq!(
                    tool_call.raw_arguments,
                    r#"{"location": "mars", "units": "celsius"}"#
                );
                assert_eq!(tool_call.id, "1");
                assert_eq!(tool_call.name, Some("get_temperature".to_string()));
                assert_eq!(
                    tool_call.arguments,
                    Some(
                        serde_json::from_str(r#"{"location": "mars", "units": "celsius"}"#)
                            .unwrap()
                    )
                );
            }
            _ => panic!("Expected a tool call block"),
        }

        // Case 5b: Parallel tool calls with one invalid call
        let inference_id = Uuid::now_v7();
        let content = vec![
            ContentBlockOutput::ToolCall(ToolCall {
                name: "get_temperature".to_string(),
                arguments: r#"{"location": "moon", "units": "celsius"}"#.to_string(),
                id: "0".to_string(),
            }),
            ContentBlockOutput::ToolCall(ToolCall {
                name: "get_temperature".to_string(),
                arguments: r#"{"invalid": "args"}"#.to_string(),
                id: "1".to_string(),
            }),
        ];
        let model_inference_responses = vec![ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: vec![],
            output: content.clone(),
            finish_reason: None,
            raw_request: raw_request.clone(),
            raw_response: "".to_string(),
            usage: usage.clone(),
            latency: Latency::NonStreaming {
                response_time: Duration::default(),
            },
            model_provider_name: "test_provider".into(),
            model_name: "test_model".into(),
            cached: false,
        }];

        let chat_inference_response = ChatInferenceResult::new(
            inference_id,
            content,
            usage.clone(),
            model_inference_responses,
            Some(&weather_tool_config),
            InferenceParams::default(),
            None,
        )
        .await;
        assert_eq!(chat_inference_response.content.len(), 2);

        // Verify first tool call (valid)
        match &chat_inference_response.content[0] {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "get_temperature");
                assert_eq!(
                    tool_call.raw_arguments,
                    r#"{"location": "moon", "units": "celsius"}"#
                );
                assert_eq!(tool_call.id, "0");
                assert_eq!(tool_call.name, Some("get_temperature".to_string()));
                assert_eq!(
                    tool_call.arguments,
                    Some(
                        serde_json::from_str(r#"{"location": "moon", "units": "celsius"}"#)
                            .unwrap()
                    )
                );
            }
            _ => panic!("Expected a tool call block"),
        }

        // Verify second tool call (invalid arguments)
        match &chat_inference_response.content[1] {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "get_temperature");
                assert_eq!(tool_call.raw_arguments, r#"{"invalid": "args"}"#);
                assert_eq!(tool_call.id, "1");
                assert_eq!(tool_call.name, Some("get_temperature".to_string()));
                assert_eq!(tool_call.arguments, None); // Arguments should be None due to validation failure
            }
            _ => panic!("Expected a tool call block"),
        }

        // Case 6: Additional tools
        let inference_id = Uuid::now_v7();
        let additional_tool_config = ToolCallConfig {
            tools_available: vec![ToolConfig::Dynamic(DynamicToolConfig {
                name: "custom_tool".to_string(),
                description: "A custom tool".to_string(),
                parameters: DynamicJSONSchema::new(
                    serde_json::from_str(
                        r#"{
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"}
                    },
                    "required": ["input"]
                }"#,
                    )
                    .unwrap(),
                ),
                strict: true,
            })],
            tool_choice: ToolChoice::None,
            parallel_tool_calls: None,
        };

        // Test valid arguments for additional tool
        let content = vec![ContentBlockOutput::ToolCall(ToolCall {
            name: "custom_tool".to_string(),
            arguments: r#"{"input": "test"}"#.to_string(),
            id: "0".to_string(),
        })];
        let model_inference_responses = vec![ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: vec![],
            output: content.clone(),
            finish_reason: None,
            raw_request: raw_request.clone(),
            raw_response: "".to_string(),
            usage: usage.clone(),
            latency: Latency::NonStreaming {
                response_time: Duration::default(),
            },
            model_provider_name: "test_provider".into(),
            model_name: "test_model".into(),
            cached: false,
        }];

        let chat_inference_response = ChatInferenceResult::new(
            inference_id,
            content,
            usage.clone(),
            model_inference_responses,
            Some(&additional_tool_config),
            InferenceParams::default(),
            None,
        )
        .await;
        assert_eq!(chat_inference_response.content.len(), 1);

        // Verify valid tool call
        match &chat_inference_response.content[0] {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "custom_tool");
                assert_eq!(tool_call.raw_arguments, r#"{"input": "test"}"#);
                assert_eq!(tool_call.id, "0");
                assert_eq!(tool_call.name, Some("custom_tool".to_string()));
                assert_eq!(
                    tool_call.arguments,
                    Some(serde_json::from_str(r#"{"input": "test"}"#).unwrap())
                );
            }
            _ => panic!("Expected a tool call block"),
        }

        // Test invalid arguments for additional tool
        let content = vec![ContentBlockOutput::ToolCall(ToolCall {
            name: "custom_tool".to_string(),
            arguments: r#"{"wrong": "field"}"#.to_string(),
            id: "1".to_string(),
        })];
        let model_inference_responses = vec![ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: vec![],
            output: content.clone(),
            finish_reason: None,
            raw_request: raw_request.clone(),
            raw_response: "".to_string(),
            usage: usage.clone(),
            latency: Latency::NonStreaming {
                response_time: Duration::default(),
            },
            model_provider_name: "test_provider".into(),
            model_name: "test_model".into(),
            cached: false,
        }];

        let chat_inference_response = ChatInferenceResult::new(
            inference_id,
            content,
            usage.clone(),
            model_inference_responses,
            Some(&additional_tool_config),
            InferenceParams::default(),
            None,
        )
        .await;
        assert_eq!(chat_inference_response.content.len(), 1);

        // Verify invalid tool call
        match &chat_inference_response.content[0] {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "custom_tool");
                assert_eq!(tool_call.raw_arguments, r#"{"wrong": "field"}"#);
                assert_eq!(tool_call.id, "1");
                assert_eq!(tool_call.name, Some("custom_tool".to_string()));
                assert_eq!(tool_call.arguments, None); // Arguments should be None due to validation failure
            }
            _ => panic!("Expected a tool call block"),
        }

        // Case 7: Allowed tools restriction
        let inference_id = Uuid::now_v7();
        let restricted_tool_config = ToolCallConfig {
            tools_available: vec![ToolConfig::Dynamic(DynamicToolConfig {
                name: "weather_tool".to_string(),
                description: "Get weather information".to_string(),
                parameters: DynamicJSONSchema::new(
                    serde_json::from_str(
                        r#"{
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }"#,
                    )
                    .unwrap(),
                ),
                strict: true,
            })],
            tool_choice: ToolChoice::None,
            parallel_tool_calls: None,
        };

        // Test allowed tool call
        let content = vec![ContentBlockOutput::ToolCall(ToolCall {
            name: "weather_tool".to_string(),
            arguments: r#"{"location": "moon", "units": "celsius"}"#.to_string(),
            id: "0".to_string(),
        })];
        let model_inference_responses = vec![ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: vec![],
            output: content.clone(),
            raw_request: raw_request.clone(),
            raw_response: "".to_string(),
            usage: usage.clone(),
            finish_reason: None,
            latency: Latency::NonStreaming {
                response_time: Duration::default(),
            },
            model_provider_name: "test_provider".into(),
            model_name: "test_model".into(),
            cached: false,
        }];

        let chat_inference_response = ChatInferenceResult::new(
            inference_id,
            content,
            usage.clone(),
            model_inference_responses,
            Some(&restricted_tool_config),
            InferenceParams::default(),
            None,
        )
        .await;
        assert_eq!(chat_inference_response.content.len(), 1);

        // Verify allowed tool call
        match &chat_inference_response.content[0] {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "weather_tool");
                assert_eq!(
                    tool_call.raw_arguments,
                    r#"{"location": "moon", "units": "celsius"}"#
                );
                assert_eq!(tool_call.id, "0");
                assert_eq!(tool_call.name, Some("weather_tool".to_string()));
                assert_eq!(
                    tool_call.arguments,
                    Some(
                        serde_json::from_str(r#"{"location": "moon", "units": "celsius"}"#)
                            .unwrap()
                    )
                );
            }
            _ => panic!("Expected a tool call block"),
        }

        // Test disallowed tool call
        let content = vec![ContentBlockOutput::ToolCall(ToolCall {
            name: "get_humidity".to_string(), // This tool is not in the restricted config
            arguments: r#"{"location": "moon"}"#.to_string(),
            id: "1".to_string(),
        })];
        let model_inference_responses = vec![ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: vec![],
            output: content.clone(),
            finish_reason: None,
            raw_request: raw_request.clone(),
            raw_response: "".to_string(),
            usage: usage.clone(),
            latency: Latency::NonStreaming {
                response_time: Duration::default(),
            },
            model_provider_name: "test_provider".into(),
            model_name: "test_model".into(),
            cached: false,
        }];

        let chat_inference_response = ChatInferenceResult::new(
            inference_id,
            content,
            usage.clone(),
            model_inference_responses,
            Some(&restricted_tool_config),
            InferenceParams::default(),
            None,
        )
        .await;
        assert_eq!(chat_inference_response.content.len(), 1);

        // Verify disallowed tool call
        match &chat_inference_response.content[0] {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "get_humidity");
                assert_eq!(tool_call.raw_arguments, r#"{"location": "moon"}"#);
                assert_eq!(tool_call.id, "1");
                assert_eq!(tool_call.name, None); // Name should be None since tool is not allowed
                assert_eq!(tool_call.arguments, None); // Arguments should be None since tool is not allowed
            }
            _ => panic!("Expected a tool call block"),
        }
    }

    #[tokio::test]
    async fn test_collect_chunks() {
        // Test case 1: empty chunks (should error)
        let templates = TemplateConfig::default();

        let chunks = vec![];
        let function_config = Arc::new(FunctionConfig::Chat(FunctionConfigChat::default()));
        let model_name = "test_model";
        let model_provider_name = "test_provider";
        let raw_request = "raw request".to_string();
        let collect_chunks_args = CollectChunksArgs {
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            value: chunks,
            system: None,
            input_messages: vec![],
            function: function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            inference_params: InferenceParams::default(),
            function_name: "",
            variant_name: "",
            dynamic_output_schema: None,
            templates: &templates,
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
        };
        let result = collect_chunks(collect_chunks_args).await;
        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::TypeConversion {
                message:
                    "Attempted to create an InferenceResult from an empty response chunk vector"
                        .to_string(),
            }
            .into()
        );

        // Test case 2: non-empty chunks with no tool calls but content exists
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let created = current_timestamp();
        let content = vec![ContentBlockChunk::Text(TextChunk {
            text: "Hello,".to_string(),
            id: "0".to_string(),
        })];
        let latency = Duration::from_millis(150);
        let chunks = vec![
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content,
                created,
                usage: None,
                raw_response: "{\"message\": \"Hello}".to_string(),
                latency,
                finish_reason: None,
            }),
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
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
                finish_reason: Some(FinishReason::Stop),
            }),
        ];
        let collect_chunks_args = CollectChunksArgs {
            inference_id,
            episode_id,
            value: chunks,
            system: None,
            input_messages: vec![],
            function: function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            inference_params: InferenceParams::default(),
            function_name: "",
            variant_name: "",
            dynamic_output_schema: None,
            templates: &templates,
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
        };
        let result = collect_chunks(collect_chunks_args).await.unwrap();
        let chat_result = match result {
            InferenceResult::Chat(chat_result) => chat_result,
            _ => panic!("Expected Chat inference response"),
        };
        assert_eq!(chat_result.inference_id, inference_id);
        assert_eq!(chat_result.created, created);
        assert_eq!(chat_result.finish_reason, Some(FinishReason::Stop));
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
        assert_eq!(&*model_inference_result.model_name, model_name);
        assert_eq!(
            &*model_inference_result.model_provider_name,
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
        let json_function_config = Arc::new(FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            implicit_tool_call_config,
            output_schema,
        }));
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
                raw: Some("{\"name\":".to_string()),
                thought: Some("Thought 1".to_string()),
                created,
                usage: Some(usage1.clone()),
                raw_response: "{\"name\":".to_string(),
                latency: Duration::from_millis(150),
                finish_reason: Some(FinishReason::ToolCall),
            }),
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                raw: Some("\"John\",\"age\":30}".to_string()),
                thought: Some("Thought 2".to_string()),
                created,
                usage: Some(usage2.clone()),
                raw_response: "\"John\",\"age\":30}".to_string(),
                latency: Duration::from_millis(250),
                finish_reason: Some(FinishReason::Stop),
            }),
        ];
        let collect_chunks_args = CollectChunksArgs {
            value: chunks,
            system: None,
            inference_id,
            episode_id,
            input_messages: vec![],
            function: json_function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            inference_params: InferenceParams::default(),
            function_name: "",
            variant_name: "",
            dynamic_output_schema: None,
            templates: &templates,
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
        };
        let response = collect_chunks(collect_chunks_args).await.unwrap();
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
                assert_eq!(json_result.finish_reason, Some(FinishReason::Stop));
                assert_eq!(json_result.model_inference_results.len(), 1);
                let model_inference_result = json_result.model_inference_results.first().unwrap();
                assert_eq!(&*model_inference_result.model_name, model_name);
                assert_eq!(
                    &*model_inference_result.model_provider_name,
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
                raw: Some("{\"name\":".to_string()),
                thought: Some("Thought 1".to_string()),
                created,
                usage: Some(usage.clone()),
                raw_response: "{\"name\":".to_string(),
                latency: Duration::from_millis(100),
                finish_reason: Some(FinishReason::ToolCall),
            }),
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                raw: Some("\"John\"}".to_string()),
                thought: None,
                created,
                usage: None,
                raw_response: "\"John\"}".to_string(),
                latency: Duration::from_millis(200),
                finish_reason: None,
            }),
        ];
        let collect_chunks_args = CollectChunksArgs {
            value: chunks,
            inference_id,
            episode_id,
            system: None,
            input_messages: vec![],
            function: json_function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            inference_params: InferenceParams::default(),
            function_name: "",
            variant_name: "",
            dynamic_output_schema: None,
            templates: &templates,
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
        };
        let result = collect_chunks(collect_chunks_args).await;
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
                assert_eq!(&*model_inference_result.model_name, model_name);
                assert_eq!(
                    &*model_inference_result.model_provider_name,
                    model_provider_name
                );
                assert_eq!(model_inference_result.raw_request, raw_request);
            }
            _ => panic!("Expected Json inference response"),
        }

        // Test case 5: chunks with some None content
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let created = current_timestamp();
        let usage = Usage {
            input_tokens: 15,
            output_tokens: 10,
        };
        let chunks = vec![
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                raw: Some("{\"name\":\"John\",".to_string()),
                thought: None,
                created,
                usage: Some(usage.clone()),
                raw_response: "{\"name\":\"John\",".to_string(),
                latency: Duration::from_millis(100),
                finish_reason: None,
            }),
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                raw: Some("".to_string()),
                thought: Some("Thought 2".to_string()),
                created,
                usage: None,
                raw_response: "".to_string(),
                latency: Duration::from_millis(200),
                finish_reason: None,
            }),
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                raw: Some("\"age\":30}".to_string()),
                thought: None,
                created,
                usage: None,
                raw_response: "\"age\":30}".to_string(),
                latency: Duration::from_millis(300),
                finish_reason: Some(FinishReason::Stop),
            }),
        ];
        let collect_chunks_args = CollectChunksArgs {
            value: chunks,
            inference_id,
            episode_id,
            system: None,
            input_messages: vec![],
            function: function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            inference_params: InferenceParams::default(),
            function_name: "",
            variant_name: "",
            dynamic_output_schema: None,
            templates: &templates,
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
        };
        let result = collect_chunks(collect_chunks_args).await;
        if let Ok(InferenceResult::Chat(chat_response)) = result {
            assert_eq!(chat_response.inference_id, inference_id);
            assert_eq!(chat_response.created, created);
            assert_eq!(
                chat_response.content,
                vec![
                    ContentBlockChatOutput::Thought(Thought {
                        text: "Thought 2".to_string(),
                        signature: None,
                    }),
                    ContentBlockChatOutput::Text(Text {
                        text: "{\"name\":\"John\",\"age\":30}".to_string()
                    }),
                ]
            );
            assert_eq!(chat_response.usage, usage);
            assert_eq!(chat_response.model_inference_results.len(), 1);
            let model_inference_result = chat_response.model_inference_results.first().unwrap();
            assert_eq!(&*model_inference_result.model_name, model_name);
            assert_eq!(chat_response.finish_reason, Some(FinishReason::Stop));
            assert_eq!(
                model_inference_result.finish_reason,
                Some(FinishReason::Stop)
            );
            assert_eq!(
                &*model_inference_result.model_provider_name,
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
        let json_function_config = Arc::new(FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            implicit_tool_call_config,
            output_schema,
        }));
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
                raw: Some("{\"name\":".to_string()),
                thought: Some("Thought 1".to_string()),
                created,
                usage: Some(usage1.clone()),
                raw_response: "{\"name\":".to_string(),
                latency: Duration::from_millis(150),
                finish_reason: Some(FinishReason::ToolCall),
            }),
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                raw: Some("\"John\",\"age\":30}".to_string()),
                thought: Some("Thought 2".to_string()),
                created,
                usage: Some(usage2.clone()),
                raw_response: "\"John\",\"age\":30}".to_string(),
                latency: Duration::from_millis(250),
                finish_reason: Some(FinishReason::Stop),
            }),
        ];
        let collect_chunks_args = CollectChunksArgs {
            inference_id,
            episode_id,
            value: chunks,
            system: None,
            input_messages: vec![],
            function: json_function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            inference_params: InferenceParams::default(),
            function_name: "",
            variant_name: "",
            dynamic_output_schema: None,
            templates: &templates,
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
        };
        let response = collect_chunks(collect_chunks_args).await.unwrap();
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
                assert_eq!(json_result.finish_reason, Some(FinishReason::Stop));
                assert_eq!(json_result.model_inference_results.len(), 1);
                let model_inference_result = json_result.model_inference_results.first().unwrap();
                assert_eq!(&*model_inference_result.model_name, model_name);
                assert_eq!(
                    &*model_inference_result.model_provider_name,
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
        let json_function_config = Arc::new(FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            implicit_tool_call_config,
            output_schema,
        }));
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
        let templates = TemplateConfig::default();
        let chunks = vec![
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                raw: Some("{\"name\":".to_string()),
                thought: Some("Thought 1".to_string()),
                created,
                usage: Some(usage1.clone()),
                finish_reason: Some(FinishReason::Stop),
                raw_response: "{\"name\":".to_string(),
                latency: Duration::from_millis(150),
            }),
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                raw: Some("\"John\",\"age\":30}".to_string()),
                thought: Some("Thought 2".to_string()),
                created,
                usage: Some(usage2.clone()),
                finish_reason: Some(FinishReason::ToolCall),
                raw_response: "\"John\",\"age\":30}".to_string(),
                latency: Duration::from_millis(250),
            }),
        ];
        let collect_chunks_args = CollectChunksArgs {
            inference_id,
            episode_id,
            value: chunks,
            system: None,
            input_messages: vec![],
            function: json_function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            inference_params: InferenceParams::default(),
            function_name: "",
            variant_name: "",
            dynamic_output_schema: Some(dynamic_output_schema),
            templates: &templates,
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
        };
        let response = collect_chunks(collect_chunks_args).await.unwrap();
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
                assert_eq!(&*model_inference_result.model_name, model_name);
                assert_eq!(
                    model_inference_result.finish_reason,
                    Some(FinishReason::ToolCall)
                );
                assert_eq!(
                    &*model_inference_result.model_provider_name,
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
            InputMessageContent::Text(TextKind::Text { text }) => {
                assert_eq!(text, "Hello, world!")
            }
            _ => panic!("Expected Text content: {message:?}"),
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
            InputMessageContent::Text(TextKind::Arguments { arguments }) => {
                assert_eq!(arguments, json!({"key": "value"}).as_object().unwrap())
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
            InputMessageContent::Text(TextKind::LegacyValue { value }) => {
                assert_eq!(value, "Hello")
            }
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
                {"type": "tool_call", "id": "456", "name": "another_tool", "arguments": {"key": "value"}}
            ]
        });
        let message: InputMessage = serde_json::from_value(input).unwrap();
        assert_eq!(message.role, Role::User);
        assert_eq!(message.content.len(), 2);
        match &message.content[0] {
            InputMessageContent::Text(TextKind::LegacyValue { value }) => {
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
                assert_eq!(tool_call.arguments, "{\"key\":\"value\"}");
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

    #[test]
    fn test_json_inference_result_chunk_from_provider_chunk() {
        use std::time::Duration;

        // Test case for ToolCall content
        let tool_chunk = ProviderInferenceResponseChunk {
            content: vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "123".to_string(),
                raw_arguments: "{\"key\": \"value\"}".to_string(),
                raw_name: "test_tool".to_string(),
            })],
            created: 1234567890,
            usage: Some(Usage {
                input_tokens: 10,
                output_tokens: 20,
            }),
            raw_response: "raw response".to_string(),
            latency: Duration::from_secs(1),
            finish_reason: Some(FinishReason::ToolCall),
        };

        let result = JsonInferenceResultChunk::from(tool_chunk);
        assert_eq!(result.raw, Some("{\"key\": \"value\"}".to_string()));
        assert_eq!(result.thought, None);
        assert_eq!(result.created, 1234567890);
        assert_eq!(result.raw_response, "raw response");
        assert_eq!(result.latency, Duration::from_secs(1));
        assert_eq!(
            result.usage,
            Some(Usage {
                input_tokens: 10,
                output_tokens: 20
            })
        );
        assert_eq!(result.finish_reason, Some(FinishReason::ToolCall));
        // Test case for Text content
        let text_chunk = ProviderInferenceResponseChunk {
            content: vec![ContentBlockChunk::Text(TextChunk {
                id: "123".to_string(),
                text: "some text".to_string(),
            })],
            created: 1234567890,
            usage: None,
            raw_response: "raw response".to_string(),
            latency: Duration::from_secs(1),
            finish_reason: None,
        };

        let result = JsonInferenceResultChunk::from(text_chunk);
        assert_eq!(result.raw, Some("some text".to_string()));
        assert_eq!(result.thought, None);

        // Test case for Thought content
        let thought_chunk = ProviderInferenceResponseChunk {
            content: vec![ContentBlockChunk::Thought(ThoughtChunk {
                id: "123".to_string(),
                text: Some("thinking...".to_string()),
                signature: None,
            })],
            created: 1234567890,
            usage: None,
            raw_response: "raw response".to_string(),
            latency: Duration::from_secs(1),
            finish_reason: None,
        };

        let result = JsonInferenceResultChunk::from(thought_chunk);
        assert_eq!(result.raw, None);
        assert_eq!(result.thought, Some("thinking...".to_string()));
        assert_eq!(result.finish_reason, None);
        // Test case for multiple content blocks - should use last raw content
        let mixed_chunk = ProviderInferenceResponseChunk {
            content: vec![
                ContentBlockChunk::Text(TextChunk {
                    id: "123".to_string(),
                    text: "first text".to_string(),
                }),
                ContentBlockChunk::ToolCall(ToolCallChunk {
                    id: "456".to_string(),
                    raw_arguments: "final content".to_string(),
                    raw_name: "test_tool".to_string(),
                }),
                ContentBlockChunk::Thought(ThoughtChunk {
                    id: "789".to_string(),
                    text: Some("final thought".to_string()),
                    signature: None,
                }),
            ],
            created: 1234567890,
            usage: None,
            raw_response: "raw response".to_string(),
            latency: Duration::from_secs(1),
            finish_reason: None,
        };

        let result = JsonInferenceResultChunk::from(mixed_chunk);
        assert_eq!(result.raw, Some("final content".to_string()));
        assert_eq!(result.thought, Some("final thought".to_string()));

        // Test case for empty content
        let empty_chunk = ProviderInferenceResponseChunk {
            content: vec![],
            created: 1234567890,
            usage: None,
            raw_response: "raw response".to_string(),
            latency: Duration::from_secs(1),
            finish_reason: None,
        };

        let result = JsonInferenceResultChunk::from(empty_chunk);
        assert_eq!(result.raw, None);
        assert_eq!(result.thought, None);
        assert_eq!(result.finish_reason, None);
    }

    #[test]
    fn test_handle_textual_content_block() {
        let mut blocks: HashMap<String, ContentBlockOutput> = HashMap::new();
        let mut ttft: Option<Duration> = None;
        let chunk_latency = Duration::from_millis(100);

        // Test case 1: Create new text block
        handle_textual_content_block(
            &mut blocks,
            "1".to_string(),
            "Hello".to_string(),
            &mut ttft,
            chunk_latency,
            |text| ContentBlockOutput::Text(Text { text }),
            |block, text| {
                if let ContentBlockOutput::Text(Text {
                    text: existing_text,
                }) = block
                {
                    existing_text.push_str(text);
                }
            },
        );

        assert_eq!(blocks.len(), 1);
        assert_eq!(ttft, Some(chunk_latency));
        match blocks.get("1").unwrap() {
            ContentBlockOutput::Text(Text { text }) => assert_eq!(text, "Hello"),
            _ => panic!("Expected text block"),
        }

        // Test case 2: Append to existing text block
        handle_textual_content_block(
            &mut blocks,
            "1".to_string(),
            " World".to_string(),
            &mut ttft,
            chunk_latency,
            |text| ContentBlockOutput::Text(Text { text }),
            |block, text| {
                if let ContentBlockOutput::Text(Text {
                    text: existing_text,
                }) = block
                {
                    existing_text.push_str(text);
                }
            },
        );

        assert_eq!(blocks.len(), 1);
        match blocks.get("1").unwrap() {
            ContentBlockOutput::Text(Text { text }) => assert_eq!(text, "Hello World"),
            _ => panic!("Expected text block"),
        }

        // Test case 3: Empty text should not create block
        handle_textual_content_block(
            &mut blocks,
            "2".to_string(),
            "".to_string(),
            &mut ttft,
            chunk_latency,
            |text| ContentBlockOutput::Text(Text { text }),
            |block, text| {
                if let ContentBlockOutput::Text(Text {
                    text: existing_text,
                }) = block
                {
                    existing_text.push_str(text);
                }
            },
        );

        assert_eq!(blocks.len(), 1); // Should still only have the first block

        // Test case 4: Create thought block
        handle_textual_content_block(
            &mut blocks,
            "3".to_string(),
            "Thinking...".to_string(),
            &mut ttft,
            chunk_latency,
            |text| {
                ContentBlockOutput::Thought(Thought {
                    text,
                    signature: None,
                })
            },
            |block, text| {
                if let ContentBlockOutput::Thought(thought) = block {
                    thought.text.push_str(text);
                }
            },
        );

        assert_eq!(blocks.len(), 2);
        match blocks.get("3").unwrap() {
            ContentBlockOutput::Thought(Thought { text, signature: _ }) => {
                assert_eq!(text, "Thinking...")
            }
            _ => panic!("Expected thought block"),
        }
    }
}
