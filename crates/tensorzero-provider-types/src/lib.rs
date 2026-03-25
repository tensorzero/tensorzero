use std::borrow::Cow;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use futures::Stream;
use futures::future::Shared;
use futures::stream::Peekable;
use mime::MediaType;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;
use tensorzero_derive::TensorZeroDeserialize;
use tensorzero_error::Error;
use tensorzero_types::inference_params::JsonMode;
use tensorzero_types::{ApiType, Text, Thought, ToolCall, Unknown};
use tensorzero_types_providers::fireworks::FireworksFinishReason;
use tensorzero_types_providers::openai::{OpenAIFinishReason, OpenAIUsage};
use tensorzero_types_providers::together::TogetherFinishReason;
use tensorzero_types_providers::xai::XAIUsage;
use url::Url;
use uuid::Uuid;

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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct Usage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_cache_read_input_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_cache_write_input_tokens: Option<u32>,
    #[serde(default, with = "decimal_float_option")]
    #[cfg_attr(feature = "ts-bindings", ts(type = "number | null"))]
    pub cost: Option<Decimal>,
}

/// Custom serde module for `Option<Decimal>` as float.
///
/// Serializes identically to `rust_decimal::serde::float_option`.
/// Deserializes via `Option<f64>` instead of `deserialize_option` so that
/// serde's untagged-enum `ContentDeserializer` (which maps JSON `null` to
/// `Content::Unit` → `visit_unit`) is handled correctly.  The upstream
/// `OptionDecimalVisitor` only implements `visit_none`, not `visit_unit`,
/// which causes failures inside `#[serde(untagged)]` enums.
mod decimal_float_option {
    use rust_decimal::Decimal;
    use serde::{Deserialize, Deserializer, Serializer};

    #[expect(clippy::ref_option)]
    pub fn serialize<S>(value: &Option<Decimal>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        rust_decimal::serde::float_option::serialize(value, serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Decimal>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<f64>::deserialize(deserializer)?
            .map(|f| Decimal::try_from(f).map_err(serde::de::Error::custom))
            .transpose()
    }
}

impl Usage {
    /// Returns a `Usage` with core fields at zero and cache fields at `None`.
    ///
    /// Cache fields start as `None` (meaning "not reported") because not all
    /// providers support prompt caching. The lenient aggregation helpers
    /// (`aggregate_usage_across_model_inferences`, `sum_usage_strict`) will
    /// preserve any `Some` value they encounter rather than contaminating to `None`.
    pub fn zero() -> Usage {
        Usage {
            input_tokens: Some(0),
            output_tokens: Some(0),
            provider_cache_read_input_tokens: None,
            provider_cache_write_input_tokens: None,
            cost: Some(Decimal::ZERO),
        }
    }

    pub fn total_tokens(&self) -> Option<u32> {
        match (self.input_tokens, self.output_tokens) {
            (Some(input), Some(output)) => Some(input + output),
            _ => None,
        }
    }
}

/// A single entry in the raw usage array, representing usage data from one model inference.
/// This preserves the original provider-specific usage object for fields that TensorZero
/// normalizes away (e.g., OpenAI's `reasoning_tokens`, Anthropic's `cache_read_input_tokens`).
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct RawUsageEntry {
    pub model_inference_id: Uuid,
    pub provider_type: String,
    pub api_type: ApiType,
    pub data: Value,
}

pub fn raw_usage_entries_from_value(
    model_inference_id: Uuid,
    provider_type: &str,
    api_type: ApiType,
    usage: Value,
) -> Vec<RawUsageEntry> {
    vec![RawUsageEntry {
        model_inference_id,
        provider_type: provider_type.to_string(),
        api_type,
        data: usage,
    }]
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize, sqlx::Type)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(rename_all = "snake_case")]
#[sqlx(type_name = "text", rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    StopSequence,
    Length,
    ToolCall,
    ContentFilter,
    Unknown,
}

#[derive(Clone, Copy, Default, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelInferenceRequestJsonMode {
    #[default]
    Off,
    On,
    Strict,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, TensorZeroDeserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ContentBlockOutput {
    Text(Text),
    ToolCall(ToolCall),
    Thought(Thought),
    Unknown(Unknown),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum FlattenUnknown<'a, T> {
    Normal(T),
    Unknown(Cow<'a, Value>),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolCallChunk {
    pub id: String,
    #[serde(serialize_with = "serialize_option_string_as_empty")]
    pub raw_name: Option<String>,
    pub raw_arguments: String,
}

#[expect(clippy::ref_option)]
fn serialize_option_string_as_empty<S>(
    value: &Option<String>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match value {
        Some(s) => serializer.serialize_str(s),
        None => serializer.serialize_str(""),
    }
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
    pub summary_id: Option<String>,
    pub summary_text: Option<String>,

    /// See `Thought.provider_type`
    #[serde(
        alias = "_internal_provider_type",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_type: Option<String>,
    /// Provider-specific opaque data for multi-turn reasoning support.
    /// See `Thought.extra_data`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_data: Option<Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnknownChunk {
    pub id: String,
    pub data: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, TensorZeroDeserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ContentBlockChunk {
    Text(TextChunk),
    ToolCall(ToolCallChunk),
    Thought(ThoughtChunk),
    Unknown(UnknownChunk),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ProviderInferenceResponseChunk {
    pub content: Vec<ContentBlockChunk>,
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_usage: Option<Vec<RawUsageEntry>>,
    pub raw_response: String,
    /// Time elapsed between making the request to the model provider and receiving this chunk.
    /// Important: this is NOT latency from the start of the TensorZero request.
    pub provider_latency: Duration,
    pub finish_reason: Option<FinishReason>,
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
            usage,
            raw_usage: None,
            raw_response,
            provider_latency: latency,
            finish_reason,
        }
    }

    pub fn new_with_raw_usage(
        content: Vec<ContentBlockChunk>,
        usage: Option<Usage>,
        raw_response: String,
        latency: Duration,
        finish_reason: Option<FinishReason>,
        raw_usage: Option<Vec<RawUsageEntry>>,
    ) -> Self {
        Self {
            content,
            usage,
            raw_usage,
            raw_response,
            provider_latency: latency,
            finish_reason,
        }
    }
}

pub type ProviderInferenceResponseStreamInner =
    Pin<Box<dyn Stream<Item = Result<ProviderInferenceResponseChunk, Error>> + Send>>;

pub type PeekableProviderInferenceResponseStream = Peekable<ProviderInferenceResponseStreamInner>;

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(any(feature = "e2e_tests", test), derive(PartialEq))]
pub struct ProviderInferenceResponse {
    pub id: Uuid,
    pub output: Vec<ContentBlockOutput>,
    pub system: Option<String>,
    pub input_messages: Vec<RequestMessage>,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    /// Time elapsed between making the request to the model provider and receiving the response.
    /// Important: this is NOT latency from the start of the TensorZero request.
    pub provider_latency: Latency,
    pub finish_reason: Option<FinishReason>,
    pub raw_usage: Option<Vec<RawUsageEntry>>,
    pub relay_raw_response: Option<Vec<tensorzero_types::RawResponseEntry>>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(any(feature = "e2e_tests", test), derive(PartialEq))]
pub struct RequestMessage {
    pub role: tensorzero_types::Role,
    pub content: Vec<ContentBlock>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(any(feature = "e2e_tests", test), derive(PartialEq))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text(Text),
    ToolCall(ToolCall),
    ToolResult(tensorzero_types::ToolResult),
    #[serde(alias = "image")]
    File(Box<LazyFile>),
    Thought(Thought),
    Unknown(Unknown),
}

pub type FileFuture = Shared<
    Pin<Box<dyn Future<Output = Result<tensorzero_types::ObjectStorageFile, Error>> + Send>>,
>;

// This gets serialized as part of a `ModelInferenceRequest` when we compute a cache key.
#[derive(Clone, Debug, Serialize)]
pub enum LazyFile {
    // Client sent a file URL → must fetch & store
    Url {
        file_url: FileUrl,
        #[serde(skip)]
        future: FileFuture,
    },
    // Client sent a base64-encoded file → skip fetch, must store
    Base64(PendingObjectStoreFile),
    // Client sent an object storage file → must fetch, skip store
    ObjectStoragePointer {
        metadata: tensorzero_types::Base64FileMetadata,
        storage_path: tensorzero_types::StoragePath,
        #[serde(skip)]
        future: FileFuture,
    },
    // Client sent a resolved object storage file → skip fetch & store
    ObjectStorage(tensorzero_types::ObjectStorageFile),
}

#[cfg(any(test, feature = "e2e_tests"))]
impl std::cmp::PartialEq for LazyFile {
    #[expect(clippy::panic)]
    fn eq(&self, _other: &Self) -> bool {
        panic!("Tried to check LazyFile equality")
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct FileUrl {
    pub url: Url,
    pub mime_type: Option<MediaType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<tensorzero_types::Detail>,
    pub filename: Option<String>,
}

#[cfg(any(test, feature = "e2e_tests"))]
impl PartialEq for FileUrl {
    fn eq(&self, other: &Self) -> bool {
        self.url == other.url
            && self.mime_type == other.mime_type
            && self.detail == other.detail
            && self.filename == other.filename
    }
}

/// A newtype wrapper around `ObjectStorageFile` that represents file data
/// from a base64 input that needs to be written to object storage.
/// The `storage_path` inside is content-addressed (computed from data) and represents
/// where the file WILL be written, not where it currently exists.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PendingObjectStoreFile(pub tensorzero_types::ObjectStorageFile);

impl std::ops::Deref for PendingObjectStoreFile {
    type Target = tensorzero_types::ObjectStorageFile;
    fn deref(&self) -> &Self::Target {
        &self.0
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

impl std::fmt::Display for RequestMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl From<JsonMode> for ModelInferenceRequestJsonMode {
    fn from(mode: JsonMode) -> Self {
        match mode {
            JsonMode::On => ModelInferenceRequestJsonMode::On,
            JsonMode::Strict => ModelInferenceRequestJsonMode::Strict,
            JsonMode::Tool => ModelInferenceRequestJsonMode::Off,
            JsonMode::Off => ModelInferenceRequestJsonMode::Off,
        }
    }
}

impl From<OpenAIUsage> for Usage {
    fn from(usage: OpenAIUsage) -> Self {
        Usage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            provider_cache_read_input_tokens: usage
                .prompt_tokens_details
                .and_then(|d| d.cached_tokens),
            provider_cache_write_input_tokens: None,
            cost: None,
        }
    }
}

impl From<Option<OpenAIUsage>> for Usage {
    fn from(usage: Option<OpenAIUsage>) -> Self {
        match usage {
            Some(u) => u.into(),
            None => Usage::default(),
        }
    }
}

impl From<OpenAIFinishReason> for FinishReason {
    fn from(reason: OpenAIFinishReason) -> Self {
        match reason {
            OpenAIFinishReason::Stop => FinishReason::Stop,
            OpenAIFinishReason::Length => FinishReason::Length,
            OpenAIFinishReason::ContentFilter => FinishReason::ContentFilter,
            OpenAIFinishReason::ToolCalls => FinishReason::ToolCall,
            OpenAIFinishReason::FunctionCall => FinishReason::ToolCall,
            OpenAIFinishReason::Unknown => FinishReason::Unknown,
        }
    }
}

impl From<FireworksFinishReason> for FinishReason {
    fn from(reason: FireworksFinishReason) -> Self {
        match reason {
            FireworksFinishReason::Stop => FinishReason::Stop,
            FireworksFinishReason::Length => FinishReason::Length,
            FireworksFinishReason::ToolCalls => FinishReason::ToolCall,
            FireworksFinishReason::ContentFilter => FinishReason::ContentFilter,
            FireworksFinishReason::Unknown => FinishReason::Unknown,
        }
    }
}

impl From<TogetherFinishReason> for FinishReason {
    fn from(reason: TogetherFinishReason) -> Self {
        match reason {
            TogetherFinishReason::Stop => FinishReason::Stop,
            TogetherFinishReason::Eos => FinishReason::Stop,
            TogetherFinishReason::Length => FinishReason::Length,
            TogetherFinishReason::ToolCalls => FinishReason::ToolCall,
            TogetherFinishReason::FunctionCall => FinishReason::ToolCall,
            TogetherFinishReason::Unknown => FinishReason::Unknown,
        }
    }
}

impl From<XAIUsage> for Usage {
    fn from(usage: XAIUsage) -> Self {
        let output_tokens = match (usage.completion_tokens, usage.completion_tokens_details) {
            (Some(completion), Some(details)) => {
                Some(completion + details.reasoning_tokens.unwrap_or(0))
            }
            (Some(completion), None) => Some(completion),
            (None, Some(details)) => details.reasoning_tokens,
            (None, None) => None,
        };
        Usage {
            input_tokens: usage.prompt_tokens,
            output_tokens,
            provider_cache_read_input_tokens: None,
            provider_cache_write_input_tokens: None,
            cost: None,
        }
    }
}
