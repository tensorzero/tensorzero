//! Shared types for the TensorZero provider interface.
//!
//! This crate contains types that form the boundary between provider implementations
//! and TensorZero core. By isolating these types in their own crate, we enable:
//! - Parallel compilation of providers and core
//! - Isolated serde derive expansion
//! - Tighter incremental compilation boundaries

pub mod credential_validation;
pub mod credentials;
pub mod embeddings;
pub mod extra_body;
pub mod extra_headers;
pub mod inference_response;
pub(crate) mod serde_helpers;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

use std::borrow::Cow;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use derive_builder::Builder;
use futures::FutureExt;
use futures::Stream;
use futures::future::Shared;
use futures::stream::Peekable;
use mime::MediaType;
use rust_decimal::Decimal;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;
use tensorzero_derive::{TensorZeroDeserialize, export_schema};
use tensorzero_error::Error;
use tensorzero_types::inference_params::JsonMode;
use tensorzero_types::{ApiType, Text, Thought, ToolCall, Unknown};
use tensorzero_types_providers::deepseek::DeepSeekUsage;
use tensorzero_types_providers::fireworks::FireworksFinishReason;
use tensorzero_types_providers::openai::{OpenAIFinishReason, OpenAIUsage};
use tensorzero_types_providers::together::TogetherFinishReason;
use tensorzero_types_providers::xai::XAIUsage;
use url::Url;
use uuid::Uuid;

// =============================================================================
// Latency
// =============================================================================

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

// =============================================================================
// Usage
// =============================================================================

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct Usage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    // Omit from serialized output when None (per AGENTS.md convention for optional fields).
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

    // Signature is required by serde's `with` attribute.
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

// =============================================================================
// RawUsageEntry
// =============================================================================

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

// =============================================================================
// FinishReason
// =============================================================================

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

// =============================================================================
// ModelInferenceRequestJsonMode
// =============================================================================

#[derive(Clone, Copy, Default, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelInferenceRequestJsonMode {
    #[default]
    Off,
    On,
    Strict,
}

// =============================================================================
// EmbeddingEncodingFormat
// =============================================================================

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingEncodingFormat {
    #[default]
    Float,
    Base64,
}

// =============================================================================
// MIME type helpers
// =============================================================================

/// Tries to convert a mime type to a file extension, picking an arbitrary extension if there are multiple
/// extensions for the mime type.
/// This is used when writing a file input to object storage, and when determining the file name
/// to provide to OpenAI (which doesn't accept mime types for file input)
pub fn mime_type_to_ext(mime_type: &MediaType) -> Result<Option<&'static str>, Error> {
    Ok(match mime_type {
        _ if mime_type == "image/jpeg" => Some("jpg"),
        _ if mime_type == "image/png" => Some("png"),
        _ if mime_type == "image/gif" => Some("gif"),
        _ if mime_type == "application/pdf" => Some("pdf"),
        _ if mime_type == "image/webp" => Some("webp"),
        _ if mime_type == "text/plain" => Some("txt"),
        _ if mime_type == "audio/midi" => Some("mid"),
        _ if mime_type == "audio/mpeg" || mime_type == "audio/mp3" => Some("mp3"),
        _ if mime_type == "audio/m4a" || mime_type == "audio/mp4" => Some("m4a"),
        _ if mime_type == "audio/ogg" => Some("ogg"),
        _ if mime_type == "audio/x-flac" || mime_type == "audio/flac" => Some("flac"),
        _ if mime_type == "audio/x-wav"
            || mime_type == "audio/wav"
            || mime_type == "audio/wave" =>
        {
            Some("wav")
        }
        _ if mime_type == "audio/amr" => Some("amr"),
        _ if mime_type == "audio/aac" || mime_type == "audio/x-aac" => Some("aac"),
        _ if mime_type == "audio/x-aiff" || mime_type == "audio/aiff" => Some("aiff"),
        _ if mime_type == "audio/x-dsf" => Some("dsf"),
        _ if mime_type == "audio/x-ape" => Some("ape"),
        _ if mime_type == "audio/webm" => Some("webm"),
        _ => {
            let guess = mime_guess::get_mime_extensions_str(mime_type.as_ref())
                .and_then(|types| types.last());
            if guess.is_some() {
                tracing::warn!(
                    "Guessed file extension `{guess:?}` for MIME type `{mime_type}`. This may not be correct."
                );
            }
            guess.copied()
        }
    })
}

/// Converts audio MIME types to OpenAI's audio format strings.
pub fn mime_type_to_audio_format(mime_type: &MediaType) -> Result<&'static str, Error> {
    if mime_type.type_() != mime::AUDIO {
        return Err(Error::new(tensorzero_error::ErrorDetails::InvalidMessage {
            message: format!("Expected audio MIME type, got: {mime_type}"),
        }));
    }

    mime_type_to_ext(mime_type)?.ok_or_else(|| {
        Error::new(tensorzero_error::ErrorDetails::InvalidMessage {
            message: format!(
                "Unsupported audio MIME type: {mime_type}. Supported types: audio/midi, audio/mpeg, audio/m4a, audio/mp4, audio/ogg, audio/x-flac, audio/x-wav, audio/amr, audio/aac, audio/x-aiff, audio/x-dsf, audio/x-ape. Please open a feature request if your provider supports another audio format: https://github.com/tensorzero/tensorzero/discussions/categories/feature-requests"
            ),
        })
    })
}

// =============================================================================
// LazyFileExt
// =============================================================================

pub trait LazyFileExt {
    fn resolve(
        &self,
    ) -> impl Future<Output = Result<Cow<'_, tensorzero_types::ObjectStorageFile>, Error>> + Send;
}

impl LazyFileExt for LazyFile {
    async fn resolve(&self) -> Result<Cow<'_, tensorzero_types::ObjectStorageFile>, Error> {
        match self {
            LazyFile::Url {
                future,
                file_url: _,
            } => Ok(Cow::Owned(future.clone().await?)),
            LazyFile::Base64(pending) => Ok(Cow::Borrowed(&pending.0)),
            LazyFile::ObjectStoragePointer { future, .. } => Ok(Cow::Owned(future.clone().await?)),
            LazyFile::ObjectStorage(resolved) => Ok(Cow::Borrowed(resolved)),
        }
    }
}

// =============================================================================
// ContentBlockOutput
// =============================================================================

/// Types of content blocks that can be returned by a model provider
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

// =============================================================================
// FlattenUnknown
// =============================================================================

/// Helper enum for deserializing provider responses that may contain unknown fields.
/// When a provider returns a known type, it's deserialized as `Normal(T)`.
/// When a provider returns an unknown type, it's captured as `Unknown(Value)`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum FlattenUnknown<'a, T> {
    Normal(T),
    Unknown(Cow<'a, Value>),
}

// =============================================================================
// ToolCallChunk
// =============================================================================

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolCallChunk {
    pub id: String,
    #[serde(serialize_with = "serialize_option_string_as_empty")]
    pub raw_name: Option<String>,
    pub raw_arguments: String,
}

// Signature dictated by Serde
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

// =============================================================================
// Streaming chunk types
// =============================================================================

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
        // This alias is written to the database, so we cannot remove it.
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

// =============================================================================
// ProviderInferenceResponseChunk
// =============================================================================

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

// =============================================================================
// Stream type aliases
// =============================================================================

pub type ProviderInferenceResponseStreamInner =
    Pin<Box<dyn Stream<Item = Result<ProviderInferenceResponseChunk, Error>> + Send>>;

pub type PeekableProviderInferenceResponseStream = Peekable<ProviderInferenceResponseStreamInner>;

// =============================================================================
// ProviderInferenceResponse and Args
// =============================================================================

/// Each provider transforms a `ModelInferenceRequest` into a provider-specific request type,
/// then transforms the provider response into a `ProviderInferenceResponse`.
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(any(feature = "e2e_tests", test), derive(PartialEq))]
#[non_exhaustive]
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
    /// Raw usage entries for `include_raw_usage` feature.
    pub raw_usage: Option<Vec<RawUsageEntry>>,
    /// Raw response entries for `include_raw_response` feature.
    pub relay_raw_response: Option<Vec<tensorzero_types::RawResponseEntry>>,
}

pub struct ProviderInferenceResponseArgs {
    pub output: Vec<ContentBlockOutput>,
    pub system: Option<String>,
    pub input_messages: Vec<RequestMessage>,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub raw_usage: Option<Vec<RawUsageEntry>>,
    pub relay_raw_response: Option<Vec<tensorzero_types::RawResponseEntry>>,
    /// Time elapsed between making the request to the model provider and receiving the response.
    /// Important: this is NOT latency from the start of the TensorZero request.
    pub provider_latency: Latency,
    pub finish_reason: Option<FinishReason>,
    pub id: Uuid,
}

impl ProviderInferenceResponse {
    pub fn new(args: ProviderInferenceResponseArgs) -> Self {
        let sanitized_raw_request = sanitize_raw_request(&args.input_messages, args.raw_request);

        Self {
            id: args.id,
            output: args.output,
            system: args.system,
            input_messages: args.input_messages,
            raw_request: sanitized_raw_request,
            raw_response: args.raw_response,
            usage: args.usage,
            provider_latency: args.provider_latency,
            finish_reason: args.finish_reason,
            raw_usage: args.raw_usage,
            relay_raw_response: args.relay_raw_response,
        }
    }
}

/// Strips out file data from the raw request, replacing it with a placeholder.
/// This is a best-effort attempt to avoid filling up ClickHouse with file data.
pub fn sanitize_raw_request(input_messages: &[RequestMessage], mut raw_request: String) -> String {
    let mut i = 0;
    for message in input_messages {
        for content in &message.content {
            if let ContentBlock::File(file) = content {
                let file_with_path = match &**file {
                    LazyFile::Url {
                        future,
                        file_url: _,
                    } => {
                        if let Some(Ok(resolved)) = future.clone().now_or_never() {
                            Some(Cow::Owned(tensorzero_types::File::ObjectStorage(resolved)))
                        } else {
                            None
                        }
                    }
                    LazyFile::Base64(pending) => Some(Cow::Owned(
                        tensorzero_types::File::ObjectStorage(pending.0.clone()),
                    )),
                    LazyFile::ObjectStorage(resolved) => Some(Cow::Owned(
                        tensorzero_types::File::ObjectStorage(resolved.clone()),
                    )),
                    LazyFile::ObjectStoragePointer { future, .. } => {
                        if let Some(Ok(resolved)) = future.clone().now_or_never() {
                            Some(Cow::Owned(tensorzero_types::File::ObjectStorage(resolved)))
                        } else {
                            None
                        }
                    }
                };
                if let Some(file) = file_with_path {
                    let data = match &*file {
                        tensorzero_types::File::ObjectStorage(resolved) => &resolved.data,
                        tensorzero_types::File::Base64(base64) => base64.data(),
                        tensorzero_types::File::Url(_)
                        | tensorzero_types::File::ObjectStoragePointer(_)
                        | tensorzero_types::File::ObjectStorageError(_) => {
                            continue;
                        }
                    };
                    raw_request = raw_request.replace(data, &format!("<TENSORZERO_FILE_{i}>"));
                    i += 1;
                }
            }
        }
    }
    raw_request
}

// =============================================================================
// RequestMessage and ContentBlock
// =============================================================================

/// A RequestMessage is a message sent to a model
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(any(feature = "e2e_tests", test), derive(PartialEq))]
pub struct RequestMessage {
    pub role: tensorzero_types::Role,
    pub content: Vec<ContentBlock>,
}

/// Core representation of the types of content that could go into a model provider
/// The `PartialEq` impl will panic if we try to compare a `LazyFile`, so we make it
/// test-only to prevent production code from panicking.
/// This *does not* implement `Deserialize`, since we need object store information
/// to produce a `LazyFile::Url`
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

// =============================================================================
// LazyFile, FileUrl, PendingObjectStoreFile, FileFuture
// =============================================================================

/// Holds a lazily-resolved file from a `LazyResolvedInputMessageContent::File`.
/// This is constructed as either:
/// 1. An immediately-ready future, when we're converting a `ResolvedInputMessageContent` to a `LazyResolvedInputMessageContent`
/// 2. A network fetch future, when we're resolving an image url in `InputMessageContent::File`.
///
/// This future is `Shared`, so that we can `.await` it from multiple different model providers
/// (if we're not forwarding an image url to the model provider), as well as when writing the
/// file to the object store (if enabled).
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
    // This is only used in tests, so it's fine to panic
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

// =============================================================================
// Trait impls
// =============================================================================

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

// =============================================================================
// From impls for provider-specific types
// =============================================================================

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
        // Add `reasoning_tokens` to `completion_tokens` for total output tokens
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
            provider_cache_read_input_tokens: usage
                .prompt_tokens_details
                .and_then(|d| d.cached_tokens),
            provider_cache_write_input_tokens: None,
            cost: None,
        }
    }
}

impl From<DeepSeekUsage> for Usage {
    fn from(usage: DeepSeekUsage) -> Self {
        Usage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            provider_cache_read_input_tokens: usage.prompt_cache_hit_tokens,
            // DeepSeek's `prompt_cache_miss_tokens` = tokens not in cache, which are
            // written to cache for future requests, so we map miss → write.
            provider_cache_write_input_tokens: usage.prompt_cache_miss_tokens,
            cost: None,
        }
    }
}

// =============================================================================
// BatchStatus, StartBatchProviderInferenceResponse, PollBatchInferenceResponse,
// ProviderBatchInferenceOutput, ProviderBatchInferenceResponse
// =============================================================================

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, sqlx::Type)]
#[serde(rename_all = "snake_case")]
#[sqlx(type_name = "text", rename_all = "snake_case")]
pub enum BatchStatus {
    Pending,
    Completed,
    Failed,
}

/// Returned from start_batch_inference from an InferenceProvider
pub struct StartBatchProviderInferenceResponse {
    pub batch_id: Uuid,
    pub raw_requests: Vec<String>,
    pub batch_params: Value,
    pub raw_request: String,
    pub raw_response: String,
    pub status: BatchStatus,
    pub errors: Vec<Value>,
}

#[derive(Debug)]
pub enum PollBatchInferenceResponse {
    Pending {
        raw_request: String,
        raw_response: String,
    },
    Completed(ProviderBatchInferenceResponse),
    Failed {
        raw_request: String,
        raw_response: String,
    },
}

#[derive(Debug)]
pub struct ProviderBatchInferenceOutput {
    pub id: Uuid,
    pub output: Vec<ContentBlockOutput>,
    pub raw_response: String,
    pub usage: Usage,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug)]
pub struct ProviderBatchInferenceResponse {
    pub raw_request: String,
    pub raw_response: String,
    pub elements: HashMap<Uuid, ProviderBatchInferenceOutput>,
}

// =============================================================================
// Tool types (no core deps)
// =============================================================================

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
#[schemars(title = "ProviderToolScopeModelProvider")]
#[cfg_attr(feature = "ts-bindings", ts(optional_fields))]
pub struct ProviderToolScopeModelProvider {
    pub model_name: String,
    #[serde(alias = "model_provider_name", skip_serializing_if = "Option::is_none")] // legacy
    pub provider_name: Option<String>,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize, JsonSchema)]
#[serde(untagged)]
#[export_schema]
#[cfg_attr(feature = "ts-bindings", ts(optional_fields))]
pub enum ProviderToolScope {
    #[default]
    Unscoped,
    ModelProvider(ProviderToolScopeModelProvider),
}

impl ProviderToolScope {
    pub fn matches(&self, scope_model_name: &str, scope_provider_name: &str) -> bool {
        match self {
            ProviderToolScope::Unscoped => true,
            ProviderToolScope::ModelProvider(mp) => {
                if scope_model_name != mp.model_name {
                    return false;
                }
                match &mp.provider_name {
                    Some(pn) => scope_provider_name == pn,
                    None => true, // If provider_name is None, match any provider for this model
                }
            }
        }
    }
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct ProviderTool {
    #[serde(default)]
    pub scope: ProviderToolScope,
    pub tool: Value,
}

impl std::fmt::Display for ProviderTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct OpenAICustomTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<OpenAICustomToolFormat>,
}

impl std::fmt::Display for OpenAICustomTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl OpenAICustomTool {
    #[getter]
    pub fn get_name(&self) -> &str {
        &self.name
    }

    #[getter]
    pub fn get_description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    #[getter]
    pub fn get_format<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<pyo3::Bound<'py, pyo3::PyAny>>> {
        match &self.format {
            Some(format) => {
                let json_str = serde_json::to_string(format).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Failed to serialize format: {e:?}"
                    ))
                })?;
                let json_mod = py.import("json")?;
                let result = json_mod.call_method1("loads", (json_str,))?;
                Ok(Some(result))
            }
            None => Ok(None),
        }
    }

    pub fn __repr__(&self) -> String {
        format!("OpenAICustomTool(name='{}')", self.name)
    }
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAICustomToolFormat {
    #[schemars(title = "OpenAICustomToolFormatText")]
    Text,
    #[schemars(title = "OpenAICustomToolFormatGrammar")]
    Grammar { grammar: OpenAIGrammarDefinition },
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct OpenAIGrammarDefinition {
    pub syntax: OpenAIGrammarSyntax,
    pub definition: String,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(rename_all = "snake_case")]
pub enum OpenAIGrammarSyntax {
    Lark,
    Regex,
}

/// Records / lists the tools that were allowed in the request.
/// Also lists how they were set (default, dynamically set).
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct AllowedTools {
    pub tools: Vec<String>,
    pub choice: AllowedToolsChoice,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(rename_all = "snake_case")]
pub enum AllowedToolsChoice {
    /// If `allowed_tools` is not explicitly passed, we set the function tools
    /// by default and add any dynamic tools
    #[default]
    FunctionDefault,
    /// If `allowed_tools` was explicitly passed we use that list only and then automatically add dynamically set tools.
    /// This is deprecated but we keep it around as it may still be in the database.
    #[deprecated]
    DynamicAllowedTools,
    /// Currently, we match OpenAI in that if allowed tools is set we only allow the tools that are in it.
    Explicit,
}

impl AllowedTools {
    pub fn into_dynamic_allowed_tools(self) -> Option<Vec<String>> {
        #[expect(deprecated)]
        match self.choice {
            AllowedToolsChoice::FunctionDefault => None,
            AllowedToolsChoice::DynamicAllowedTools | AllowedToolsChoice::Explicit => {
                Some(self.tools)
            }
        }
    }

    pub fn as_dynamic_allowed_tools(&self) -> Option<Vec<&str>> {
        #[expect(deprecated)]
        match self.choice {
            AllowedToolsChoice::FunctionDefault => None,
            AllowedToolsChoice::DynamicAllowedTools | AllowedToolsChoice::Explicit => {
                Some(self.tools.iter().map(|s| s.as_str()).collect())
            }
        }
    }
}

// =============================================================================
// FunctionToolDef, ProviderToolCallConfig, ToolConfigRef
// =============================================================================

/// A tool definition as seen by providers — carries the compiled schema value.
/// This is the provider-facing counterpart of `FunctionToolConfig` in tensorzero-core.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[cfg_attr(any(feature = "e2e_tests", test), derive(Deserialize))]
pub struct FunctionToolDef {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub strict: bool,
}

/// Provider-facing tool call configuration.
/// This is constructed from `ToolCallConfig` in tensorzero-core before building a `ModelInferenceRequest`.
/// Note: `ToolCallConfig` in tensorzero-core holds compiled `JSONSchema` objects for server-side
/// validation; this type holds only what providers need (raw `Value` parameters).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(any(feature = "e2e_tests", test), derive(Deserialize))]
pub struct ProviderToolCallConfig {
    pub tools: Vec<FunctionToolDef>,
    pub provider_tools: Vec<ProviderTool>,
    pub openai_custom_tools: Vec<OpenAICustomTool>,
    pub tool_choice: tensorzero_types::ToolChoice,
    pub parallel_tool_calls: Option<bool>,
    pub allowed_tools: AllowedTools,
}

impl ProviderToolCallConfig {
    /// Returns an iterator over all function tool defs (no custom tools).
    /// Returns an error if OpenAI custom tools are present and the provider doesn't support them.
    pub fn tools_available(
        &self,
    ) -> Result<impl Iterator<Item = &FunctionToolDef>, tensorzero_error::Error> {
        if !self.openai_custom_tools.is_empty() {
            return Err(tensorzero_error::Error::new(
                tensorzero_error::ErrorDetails::IncompatibleTool {
                    message: "OpenAI custom tools are not supported by this provider".to_string(),
                },
            ));
        }
        Ok(self.tools.iter())
    }

    /// Like `tools_available` but also applies `AllowedTools::Explicit` filtering.
    pub fn strict_tools_available(
        &self,
    ) -> Result<impl Iterator<Item = &FunctionToolDef>, tensorzero_error::Error> {
        if !self.openai_custom_tools.is_empty() {
            return Err(tensorzero_error::Error::new(
                tensorzero_error::ErrorDetails::IncompatibleTool {
                    message: "OpenAI custom tools are not supported by this provider".to_string(),
                },
            ));
        }
        let iter: Box<dyn Iterator<Item = &FunctionToolDef>> = match self.allowed_tools.choice {
            #[expect(deprecated)]
            AllowedToolsChoice::FunctionDefault | AllowedToolsChoice::DynamicAllowedTools => {
                Box::new(self.tools.iter())
            }
            AllowedToolsChoice::Explicit => Box::new(
                self.tools
                    .iter()
                    .filter(|t| self.allowed_tools.tools.contains(&t.name)),
            ),
        };
        Ok(iter)
    }

    pub fn any_tools_available(&self) -> bool {
        !self.tools.is_empty()
    }

    pub fn get_function_tool(&self, name: &str) -> Option<&FunctionToolDef> {
        self.tools.iter().find(|t| t.name == name)
    }

    pub fn get_scoped_provider_tools(
        &self,
        model_name: &str,
        model_provider_name: &str,
    ) -> Vec<&ProviderTool> {
        self.provider_tools
            .iter()
            .filter(|t| t.scope.matches(model_name, model_provider_name))
            .collect()
    }

    /// Returns an iterator over all tools including OpenAI custom tools.
    /// Used by the OpenAI provider.
    pub fn tools_available_with_openai_custom(&self) -> impl Iterator<Item = ToolConfigRef<'_>> {
        self.tools.iter().map(ToolConfigRef::Function).chain(
            self.openai_custom_tools
                .iter()
                .map(ToolConfigRef::OpenAICustom),
        )
    }
}

/// Reference to either a function tool def or an OpenAI custom tool.
pub enum ToolConfigRef<'a> {
    Function(&'a FunctionToolDef),
    OpenAICustom(&'a OpenAICustomTool),
}

// =============================================================================
// ModelInferenceRequest
// =============================================================================

/// Top-level TensorZero type for an inference request to a particular model.
/// This should contain all the information required to make a valid inference request
/// for a provider, except for information about what model to actually request,
/// and to convert it back to the appropriate response format.
#[derive(Builder, Clone, Debug, Default, Serialize)]
#[cfg_attr(any(feature = "e2e_tests", test), derive(PartialEq))]
#[builder(setter(into, strip_option), default)]
pub struct ModelInferenceRequest<'a> {
    pub inference_id: Uuid,
    pub messages: Vec<RequestMessage>,
    pub system: Option<String>,
    pub tool_config: Option<Cow<'a, ProviderToolCallConfig>>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<u32>,
    pub stop_sequences: Option<Cow<'a, [String]>>,
    pub stream: bool,
    pub json_mode: ModelInferenceRequestJsonMode,
    pub function_type: tensorzero_types::FunctionType,
    pub output_schema: Option<&'a Value>,
    pub extra_body: extra_body::FullExtraBodyConfig,
    pub extra_headers: extra_headers::FullExtraHeadersConfig,
    pub extra_cache_key: Option<String>,
    #[serde(flatten)]
    pub inference_params_v2: tensorzero_types::inference_params::ChatCompletionInferenceParamsV2,
    pub fetch_and_encode_input_files_before_inference: bool,
}

impl<'a> ModelInferenceRequest<'a> {
    pub fn borrow_stop_sequences(&'a self) -> Option<Cow<'a, [String]>> {
        self.stop_sequences.as_ref().map(|cow| match cow {
            Cow::Borrowed(s) => Cow::Borrowed(*s),
            Cow::Owned(s) => Cow::Borrowed(s.as_slice()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;
    use rust_decimal::Decimal;
    use serde_json::json;
    use std::time::Duration;
    use tensorzero_types::ToolChoice;
    use uuid::Uuid;

    // =========================================================================
    // ModelInferenceRequest
    // =========================================================================

    #[gtest]
    fn test_model_inference_request_roundtrip_serialization() {
        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: tensorzero_types::Role::User,
                content: vec!["What is the weather?".to_string().into()],
            }],
            system: Some("You are a helpful assistant.".to_string()),
            temperature: Some(0.5),
            max_tokens: Some(1024),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            seed: Some(42),
            stream: true,
            json_mode: ModelInferenceRequestJsonMode::Off,
            ..Default::default()
        };

        let json = serde_json::to_value(&request).expect("should serialize");
        assert_eq!(json["temperature"], json!(0.5));
        assert_eq!(json["max_tokens"], json!(1024));
        assert_eq!(json["seed"], json!(42));
        assert_eq!(json["stream"], json!(true));
        assert_eq!(json["messages"].as_array().map(|a| a.len()), Some(1));
    }

    #[gtest]
    fn test_model_inference_request_default() {
        let request = ModelInferenceRequest::default();
        expect_that!(request.temperature, eq(None));
        expect_that!(request.max_tokens, eq(None));
        expect_that!(request.stream, eq(false));
        expect_that!(request.messages.len(), eq(0));
        assert_eq!(request.system, None);
    }

    #[gtest]
    fn test_borrow_stop_sequences_owned() {
        let request = ModelInferenceRequest {
            stop_sequences: Some(Cow::Owned(vec!["STOP".to_string(), "END".to_string()])),
            ..Default::default()
        };
        let borrowed = request.borrow_stop_sequences();
        expect_that!(borrowed.is_some(), eq(true));
        let seqs = borrowed.unwrap();
        expect_that!(seqs.len(), eq(2));
        expect_that!(seqs[0].as_str(), eq("STOP"));
    }

    #[gtest]
    fn test_borrow_stop_sequences_none() {
        let request = ModelInferenceRequest::default();
        assert!(request.borrow_stop_sequences().is_none());
    }

    // =========================================================================
    // Usage
    // =========================================================================

    #[gtest]
    fn test_usage_zero() {
        let usage = Usage::zero();
        expect_that!(usage.input_tokens, eq(Some(0)));
        expect_that!(usage.output_tokens, eq(Some(0)));
        expect_that!(usage.cost, eq(Some(Decimal::ZERO)));
    }

    #[gtest]
    fn test_usage_total_tokens() {
        let usage = Usage {
            input_tokens: Some(10),
            output_tokens: Some(25),
            ..Default::default()
        };
        expect_that!(usage.total_tokens(), eq(Some(35)));
    }

    #[gtest]
    fn test_usage_total_tokens_partial() {
        let usage = Usage {
            input_tokens: Some(10),
            output_tokens: None,
            ..Default::default()
        };
        expect_that!(usage.total_tokens(), eq(None));
    }

    #[gtest]
    fn test_usage_serde_roundtrip() {
        let usage = Usage {
            input_tokens: Some(100),
            output_tokens: Some(50),
            cost: Some(Decimal::new(15, 4)), // 0.0015
            ..Default::default()
        };
        let json = serde_json::to_value(usage).expect("should serialize");
        assert_eq!(json["input_tokens"], json!(100));
        assert_eq!(json["output_tokens"], json!(50));

        let deserialized: Usage = serde_json::from_value(json).expect("should deserialize");
        expect_that!(deserialized.input_tokens, eq(Some(100)));
        expect_that!(deserialized.output_tokens, eq(Some(50)));
        expect_that!(deserialized.cost, eq(Some(Decimal::new(15, 4))));
    }

    // =========================================================================
    // FinishReason conversions from provider types
    // =========================================================================

    #[gtest]
    fn test_finish_reason_from_openai() {
        expect_that!(
            FinishReason::from(OpenAIFinishReason::Stop),
            eq(FinishReason::Stop)
        );
        expect_that!(
            FinishReason::from(OpenAIFinishReason::Length),
            eq(FinishReason::Length)
        );
        expect_that!(
            FinishReason::from(OpenAIFinishReason::ToolCalls),
            eq(FinishReason::ToolCall)
        );
        expect_that!(
            FinishReason::from(OpenAIFinishReason::ContentFilter),
            eq(FinishReason::ContentFilter)
        );
    }

    #[gtest]
    fn test_finish_reason_from_fireworks() {
        expect_that!(
            FinishReason::from(FireworksFinishReason::Stop),
            eq(FinishReason::Stop)
        );
        expect_that!(
            FinishReason::from(FireworksFinishReason::ToolCalls),
            eq(FinishReason::ToolCall)
        );
    }

    #[gtest]
    fn test_finish_reason_from_together() {
        expect_that!(
            FinishReason::from(TogetherFinishReason::Stop),
            eq(FinishReason::Stop)
        );
        expect_that!(
            FinishReason::from(TogetherFinishReason::Eos),
            eq(FinishReason::Stop)
        );
        expect_that!(
            FinishReason::from(TogetherFinishReason::ToolCalls),
            eq(FinishReason::ToolCall)
        );
    }

    // =========================================================================
    // Streaming chunks — construction and usage tracking
    // =========================================================================

    #[gtest]
    fn test_response_chunk_construction() {
        let chunk = ProviderInferenceResponseChunk::new(
            vec![ContentBlockChunk::Text(TextChunk {
                id: "0".to_string(),
                text: "Hello".to_string(),
            })],
            Some(Usage {
                input_tokens: Some(10),
                output_tokens: Some(1),
                ..Default::default()
            }),
            r#"{"choices":[...]}"#.to_string(),
            Duration::from_millis(50),
            None,
        );

        expect_that!(chunk.content.len(), eq(1));
        assert_eq!(chunk.usage.unwrap().input_tokens, Some(10));
        assert_eq!(chunk.finish_reason, None);
        assert_eq!(chunk.raw_usage, None);
    }

    #[gtest]
    fn test_response_chunk_with_raw_usage() {
        let raw = json!({"prompt_tokens": 10, "cached_tokens": 3});
        let model_id = Uuid::now_v7();
        let raw_entries =
            raw_usage_entries_from_value(model_id, "openai", ApiType::ChatCompletions, raw.clone());

        let chunk = ProviderInferenceResponseChunk::new_with_raw_usage(
            vec![],
            Some(Usage {
                input_tokens: Some(10),
                output_tokens: Some(5),
                ..Default::default()
            }),
            "raw".to_string(),
            Duration::from_millis(100),
            Some(FinishReason::Stop),
            Some(raw_entries),
        );

        expect_that!(chunk.finish_reason, eq(Some(FinishReason::Stop)));
        let raw_usage = chunk.raw_usage.expect("should have raw_usage");
        expect_that!(raw_usage.len(), eq(1));
        assert_eq!(raw_usage[0].provider_type, "openai");
        assert_eq!(raw_usage[0].data["cached_tokens"], json!(3));
    }

    #[gtest]
    fn test_streaming_usage_accumulation_across_chunks() {
        // Simulate a streaming response where usage arrives across chunks
        let chunks = [
            ProviderInferenceResponseChunk::new(
                vec![ContentBlockChunk::Text(TextChunk {
                    id: "0".to_string(),
                    text: "Hello".to_string(),
                })],
                None, // no usage in first content chunk
                "chunk1".to_string(),
                Duration::from_millis(50),
                None,
            ),
            ProviderInferenceResponseChunk::new(
                vec![ContentBlockChunk::Text(TextChunk {
                    id: "0".to_string(),
                    text: " world".to_string(),
                })],
                None,
                "chunk2".to_string(),
                Duration::from_millis(80),
                None,
            ),
            ProviderInferenceResponseChunk::new(
                vec![],
                Some(Usage {
                    input_tokens: Some(15),
                    output_tokens: Some(8),
                    ..Default::default()
                }),
                "chunk3".to_string(),
                Duration::from_millis(120),
                Some(FinishReason::Stop),
            ),
        ];

        // Accumulate text across chunks
        let full_text: String = chunks
            .iter()
            .flat_map(|c| c.content.iter())
            .filter_map(|block| match block {
                ContentBlockChunk::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect();
        expect_that!(full_text.as_str(), eq("Hello world"));

        // Usage arrives in the final chunk
        let final_usage = chunks
            .iter()
            .rev()
            .find_map(|c| c.usage.as_ref())
            .expect("should have usage in final chunk");
        expect_that!(final_usage.input_tokens, eq(Some(15)));
        expect_that!(final_usage.output_tokens, eq(Some(8)));

        // Finish reason in the last chunk
        expect_that!(
            chunks.last().unwrap().finish_reason,
            eq(Some(FinishReason::Stop))
        );
    }

    // =========================================================================
    // ContentBlockOutput and ContentBlockChunk serde
    // =========================================================================

    #[gtest]
    fn test_content_block_output_from_string() {
        let block: ContentBlockOutput = "test output".to_string().into();
        let json = serde_json::to_value(&block).expect("should serialize");
        assert_eq!(json["type"], json!("text"));
        assert_eq!(json["text"], json!("test output"));
    }

    #[gtest]
    fn test_content_block_chunk_serde_roundtrip() {
        let chunk = ContentBlockChunk::ToolCall(ToolCallChunk {
            id: "call_123".to_string(),
            raw_name: Some("get_weather".to_string()),
            raw_arguments: r#"{"location":"NYC"}"#.to_string(),
        });

        let json = serde_json::to_value(&chunk).expect("should serialize");
        assert_eq!(json["type"], json!("tool_call"));
        assert_eq!(json["id"], json!("call_123"));
        assert_eq!(json["raw_name"], json!("get_weather"));

        let deserialized: ContentBlockChunk =
            serde_json::from_value(json).expect("should deserialize");
        assert_eq!(deserialized, chunk);
    }

    #[gtest]
    fn test_tool_call_chunk_serializes_none_name_as_empty() {
        let chunk = ToolCallChunk {
            id: "call_456".to_string(),
            raw_name: None,
            raw_arguments: "{}".to_string(),
        };
        let json = serde_json::to_value(&chunk).expect("should serialize");
        // None raw_name should serialize as empty string
        assert_eq!(json["raw_name"], json!(""));
    }

    // =========================================================================
    // ProviderToolCallConfig
    // =========================================================================

    fn sample_tool_config() -> ProviderToolCallConfig {
        ProviderToolCallConfig {
            tools: vec![
                FunctionToolDef {
                    name: "get_weather".to_string(),
                    description: "Get weather for a location".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }),
                    strict: false,
                },
                FunctionToolDef {
                    name: "search".to_string(),
                    description: "Search the web".to_string(),
                    parameters: json!({"type": "object", "properties": {"query": {"type": "string"}}}),
                    strict: true,
                },
            ],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: Some(true),
            allowed_tools: AllowedTools {
                tools: vec!["get_weather".to_string()],
                choice: AllowedToolsChoice::Explicit,
            },
            ..Default::default()
        }
    }

    #[gtest]
    fn test_tool_config_any_tools_available() {
        let config = sample_tool_config();
        expect_that!(config.any_tools_available(), eq(true));

        let empty = ProviderToolCallConfig::default();
        expect_that!(empty.any_tools_available(), eq(false));
    }

    #[gtest]
    fn test_tool_config_get_function_tool() {
        let config = sample_tool_config();
        let weather = config.get_function_tool("get_weather");
        expect_that!(weather.is_some(), eq(true));
        expect_that!(weather.unwrap().strict, eq(false));

        let missing = config.get_function_tool("nonexistent");
        expect_that!(missing.is_none(), eq(true));
    }

    #[gtest]
    fn test_tool_config_tools_available_rejects_openai_custom() {
        let config = ProviderToolCallConfig {
            openai_custom_tools: vec![OpenAICustomTool {
                name: "web_search".to_string(),
                description: None,
                format: None,
            }],
            ..Default::default()
        };
        let result = config.tools_available();
        expect_that!(result.is_err(), eq(true));
    }

    #[gtest]
    fn test_strict_tools_available_filters_by_allowed() {
        let config = sample_tool_config();
        // AllowedToolsChoice::Explicit with only "get_weather" allowed
        let strict_tools: Vec<_> = config
            .strict_tools_available()
            .expect("should succeed")
            .collect();
        // Only "get_weather" should pass, "search" should be filtered out
        expect_that!(strict_tools.len(), eq(1));
        expect_that!(strict_tools[0].name.as_str(), eq("get_weather"));
    }

    #[gtest]
    fn test_provider_tool_scope_matching() {
        let unscoped = ProviderToolScope::Unscoped;
        expect_that!(unscoped.matches("any_model", "any_provider"), eq(true));

        let scoped = ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
            model_name: "gpt-4".to_string(),
            provider_name: Some("openai".to_string()),
        });
        expect_that!(scoped.matches("gpt-4", "openai"), eq(true));
        expect_that!(scoped.matches("gpt-4", "azure"), eq(false));
        expect_that!(scoped.matches("claude", "openai"), eq(false));

        let model_only = ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
            model_name: "gpt-4".to_string(),
            provider_name: None,
        });
        expect_that!(model_only.matches("gpt-4", "openai"), eq(true));
        expect_that!(model_only.matches("gpt-4", "azure"), eq(true));
        expect_that!(model_only.matches("claude", "openai"), eq(false));
    }

    // =========================================================================
    // ModelInferenceRequestJsonMode conversion from JsonMode
    // =========================================================================

    #[gtest]
    fn test_json_mode_conversion() {
        expect_that!(
            ModelInferenceRequestJsonMode::from(JsonMode::On),
            eq(ModelInferenceRequestJsonMode::On)
        );
        expect_that!(
            ModelInferenceRequestJsonMode::from(JsonMode::Strict),
            eq(ModelInferenceRequestJsonMode::Strict)
        );
        expect_that!(
            ModelInferenceRequestJsonMode::from(JsonMode::Off),
            eq(ModelInferenceRequestJsonMode::Off)
        );
        // Tool mode maps to Off for providers
        expect_that!(
            ModelInferenceRequestJsonMode::from(JsonMode::Tool),
            eq(ModelInferenceRequestJsonMode::Off)
        );
    }

    // =========================================================================
    // raw_usage_entries_from_value
    // =========================================================================

    #[gtest]
    fn test_raw_usage_entries_from_value() {
        let model_id = Uuid::now_v7();
        let raw = json!({"prompt_tokens": 100, "completion_tokens": 50, "reasoning_tokens": 10});
        let entries =
            raw_usage_entries_from_value(model_id, "openai", ApiType::ChatCompletions, raw.clone());

        expect_that!(entries.len(), eq(1));
        expect_that!(entries[0].model_inference_id, eq(model_id));
        assert_eq!(entries[0].provider_type, "openai");
        assert_eq!(entries[0].api_type, ApiType::ChatCompletions);
        assert_eq!(entries[0].data, raw);
    }

    // =========================================================================
    // Latency serialization
    // =========================================================================

    #[gtest]
    fn test_latency_streaming_serialization() {
        let latency = Latency::Streaming {
            ttft: Duration::from_millis(150),
            response_time: Duration::from_millis(2000),
        };
        let json = serde_json::to_value(&latency).expect("should serialize");
        // Streaming variant has ttft and response_time
        assert!(json["ttft"].is_object(), "ttft should be present");
        assert!(
            json["response_time"].is_object(),
            "response_time should be present"
        );
    }

    #[gtest]
    fn test_latency_non_streaming_serialization() {
        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(500),
        };
        let json = serde_json::to_value(&latency).expect("should serialize");
        assert!(
            json["response_time"].is_object(),
            "response_time should be present"
        );
        // Should NOT have ttft
        assert!(json.get("ttft").is_none(), "ttft should not be present");
    }

    // =========================================================================
    // AllowedTools
    // =========================================================================

    #[gtest]
    fn test_allowed_tools_into_dynamic() {
        let explicit = AllowedTools {
            tools: vec!["tool_a".to_string(), "tool_b".to_string()],
            choice: AllowedToolsChoice::Explicit,
        };
        let dynamic = explicit.into_dynamic_allowed_tools();
        assert!(dynamic.is_some(), "explicit should produce dynamic tools");
        assert_eq!(dynamic.unwrap().len(), 2);

        let default_tools = AllowedTools {
            tools: vec!["tool_a".to_string()],
            choice: AllowedToolsChoice::FunctionDefault,
        };
        assert!(
            default_tools.into_dynamic_allowed_tools().is_none(),
            "FunctionDefault should return None"
        );
    }
}
