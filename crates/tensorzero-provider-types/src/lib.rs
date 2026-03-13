//! Shared types for the TensorZero provider interface.
//!
//! This crate contains types that form the boundary between provider implementations
//! and TensorZero core. By isolating these types in their own crate, we enable:
//! - Parallel compilation of providers and core
//! - Isolated serde derive expansion
//! - Tighter incremental compilation boundaries

pub mod extra_body;
pub mod extra_headers;
pub(crate) mod serde_helpers;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

use std::borrow::Cow;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use derive_builder::Builder;
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
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct Usage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
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
    pub fn zero() -> Usage {
        Usage {
            input_tokens: Some(0),
            output_tokens: Some(0),
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
                Some(self.tools.into_iter().collect())
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
