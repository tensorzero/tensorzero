//! Main TensorZero types for inference requests and responses
//!
//! During inference processing, we transform between several different input types:
//! * Input/InputMessage/InputMessageContent:
//!
//!   These types hold an input request deserialized directly from the client.
//!   At this point, we have not fetched any network resources (e.g. file urls),
//!   and we may have various legacy input types (e.g. `{"type": "text", "value": ...}`).
//!   Templates have not yet been applied
//! * `LazyResolvedInput`/`LazyResolvedInputMessage`/`LazyResolvedInputMessageContent`:
//!
//!   These types hold input with legacy input types normalized
//!   (e.g. a `{"type": "text", "value": ...}` block is converted to a `{"type": "text", "template": <role>, "arguments": {}}` block
//!   with the template name chosen based on the message role).
//!   We also construct (but do not yet `.await`) and store futures to fetch any file urls in the input.
//!   Templates have not yet been applied
//! * `ResolvedInput/ResolvedInputMessage/ResolvedInputMessageContent`:
//!
//!  These types are almost the same as the `LazyResolvedInput`/`LazyResolvedInputMessage`/`LazyResolvedInputMessageContent` types,
//!  but each file future is now resolved to an in-memory file. No network requests are needed to resolve any data
//!  within these input types.
//!  Templates have been not yet been applied.
//! * `RequestMessage/ContentBlock`:
//!
//!  These types hold input specialized for a particular variant.
//!  Templating has been applied, which prevents converting back to a `LazyResolvedInput`/`ResolvedInput` type.
//!  All files are fully resolved to in-memory files.
//! * `StoredInput/StoredInputMessage/StoredInputMessageContent`:
//!
//!  These types represent the actual data written to `ChatInference`/`JsonInference` in ClickHouse.
//!  Files are stored as object store paths, without the actual file contents (since we only write paths to ClickHouse)
//!  Templating has been applied.
//! * `StoredRequestMessage/StoredContentBlock`:
//!
//!  These types represent the actual data written to `ModelInference` in ClickHouse.
//!  Files are stored as object store paths, without the actual file contents (since we only write paths to ClickHouse)
//!  Templating has been applied.
//!
//! During normal inference processing, the types are transformed as:
//!
//!                                                   -> `RequestMessage` -> `StoredRequestMessage`
//! `Input` -> `LazyResolvedInput` -> `ResolvedInput`
//!                                                   -> `StoredInput`
//!
//! The upper branch (constructing a `RequestMessage`) is used when invoking a chat completion variant.
//! The lower branch (constructing a `StoredInput`) is used when we to write to `ChatInference`/`JsonInference` in ClickHouse.
use crate::db::postgres::PostgresConnectionInfo;
use crate::http::TensorzeroHttpClient;
use crate::inference::types::resolved_input::{
    FileUrl, LazyFile, LazyResolvedInput, LazyResolvedInputMessage, LazyResolvedInputMessageContent,
};
use crate::inference::types::stored_input::StoredFile;
use crate::rate_limiting::{
    get_estimated_tokens, RateLimitResourceUsage, RateLimitedInputContent, RateLimitedRequest,
    TicketBorrows,
};
use crate::serde_util::{
    deserialize_defaulted_json_string, deserialize_json_string, deserialize_optional_json_string,
};
use crate::tool::ToolCallInput;
use crate::variant::chat_completion::{ASSISTANT_TEXT_TEMPLATE_VAR, USER_TEXT_TEMPLATE_VAR};
use derive_builder::Builder;
use extra_body::{FullExtraBodyConfig, UnfilteredInferenceExtraBody};
use extra_headers::{FullExtraHeadersConfig, UnfilteredInferenceExtraHeaders};
use file::sanitize_raw_request;
pub use file::{Base64File, File};
use futures::future::{join_all, try_join_all};
use futures::stream::Peekable;
use futures::{FutureExt, Stream};
use indexmap::IndexMap;
use itertools::Itertools;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::types::PyAny;
#[cfg(feature = "pyo3")]
use pyo3_helpers::serialize_to_dict;
use resolved_input::FileWithPath;
pub use resolved_input::{ResolvedInput, ResolvedInputMessage, ResolvedInputMessageContent};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};
use serde_untagged::UntaggedEnumVisitor;
use std::borrow::Borrow;
use std::ops::Add;
use std::{
    borrow::Cow,
    collections::HashMap,
    pin::Pin,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use uuid::Uuid;

use crate::cache::NonStreamingCacheData;
use crate::{cache::CacheData, config::ObjectStoreInfo};
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
use serde::de::Error as _;

pub mod batch;
pub mod extra_body;
pub mod extra_headers;
pub mod file;
#[cfg(feature = "pyo3")]
pub mod pyo3_helpers;
pub mod resolved_input;
pub mod storage;
pub mod stored_input;

pub use resolved_input::ResolvedRequestMessage;
pub use stored_input::{
    StoredInput, StoredInputMessage, StoredInputMessageContent, StoredRequestMessage,
};

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

#[derive(Copy, Clone)]
pub struct FetchContext<'a> {
    pub client: &'a TensorzeroHttpClient,
    pub object_store_info: &'a Option<ObjectStoreInfo>,
}

impl Input {
    pub fn into_lazy_resolved_input(
        self,
        context: FetchContext<'_>,
    ) -> Result<LazyResolvedInput, Error> {
        Ok(LazyResolvedInput {
            system: self.system,
            messages: self
                .messages
                .into_iter()
                .map(|message| message.into_lazy_resolved_input_message(context))
                .collect::<Result<Vec<LazyResolvedInputMessage>, Error>>()?,
        })
    }
}

impl LazyResolvedInput {
    /// Resolves any nested network resources in the input.
    /// Currently, this resolves input image urls into base64-encoded images.
    pub async fn resolve(self) -> Result<ResolvedInput, Error> {
        let messages = futures::future::try_join_all(
            self.messages
                .into_iter()
                .map(resolved_input::LazyResolvedInputMessage::resolve),
        )
        .await?;
        Ok(ResolvedInput {
            system: self.system,
            messages,
        })
    }
}

impl InputMessage {
    pub fn into_lazy_resolved_input_message(
        self,
        context: FetchContext<'_>,
    ) -> Result<LazyResolvedInputMessage, Error> {
        Ok(LazyResolvedInputMessage {
            role: self.role,
            content: self
                .content
                .into_iter()
                .map(|content| content.into_lazy_resolved_input_message(self.role, context))
                .collect::<Result<Vec<LazyResolvedInputMessageContent>, Error>>()?,
        })
    }
}

impl LazyResolvedInputMessage {
    pub async fn resolve(self) -> Result<ResolvedInputMessage, Error> {
        let content = futures::future::try_join_all(
            self.content
                .into_iter()
                .map(resolved_input::LazyResolvedInputMessageContent::resolve),
        )
        .await?;
        Ok(ResolvedInputMessage {
            role: self.role,
            content,
        })
    }
}

impl InputMessageContent {
    /// The 'role' parameter is only used to handle legacy role-based templates (`{"type": "text", "value": ...}`).
    /// Once we removed support for these input blocks (and only support `{"type": "template", "name": "...", "arguments": ...}`),
    /// we can remove the 'role' parameter.
    pub fn into_lazy_resolved_input_message(
        self,
        role: Role,
        context: FetchContext<'_>,
    ) -> Result<LazyResolvedInputMessageContent, Error> {
        Ok(match self {
            InputMessageContent::Text(TextKind::Text { text }) => {
                LazyResolvedInputMessageContent::Text { text }
            }
            InputMessageContent::RawText { value } => {
                LazyResolvedInputMessageContent::RawText { value }
            }
            InputMessageContent::Thought(thought) => {
                LazyResolvedInputMessageContent::Thought(thought)
            }
            InputMessageContent::Template(template) => {
                LazyResolvedInputMessageContent::Template(template)
            }
            InputMessageContent::Text(TextKind::Arguments { arguments }) => {
                // Map the legacy `{{"type": "text", "arguments": ...}}` format to an explicit
                // `{{"type": "template", "name": "<role>", "arguments": ...}}` format, with the template
                // name chosen based on the message role.
                LazyResolvedInputMessageContent::Template(TemplateInput {
                    name: role.implicit_template_name().to_string(),
                    arguments,
                })
            }
            InputMessageContent::ToolCall(tool_call) => {
                LazyResolvedInputMessageContent::ToolCall(tool_call.try_into()?)
            }
            InputMessageContent::ToolResult(tool_result) => {
                LazyResolvedInputMessageContent::ToolResult(tool_result)
            }
            InputMessageContent::Text(TextKind::LegacyValue { value }) => {
                tracing::warn!(
                    r#"Deprecation Warning: `{{"type": "text", "value", ...}}` is deprecated. Please use `{{"type": "text", "text": "String input"}}` or `{{"type": "text", "arguments": {{..}}}} ` instead."#
                );
                match value {
                    Value::String(text) => LazyResolvedInputMessageContent::Text { text },
                    Value::Object(arguments) => {
                        LazyResolvedInputMessageContent::Template(TemplateInput {
                            name: role.implicit_template_name().to_string(),
                            arguments,
                        })
                    }
                    _ => {
                        return Err(Error::new(ErrorDetails::InvalidMessage {
                            message: r#"The 'value' field in a `{"type": "text", "value": ... }` content block must be a string or object"#.to_string(),
                        }));
                    }
                }
            }
            InputMessageContent::File(file) => {
                let storage_kind = context
                    .object_store_info
                    .as_ref()
                    .ok_or_else(|| {
                        Error::new(ErrorDetails::ObjectStoreUnconfigured {
                            block_type: "file".to_string(),
                        })
                    })?
                    .kind
                    .clone();
                match &file {
                    File::Url { url, mime_type } => {
                        // Check that we have an object store *outside* of the future that we're going to store in
                        // `LazyResolvedInputMessageContent::File`. We want to error immediately if the user tries
                        // to use a file input without explicitly configuring an object store (either explicit enabled or disabled)
                        let storage_kind = context
                            .object_store_info
                            .as_ref()
                            .ok_or_else(|| {
                                Error::new(ErrorDetails::ObjectStoreUnconfigured {
                                    block_type: "file".to_string(),
                                })
                            })?
                            .kind
                            .clone();
                        let client = context.client.clone();
                        // Construct a future that will actually fetch the file URL from the network.
                        // Important - we do *not* use `tokio::spawn` here. As a result, the future
                        // will not actually begin executing (including opening the network connection)
                        // until the first time the `Shared` wrapper is `.await`ed.
                        // This ensures that if we never actually need to download the file
                        // (due to model providers forwarding image urls, and object store observability being disabled),
                        // we will skip downloading the file entirely.
                        let url = url.clone();
                        let mime_type = mime_type.clone();
                        let delayed_file_future = async move {
                            let file = file.take_or_fetch(&client).await?;
                            let path = storage_kind.file_path(&file)?;
                            Ok(FileWithPath {
                                file,
                                storage_path: path,
                            })
                        };
                        LazyResolvedInputMessageContent::File(Box::new(LazyFile::Url {
                            file_url: FileUrl { url, mime_type },
                            future: delayed_file_future.boxed().shared(),
                        }))
                    }
                    File::Base64 { mime_type, data } => {
                        let file = Base64File {
                            url: None,
                            mime_type: mime_type.clone(),
                            data: data.clone(),
                        };

                        let path = storage_kind.file_path(&file)?;

                        LazyResolvedInputMessageContent::File(Box::new(LazyFile::FileWithPath(
                            FileWithPath {
                                file,
                                storage_path: path,
                            },
                        )))
                    }
                }
            }
            InputMessageContent::Unknown {
                data,
                model_provider_name,
            } => LazyResolvedInputMessageContent::Unknown {
                data,
                model_provider_name,
            },
        })
    }
}

impl LazyResolvedInputMessageContent {
    pub async fn resolve(self) -> Result<ResolvedInputMessageContent, Error> {
        Ok(match self {
            LazyResolvedInputMessageContent::Text { text } => {
                ResolvedInputMessageContent::Text { text }
            }
            LazyResolvedInputMessageContent::Template(template) => {
                ResolvedInputMessageContent::Template(template)
            }
            LazyResolvedInputMessageContent::ToolCall(tool_call) => {
                ResolvedInputMessageContent::ToolCall(tool_call)
            }
            LazyResolvedInputMessageContent::ToolResult(tool_result) => {
                ResolvedInputMessageContent::ToolResult(tool_result)
            }
            LazyResolvedInputMessageContent::RawText { value } => {
                ResolvedInputMessageContent::RawText { value }
            }
            LazyResolvedInputMessageContent::Thought(thought) => {
                ResolvedInputMessageContent::Thought(thought)
            }
            LazyResolvedInputMessageContent::File(file) => match *file {
                LazyFile::Url {
                    future,
                    file_url: _,
                } => ResolvedInputMessageContent::File(Box::new(future.await?)),
                LazyFile::FileWithPath(file) => ResolvedInputMessageContent::File(Box::new(file)),
            },
            LazyResolvedInputMessageContent::Unknown {
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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, ts_rs::TS)]
#[ts(export)]
#[serde(deny_unknown_fields)]
pub struct TemplateInput {
    pub name: String,
    pub arguments: Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputMessageContent {
    Text(TextKind),
    Template(TemplateInput),
    ToolCall(ToolCallInput),
    ToolResult(ToolResult),
    RawText {
        value: String,
    },
    Thought(Thought),
    #[serde(alias = "image")]
    File(File),
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
#[derive(ts_rs::TS)]
#[ts(export)]
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
                "Unknown key '{key}' in text content"
            ))),
        }
    }
}

#[derive(ts_rs::TS, Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[ts(export)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "pyo3", pyclass)]
pub enum Role {
    User,
    Assistant,
}

impl Role {
    /// The template name to use for `{"type": "text", "arguments": {}}` inputs.
    /// This will eventually be deprecated in favor of explicit `{"type": "template", "name": "user", "arguments": {}}` inputs.
    pub fn implicit_template_name(&self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }

    pub fn implicit_template_var(&self) -> &'static str {
        match self {
            Role::User => USER_TEXT_TEMPLATE_VAR,
            Role::Assistant => ASSISTANT_TEXT_TEMPLATE_VAR,
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
        }
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl Role {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}

/// InputMessages are validated against the input schema of the Function
/// and then templated and transformed into RequestMessages for a particular Variant.
/// They might contain tool calls or tool results along with text.
/// The abstraction we use to represent this is ContentBlock, which is a union of Text, ToolCall, and ToolResult.
/// ContentBlocks are collected into RequestMessages.
/// These RequestMessages are collected into a ModelInferenceRequest,
/// which should contain all information needed by a ModelProvider to perform the
/// inference that is called for.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(get_all, str))]
pub struct Text {
    pub text: String,
}

impl std::fmt::Display for Text {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text)
    }
}

impl RateLimitedInputContent for Text {
    fn estimated_input_token_usage(&self) -> u64 {
        let Text { text } = self;
        get_estimated_tokens(text)
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl Text {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}

/// Struct that represents Chain of Thought reasoning
#[derive(ts_rs::TS, Clone, Debug, Deserialize, PartialEq, Serialize)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(get_all))]
pub struct Thought {
    pub text: Option<String>,
    /// An optional signature - currently, this is only used with Anthropic,
    /// and is ignored by other providers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    /// When set, this 'Thought' block will only be used for providers
    /// matching this type (e.g. `anthropic`). Other providers will emit
    /// a warning and discard the block.
    #[serde(
        rename = "_internal_provider_type",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_type: Option<String>,
}

impl RateLimitedInputContent for Thought {
    fn estimated_input_token_usage(&self) -> u64 {
        let Thought {
            text,
            signature,
            provider_type: _,
        } = self;
        text.as_ref().map_or(0, |text| get_estimated_tokens(text))
            + signature
                .as_ref()
                .map_or(0, |signature| get_estimated_tokens(signature))
    }
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
    ToolResult(ToolResult),
    #[serde(alias = "image")]
    File(Box<LazyFile>),
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

impl ContentBlock {
    pub async fn into_stored_content_block(self) -> Result<StoredContentBlock, Error> {
        match self {
            ContentBlock::Text(text) => Ok(StoredContentBlock::Text(text)),
            ContentBlock::ToolCall(tool_call) => Ok(StoredContentBlock::ToolCall(tool_call)),
            ContentBlock::ToolResult(tool_result) => {
                Ok(StoredContentBlock::ToolResult(tool_result))
            }
            ContentBlock::File(file) => Ok(StoredContentBlock::File(Box::new(
                file.resolve()
                    .await?
                    .clone()
                    .into_owned()
                    .into_stored_file(),
            ))),
            ContentBlock::Thought(thought) => Ok(StoredContentBlock::Thought(thought)),
            ContentBlock::Unknown {
                data,
                model_provider_name,
            } => Ok(StoredContentBlock::Unknown {
                data,
                model_provider_name,
            }),
        }
    }

    pub async fn into_resolved_content_block(self) -> Result<ResolvedContentBlock, Error> {
        match self {
            ContentBlock::Text(text) => Ok(ResolvedContentBlock::Text(text)),
            ContentBlock::ToolCall(tool_call) => Ok(ResolvedContentBlock::ToolCall(tool_call)),
            ContentBlock::ToolResult(tool_result) => {
                Ok(ResolvedContentBlock::ToolResult(tool_result))
            }
            ContentBlock::File(file) => Ok(ResolvedContentBlock::File(Box::new(
                file.resolve().await?.clone().into_owned(),
            ))),
            ContentBlock::Thought(thought) => Ok(ResolvedContentBlock::Thought(thought)),
            ContentBlock::Unknown {
                data,
                model_provider_name,
            } => Ok(ResolvedContentBlock::Unknown {
                data,
                model_provider_name,
            }),
        }
    }
}

impl std::fmt::Display for ResolvedRequestMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl RateLimitedInputContent for ContentBlock {
    fn estimated_input_token_usage(&self) -> u64 {
        match self {
            ContentBlock::Text(text) => text.estimated_input_token_usage(),
            ContentBlock::ToolCall(tool_call) => tool_call.estimated_input_token_usage(),
            ContentBlock::ToolResult(tool_result) => tool_result.estimated_input_token_usage(),
            ContentBlock::File(file) => file.estimated_input_token_usage(),
            ContentBlock::Thought(thought) => thought.estimated_input_token_usage(),
            ContentBlock::Unknown { .. } => 0,
        }
    }
}

/// The version of `ContentBlock` that is stored in ClickHouse.
/// This is almost identical to `ContentBlock`, but without `File` data.
#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(test, ts(export))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StoredContentBlock {
    Text(Text),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    #[serde(alias = "image")]
    File(Box<StoredFile>),
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

/// Like `ContentBlock`, but stores an in-memory `FileWithPath` instead of a `LazyFile`
/// As a result, it can implement both `Serialize` and `Deserialize`
#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(test, ts(export))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResolvedContentBlock {
    Text(Text),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    File(Box<FileWithPath>),
    Thought(Thought),
    Unknown {
        data: Value,
        model_provider_name: Option<String>,
    },
}

impl ResolvedContentBlock {
    pub fn into_content_block(self) -> ContentBlock {
        match self {
            ResolvedContentBlock::Text(text) => ContentBlock::Text(text),
            ResolvedContentBlock::ToolCall(tool_call) => ContentBlock::ToolCall(tool_call),
            ResolvedContentBlock::ToolResult(tool_result) => ContentBlock::ToolResult(tool_result),
            ResolvedContentBlock::File(file) => {
                ContentBlock::File(Box::new(LazyFile::FileWithPath(*file)))
            }
            ResolvedContentBlock::Thought(thought) => ContentBlock::Thought(thought),
            ResolvedContentBlock::Unknown {
                data,
                model_provider_name,
            } => ContentBlock::Unknown {
                data,
                model_provider_name,
            },
        }
    }
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

/// Holds the variants types of `ContentBlockOutput` without any data
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ContentBlockOutputType {
    Text,
    ToolCall,
    Thought,
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
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
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
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(any(feature = "e2e_tests", test), derive(PartialEq))]
pub struct RequestMessage {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl RequestMessage {
    pub async fn into_stored_message(self) -> Result<StoredRequestMessage, Error> {
        Ok(StoredRequestMessage {
            role: self.role,
            content: try_join_all(
                self.content
                    .into_iter()
                    .map(ContentBlock::into_stored_content_block),
            )
            .await?,
        })
    }

    pub async fn into_resolved_message(self) -> Result<ResolvedRequestMessage, Error> {
        Ok(ResolvedRequestMessage {
            role: self.role,
            content: try_join_all(
                self.content
                    .into_iter()
                    .map(ContentBlock::into_resolved_content_block),
            )
            .await?,
        })
    }
}

impl RateLimitedInputContent for RequestMessage {
    fn estimated_input_token_usage(&self) -> u64 {
        let RequestMessage {
            #[expect(unused_variables)]
            role,
            content,
        } = self;
        content
            .iter()
            .map(RateLimitedInputContent::estimated_input_token_usage)
            .sum()
    }
}

impl std::fmt::Display for RequestMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FunctionType {
    #[default]
    Chat,
    Json,
}

impl FunctionType {
    pub fn inference_table_name(&self) -> &'static str {
        match self {
            FunctionType::Chat => "ChatInference",
            FunctionType::Json => "JsonInference",
        }
    }
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
#[derive(Builder, Clone, Debug, Default, Serialize)]
#[cfg_attr(any(feature = "e2e_tests", test), derive(PartialEq))]
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
    pub stop_sequences: Option<Cow<'a, [String]>>,
    pub stream: bool,
    pub json_mode: ModelInferenceRequestJsonMode,
    pub function_type: FunctionType,
    pub output_schema: Option<&'a Value>,
    pub extra_body: FullExtraBodyConfig,
    pub extra_headers: FullExtraHeadersConfig,
    /// Optional arbitrary data, only used when constructing the cache key.
    /// This is used by best_of_n/mixture_of_n to force different sub-variants
    /// to have different cache keys.
    pub extra_cache_key: Option<String>,
}

impl<'a> ModelInferenceRequest<'a> {
    pub fn borrow_stop_sequences(&'a self) -> Option<Cow<'a, [String]>> {
        self.stop_sequences.as_ref().map(borrow_cow)
    }
}

impl RateLimitedRequest for ModelInferenceRequest<'_> {
    fn estimated_resource_usage(&self) -> Result<RateLimitResourceUsage, Error> {
        let ModelInferenceRequest {
            inference_id: _,
            messages,
            system,
            tool_config: _, // TODO: should we account for this in advance?
            temperature: _,
            top_p: _,
            max_tokens,
            presence_penalty: _,
            frequency_penalty: _,
            seed: _,
            stop_sequences: _,
            stream: _,
            json_mode: _,
            function_type: _,
            output_schema: _,
            extra_body: _,
            extra_headers: _,
            extra_cache_key: _,
        } = self;
        let system_tokens = system
            .as_ref()
            .map(|s| get_estimated_tokens(s))
            .unwrap_or(0);
        let messages_tokens: u64 = messages
            .iter()
            .map(RateLimitedInputContent::estimated_input_token_usage)
            .sum();
        let output_tokens =
            max_tokens.ok_or_else(|| Error::new(ErrorDetails::RateLimitMissingMaxTokens))? as u64;
        Ok(RateLimitResourceUsage {
            tokens: system_tokens + messages_tokens + output_tokens,
            model_inferences: 1,
        })
    }
}

/// For use in rendering for optimization purposes
#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(any(feature = "e2e_tests", test), derive(PartialEq))]
#[cfg_attr(test, ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(get_all, str))]
pub struct ModelInput {
    pub system: Option<String>,
    pub messages: Vec<ResolvedRequestMessage>,
}

impl std::fmt::Display for ModelInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl ModelInput {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    StopSequence,
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

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(any(feature = "e2e_tests", test), derive(PartialEq))]
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

impl ProviderInferenceResponse {
    pub fn resource_usage(&self) -> Result<RateLimitResourceUsage, Error> {
        Ok(RateLimitResourceUsage {
            model_inferences: 1,
            tokens: self.usage.total_tokens() as u64,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl Usage {
    pub fn total_tokens(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }
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
#[derive(Clone, Debug)]
#[cfg_attr(any(feature = "e2e_tests", test), derive(PartialEq))]
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
#[derive(Clone, Debug)]
#[cfg_attr(any(feature = "e2e_tests", test), derive(PartialEq))]
pub struct ModelInferenceResponseWithMetadata {
    pub id: Uuid,
    pub created: u64,
    pub output: Vec<ContentBlockOutput>,
    pub system: Option<String>,
    pub input_messages: RequestMessagesOrBatch,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
    pub model_provider_name: Arc<str>,
    pub model_name: Arc<str>,
    pub cached: bool,
    pub finish_reason: Option<FinishReason>,
}

/// Holds `RequestMessage`s or `StoredRequestMessage`s. This used to avoid the need to duplicate types
/// that are used by batch inferences (where we read `StoredRequestMessage` from the database)
/// and normal inference code (where we pass around `RequestMessage`s in order to strip out image data
/// from our `raw_request` before writing to ClickHouse).
///
/// This is separate from any 're-resolution' logic (converting from a `StoredRequestMessage`
/// back to a `RequestMessage` by looking up image data from the object store).
/// We don't currently have re-resolution implemented in Rust, but we'll need to do so when
/// we move more ui code to Rust
#[derive(Clone, Debug)]
#[cfg_attr(any(feature = "e2e_tests", test), derive(PartialEq))]
pub enum RequestMessagesOrBatch {
    /// The typical case - we have normal `RequestMessages` from client input
    Message(Vec<RequestMessage>),
    /// We've deserialized `StoredRequestMessage` from a batch inference row in the databsae
    /// This is only used when constructing our final result in `write_completed_batch_inference`
    BatchInput(Vec<StoredRequestMessage>),
}

impl ModelInferenceResponseWithMetadata {
    /// We return the actual usage (meaning the number of tokens the user would be billed for)
    /// in the HTTP response.
    /// However, we store the number of tokens that would have been used in the database.
    /// So we need this function to compute the actual usage in order to send it in the HTTP response.
    pub fn usage_considering_cached(&self) -> Usage {
        if self.cached {
            Usage {
                input_tokens: 0,
                output_tokens: 0,
            }
        } else {
            self.usage
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
    pub created: u64,
    pub content: Vec<ContentBlockChatOutput>,
    pub model_inference_results: Vec<ModelInferenceResponseWithMetadata>,
    pub inference_params: InferenceParams,
    pub original_response: Option<String>,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Clone, Debug)]
pub struct JsonInferenceResult {
    pub inference_id: Uuid,
    pub created: u64,
    pub output: InternalJsonInferenceOutput,
    pub model_inference_results: Vec<ModelInferenceResponseWithMetadata>,
    pub output_schema: Value,
    pub inference_params: InferenceParams,
    pub original_response: Option<String>,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, ts_rs::TS)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct JsonInferenceOutput {
    pub raw: Option<String>,
    pub parsed: Option<Value>,
}

impl std::fmt::Display for JsonInferenceOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl JsonInferenceOutput {
    #[getter]
    fn get_raw(&self) -> Option<String> {
        self.raw.clone()
    }

    #[getter]
    fn get_parsed<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match &self.parsed {
            Some(value) => serialize_to_dict(py, value)?.into_bound(py),
            None => py.None().into_bound(py),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InternalJsonInferenceOutput {
    pub raw: Option<String>,
    pub parsed: Option<Value>,
    pub auxiliary_content: Vec<ContentBlockOutput>,
    // index of the JSON block in the original content blocks
    // generated by the inference
    pub json_block_index: Option<usize>,
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
    /// See `Thought.provider_type`
    #[serde(
        rename = "_internal_provider_type",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_type: Option<String>,
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
    pub input: StoredInput,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub output: Vec<ContentBlockChatOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(deserialize_with = "deserialize_optional_json_string")]
    pub tool_params: Option<ToolCallConfigDatabaseInsert>,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub inference_params: InferenceParams,
    pub processing_time_ms: Option<u32>,
    pub ttft_ms: Option<u32>,
    pub tags: HashMap<String, String>,
    #[serde(deserialize_with = "deserialize_defaulted_json_string")]
    pub extra_body: UnfilteredInferenceExtraBody,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct JsonInferenceDatabaseInsert {
    pub id: Uuid,
    pub function_name: String,
    pub variant_name: String,
    pub episode_id: Uuid,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub input: StoredInput,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub output: JsonInferenceOutput,
    // We at one point wrote empty auxiliary content to the database as "" but now write it as []
    // In either case, we want to deserialize it as [] if empty
    #[serde(deserialize_with = "deserialize_defaulted_json_string")]
    pub auxiliary_content: Vec<ContentBlockOutput>,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub inference_params: InferenceParams,
    pub processing_time_ms: Option<u32>,
    pub output_schema: Value,
    pub ttft_ms: Option<u32>,
    pub tags: HashMap<String, String>,
    #[serde(deserialize_with = "deserialize_defaulted_json_string")]
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
        ResolvedInputMessageContent::Text { text }
    }
}

#[cfg(test)]
impl From<String> for LazyResolvedInputMessageContent {
    fn from(text: String) -> Self {
        LazyResolvedInputMessageContent::Text { text }
    }
}

#[cfg(any(test, feature = "e2e_tests"))]
impl From<String> for ContentBlockChatOutput {
    fn from(text: String) -> Self {
        ContentBlockChatOutput::Text(Text { text })
    }
}

fn deserialize_content<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<InputMessageContent>, D::Error> {
    #[expect(clippy::redundant_closure_for_method_calls)]
    UntaggedEnumVisitor::new()
        .string(|text| {
            Ok(vec![InputMessageContent::Text(TextKind::Text {
                text: text.to_string(),
            })])
        })
        .map(|object| {
            tracing::warn!("Deprecation Warning: passing in an object for `content` is deprecated. Please use an array of content blocks instead.");
            Ok(vec![InputMessageContent::Text(TextKind::Arguments {
                arguments: object.deserialize()?,
            })])
        })
        .seq(|seq| seq.deserialize())
        .deserialize(deserializer)
}

impl From<String> for ContentBlock {
    fn from(text: String) -> Self {
        ContentBlock::Text(Text { text })
    }
}

impl From<String> for StoredContentBlock {
    fn from(text: String) -> Self {
        StoredContentBlock::Text(Text { text })
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
            input_messages: request.messages.clone(),
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
            input_messages: RequestMessagesOrBatch::Message(
                model_inference_response.input_messages,
            ),
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
    pub async fn new(
        result: ModelInferenceResponseWithMetadata,
        inference_id: Uuid,
    ) -> Result<Self, Error> {
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

        let stored_input_messages = match result.input_messages {
            RequestMessagesOrBatch::Message(input_messages) => {
                // In the the future, we might want to support writing 'partially broken' input messages to ClickHouse,
                // so that we write something even if one of the input files fails to resolve.
                try_join_all(
                    input_messages
                        .into_iter()
                        .map(RequestMessage::into_stored_message),
                )
                .await?
            }
            RequestMessagesOrBatch::BatchInput(stored) => stored,
        };

        Ok(Self {
            id: Uuid::now_v7(),
            inference_id,
            raw_request: result.raw_request,
            raw_response: result.raw_response,
            system: result.system,
            output: serialized_output,
            input_tokens,
            output_tokens,
            response_time_ms: latency_ms,
            ttft_ms,
            model_provider_name: result.model_provider_name.to_string(),
            model_name: result.model_name.to_string(),
            cached: result.cached,
            finish_reason: result.finish_reason,
            input_messages: serialize_or_log(&stored_input_messages),
        })
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

    pub async fn get_serialized_model_inferences(&self) -> Vec<serde_json::Value> {
        let model_inference_responses = self.model_inference_results();
        let inference_id = match self {
            InferenceResult::Chat(chat_result) => chat_result.inference_id,
            InferenceResult::Json(json_result) => json_result.inference_id,
        };
        join_all(model_inference_responses.iter().map(|r| async {
            let model_inference = ModelInferenceDatabaseInsert::new(r.clone(), inference_id).await;
            let model_inference = match model_inference {
                Ok(model_inference) => model_inference,
                Err(e) => {
                    ErrorDetails::Serialization {
                        message: format!("Failed to construct ModelInferenceDatabaseInsert: {e:?}"),
                    }
                    .log();
                    return Default::default();
                }
            };
            match serde_json::to_value(model_inference) {
                Ok(v) => v,
                Err(e) => {
                    ErrorDetails::Serialization {
                        message: format!("Failed to serialize ModelInferenceDatabaseInsert: {e:?}"),
                    }
                    .log();
                    Default::default()
                }
            }
        }))
        .await
    }

    pub fn usage_considering_cached(&self) -> Usage {
        self.model_inference_results()
            .iter()
            .map(ModelInferenceResponseWithMetadata::usage_considering_cached)
            .sum()
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
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        inference_id: Uuid,
        raw: Option<String>,
        parsed: Option<Value>,
        json_block_index: Option<usize>,
        auxiliary_content: Vec<ContentBlockOutput>,
        model_inference_results: Vec<ModelInferenceResponseWithMetadata>,
        output_schema: Value,
        inference_params: InferenceParams,
        original_response: Option<String>,
    ) -> Self {
        let output = InternalJsonInferenceOutput {
            raw,
            parsed,
            auxiliary_content,
            json_block_index,
        };
        let finish_reason = get_finish_reason(&model_inference_results);
        Self {
            inference_id,
            created: current_timestamp(),
            output,
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
        .and_then(|r| r.finish_reason)
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
        input: StoredInput,
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
            ttft_ms: metadata.ttft_ms,
            extra_body: metadata.extra_body,
        }
    }
}

impl JsonInferenceDatabaseInsert {
    pub fn new(
        json_result: JsonInferenceResult,
        input: StoredInput,
        metadata: InferenceDatabaseInsertMetadata,
    ) -> Self {
        let processing_time_ms = metadata
            .processing_time
            .map(|duration| duration.as_millis() as u32);

        let inference_params = json_result.inference_params;
        let InternalJsonInferenceOutput {
            raw,
            parsed,
            auxiliary_content,
            ..
        } = json_result.output;
        let output = JsonInferenceOutput { raw, parsed };

        Self {
            id: json_result.inference_id,
            function_name: metadata.function_name,
            variant_name: metadata.variant_name,
            episode_id: metadata.episode_id,
            input,
            auxiliary_content,
            inference_params,
            output,
            processing_time_ms,
            output_schema: json_result.output_schema,
            tags: metadata.tags,
            extra_body: metadata.extra_body,
            ttft_ms: metadata.ttft_ms,
        }
    }
}

// Function to get the current timestamp in seconds
#[expect(clippy::missing_panics_doc)]
pub fn current_timestamp() -> u64 {
    #[expect(clippy::expect_used)]
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

impl From<ContentBlockChatOutput> for ContentBlockOutput {
    fn from(output: ContentBlockChatOutput) -> Self {
        match output {
            ContentBlockChatOutput::Text(text) => ContentBlockOutput::Text(text),
            ContentBlockChatOutput::ToolCall(tool_call) => {
                ContentBlockOutput::ToolCall(tool_call.into())
            }
            ContentBlockChatOutput::Thought(thought) => ContentBlockOutput::Thought(thought),
            ContentBlockChatOutput::Unknown {
                data,
                model_provider_name,
            } => ContentBlockOutput::Unknown {
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
                    raw = Some(tool_call.raw_arguments.to_owned());
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
    /// We may sometimes construct a fake stream from a non-streaming response
    /// (e.g. in `mixture_of_n` if we have a successful non-streaming candidate, but
    /// a streaming fuser request fails).
    /// In this case, we want to store the original `raw_response`, instead of building
    /// it up from the chunks.
    pub raw_response: Option<String>,
    pub inference_params: InferenceParams,
    pub system: Option<String>,
    pub input_messages: Vec<RequestMessage>,
    pub function_name: &'b str,
    pub variant_name: &'b str,
    pub dynamic_output_schema: Option<DynamicJSONSchema>,
    pub templates: &'b TemplateConfig<'a>,
    pub tool_config: Option<&'b ToolCallConfig>,
    pub cached: bool,
    pub extra_body: UnfilteredInferenceExtraBody,
    pub extra_headers: UnfilteredInferenceExtraHeaders,
    pub ticket_borrow: TicketBorrows,
    pub postgres_connection_info: PostgresConnectionInfo,
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
        raw_response,
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
        extra_headers,
        ticket_borrow,
        postgres_connection_info,
    } = args;

    // NOTE: We will eventually need this to be per-inference-response-type and sensitive to the type of variant and function being called.
    // We preserve the order of chunks in the stream when combining them into `ContentBlockOutput`, except when
    // the same id is used by non-adjacent blocks.
    // For example, the following chunks:
    // `[TextChunk(id=0, content="Hello ""), ThoughtChunk(id=0, content=Something), TextChunk(id=0, content=World)]``
    // will be collected into the content block list: `[Text("Hello World"), Thought("Something"))]`
    //
    // All chunks with the same type and id (in this case, TextChunk id=0) are combined into a single content
    // block at the first occurrence of that type and id.
    // We use an 'IndexMap' to preserve the insertion order, so that newly-seen type/id combinations
    // are not reordered.
    let mut blocks: IndexMap<(ContentBlockOutputType, String), ContentBlockOutput> =
        IndexMap::new();
    // If the variant gave us an explicit 'raw_response', use that.
    // Otherwise, concatenate the raw_response from each chunk.
    let raw_response = raw_response.unwrap_or_else(|| {
        value
            .iter()
            .map(InferenceResultChunk::raw_response)
            .collect::<Vec<&str>>()
            .join("\n")
    });
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
                                &mut blocks,
                                (ContentBlockOutputType::Text, text.id),
                                text.text,
                                &mut ttft,
                                chunk.latency,
                                Into::into,
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
                                    &mut blocks,
                                    (ContentBlockOutputType::Thought, thought.id.clone()),
                                    text,
                                    &mut ttft,
                                    chunk.latency,
                                    |text| {
                                        ContentBlockOutput::Thought(Thought {
                                            text: Some(text),
                                            signature: None,
                                            provider_type: thought.provider_type.clone(),
                                        })
                                    },
                                    |block, text| {
                                        if let ContentBlockOutput::Thought(thought) = block {
                                            thought.text.get_or_insert_default().push_str(text);
                                        }
                                    },
                                );
                            }
                            if let Some(signature) = thought.signature {
                                handle_textual_content_block(
                                    &mut blocks,
                                    (ContentBlockOutputType::Thought, thought.id),
                                    signature,
                                    &mut ttft,
                                    chunk.latency,
                                    |signature| {
                                        ContentBlockOutput::Thought(Thought {
                                            text: None,
                                            signature: Some(signature),
                                            provider_type: thought.provider_type,
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
                            match blocks
                                .get_mut(&(ContentBlockOutputType::ToolCall, tool_call.id.clone()))
                            {
                                // If there is already a tool call block with this id, append to it
                                Some(ContentBlockOutput::ToolCall(existing_tool_call)) => {
                                    // We assume that the ID is present and complete in the first chunk
                                    // and that the name and arguments are accumulated with more chunks.
                                    if let Some(raw_name) = tool_call.raw_name {
                                        existing_tool_call.name.push_str(&raw_name);
                                    }
                                    existing_tool_call
                                        .arguments
                                        .push_str(&tool_call.raw_arguments);
                                }
                                // If there is no tool call block, create one
                                _ => {
                                    if ttft.is_none() {
                                        ttft = Some(chunk.latency);
                                    }
                                    blocks.insert(
                                        (ContentBlockOutputType::ToolCall, tool_call.id.clone()),
                                        ContentBlockOutput::ToolCall(tool_call_chunk_to_tool_call(
                                            tool_call,
                                        )),
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
                match blocks.get_mut(&(ContentBlockOutputType::Text, String::new())) {
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
                            blocks
                                .insert((ContentBlockOutputType::Text, String::new()), raw.into());
                        }
                    }
                }
                if let Some(thought) = chunk.thought {
                    match blocks.get_mut(&(ContentBlockOutputType::Thought, String::new())) {
                        // If there is already a thought block, append to it
                        Some(ContentBlockOutput::Thought(existing_thought)) => {
                            existing_thought
                                .text
                                .get_or_insert_default()
                                .push_str(&thought);
                        }
                        // If there is no thought block, create one
                        _ => {
                            blocks.insert(
                                (ContentBlockOutputType::Thought, String::new()),
                                ContentBlockOutput::Thought(Thought {
                                    text: Some(thought),
                                    signature: None,
                                    provider_type: None,
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
    let content_blocks: Vec<_> = blocks.into_values().collect();
    let model_response = ProviderInferenceResponse::new(ProviderInferenceResponseArgs {
        output: content_blocks.clone(),
        system,
        input_messages,
        raw_request,
        raw_response,
        usage,
        latency: latency.clone(),
        finish_reason,
    });
    if let Ok(actual_resource_usage) = model_response.resource_usage() {
        tokio::spawn(async move {
            if let Err(e) = ticket_borrow
                .return_tickets(&postgres_connection_info, actual_resource_usage)
                .await
            {
                tracing::error!("Failed to return rate limit tickets: {}", e);
            }
        });
    }
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
        variant_name,
        tool_config,
        templates,
        dynamic_output_schema: dynamic_output_schema.as_ref(),
        extra_body: Cow::Borrowed(&extra_body),
        extra_headers: Cow::Borrowed(&extra_headers),
        extra_cache_key: None,
    };
    function
        .prepare_response(
            inference_id,
            content_blocks,
            vec![model_inference_result],
            &inference_config,
            inference_params,
            Some(original_response),
        )
        .await
}

fn tool_call_chunk_to_tool_call(tool_call: ToolCallChunk) -> ToolCall {
    ToolCall {
        id: tool_call.id,
        name: tool_call.raw_name.unwrap_or_default(), // Since we are accumulating tool call names, we can start with "" if missing and hopefully accumulate with more chunks.
        arguments: tool_call.raw_arguments,
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

impl Add for Usage {
    type Output = Usage;
    fn add(self, other: Usage) -> Usage {
        Usage {
            input_tokens: self.input_tokens.saturating_add(other.input_tokens),
            output_tokens: self.output_tokens.saturating_add(other.output_tokens),
        }
    }
}
impl std::iter::Sum<Usage> for Usage {
    fn sum<I: Iterator<Item = Usage>>(iter: I) -> Self {
        iter.fold(Usage::default(), |acc, u| acc + u)
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
                message: format!("Failed to serialize value: {e}"),
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
    blocks: &mut IndexMap<(ContentBlockOutputType, String), ContentBlockOutput>,
    key: (ContentBlockOutputType, String),
    text: String,
    ttft: &mut Option<Duration>,
    chunk_latency: Duration,
    create_block: F,
    append_text: A,
) where
    F: FnOnce(String) -> ContentBlockOutput,
    A: FnOnce(&mut ContentBlockOutput, &str),
{
    match blocks.get_mut(&key) {
        // If there is already a block, append to it
        Some(existing_block) => append_text(existing_block, &text),
        // If there is no block, create one
        _ => {
            // We only want to set TTFT if there is some real content
            if ttft.is_none() {
                *ttft = Some(chunk_latency);
            }
            if !text.is_empty() {
                blocks.insert(key, create_block(text));
            }
        }
    }
}

/// Turns a reference to a Cow into a `Cow::Borrowed`, without cloning
fn borrow_cow<'a, T: ToOwned + ?Sized>(cow: &'a Cow<'a, T>) -> Cow<'a, T> {
    match cow {
        Cow::Borrowed(x) => Cow::Borrowed(x),
        Cow::Owned(x) => Cow::Borrowed(x.borrow()),
    }
}

pub(super) fn serialize_delete<S>(s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    true.serialize(s)
}

pub(super) fn deserialize_delete<'de, D>(d: D) -> Result<(), D::Error>
where
    D: Deserializer<'de>,
{
    let val = bool::deserialize(d)?;
    if !val {
        return Err(D::Error::custom(
            "Error deserializing replacement config: 'delete' must be 'true', or not set",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::config::SchemaData;
    use crate::function::{FunctionConfigChat, FunctionConfigJson};
    use crate::jsonschema_util::StaticJSONSchema;
    use crate::minijinja_util::TemplateConfig;
    use crate::providers::test_helpers::get_temperature_tool_config;
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
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            output: content.clone(),
            raw_request: raw_request.clone(),
            raw_response: String::new(),
            usage,
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
            model_inference_responses,
            None,
            InferenceParams::default(),
            None,
        )
        .await;
        let output_content = ["Hello, world!".to_string().into()];
        assert_eq!(chat_inference_response.content, output_content);
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
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            output: content.clone(),
            raw_request: raw_request.clone(),
            raw_response: String::new(),
            usage,
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
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            output: content.clone(),
            raw_request: raw_request.clone(),
            raw_response: String::new(),
            finish_reason: Some(FinishReason::Stop),
            usage,
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
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            output: content.clone(),
            raw_request: raw_request.clone(),
            raw_response: String::new(),
            usage,
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
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            output: content.clone(),
            finish_reason: None,
            raw_request: raw_request.clone(),
            raw_response: String::new(),
            usage,
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
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            output: content.clone(),
            finish_reason: None,
            raw_request: raw_request.clone(),
            raw_response: String::new(),
            usage,
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
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            output: content.clone(),
            finish_reason: None,
            raw_request: raw_request.clone(),
            raw_response: String::new(),
            usage,
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
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            output: content.clone(),
            finish_reason: None,
            raw_request: raw_request.clone(),
            raw_response: String::new(),
            usage,
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
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            output: content.clone(),
            raw_request: raw_request.clone(),
            raw_response: String::new(),
            usage,
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
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            output: content.clone(),
            finish_reason: None,
            raw_request: raw_request.clone(),
            raw_response: String::new(),
            usage,
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
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "",
            variant_name: "",
            dynamic_output_schema: None,
            templates: &templates,
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            ticket_borrow: TicketBorrows::empty(),
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
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "",
            variant_name: "",
            dynamic_output_schema: None,
            templates: &templates,
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            ticket_borrow: TicketBorrows::empty(),
        };
        let result = collect_chunks(collect_chunks_args).await.unwrap();
        let chat_result = match result {
            InferenceResult::Chat(chat_result) => chat_result,
            InferenceResult::Json(_) => panic!("Expected Chat inference response"),
        };
        assert_eq!(chat_result.inference_id, inference_id);
        // We make a new timestamp for `chat_result.created`, so just check that it's at least
        // the timestamp of the first chunk.
        assert!(
            chat_result.created >= created,
            "Chat result was created at {:?}, before the first chunk was created at {:?}",
            chat_result.created,
            created
        );
        assert_eq!(chat_result.finish_reason, Some(FinishReason::Stop));
        assert_eq!(
            chat_result.content,
            vec!["Hello, world!".to_string().into()]
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
        let output_schema = StaticJSONSchema::from_value(output_schema).unwrap();
        let json_function_config = Arc::new(FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            implicit_tool_call_config,
            output_schema,
            description: None,
            all_explicit_template_names: HashSet::new(),
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
                usage: Some(usage1),
                raw_response: "{\"name\":".to_string(),
                latency: Duration::from_millis(150),
                finish_reason: Some(FinishReason::ToolCall),
            }),
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                raw: Some("\"John\",\"age\":30}".to_string()),
                thought: Some("Thought 2".to_string()),
                created,
                usage: Some(usage2),
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
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "",
            variant_name: "",
            dynamic_output_schema: None,
            templates: &templates,
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            ticket_borrow: TicketBorrows::empty(),
        };
        let response = collect_chunks(collect_chunks_args).await.unwrap();
        assert_eq!(
            response.usage_considering_cached(),
            Usage {
                input_tokens: 15,
                output_tokens: 15,
            }
        );
        match response {
            InferenceResult::Json(json_result) => {
                assert_eq!(json_result.inference_id, inference_id);
                assert_eq!(
                    json_result.output.parsed,
                    Some(serde_json::json!({"name": "John", "age": 30}))
                );
                assert_eq!(
                    json_result.output.raw,
                    Some("{\"name\":\"John\",\"age\":30}".to_string())
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
            InferenceResult::Chat(_) => panic!("Expected Json inference response"),
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
                usage: Some(usage),
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
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "",
            variant_name: "",
            dynamic_output_schema: None,
            templates: &templates,
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            ticket_borrow: TicketBorrows::empty(),
        };
        let result = collect_chunks(collect_chunks_args).await.unwrap();
        assert_eq!(result.usage_considering_cached(), usage);
        match result {
            InferenceResult::Json(json_result) => {
                assert_eq!(json_result.inference_id, inference_id);
                // We make a new timestamp for `json_result.created`, so just check that it's at least
                // the timestamp of the first chunk.
                assert!(
                    json_result.created >= created,
                    "Json result was created at {:?}, before the first chunk was created at {:?}",
                    json_result.created,
                    created
                );
                assert_eq!(json_result.output.parsed, None);
                assert_eq!(
                    json_result.output.raw,
                    Some("{\"name\":\"John\"}".to_string())
                );
                assert_eq!(json_result.model_inference_results.len(), 1);
                let model_inference_result = json_result.model_inference_results.first().unwrap();
                assert_eq!(&*model_inference_result.model_name, model_name);
                assert_eq!(
                    &*model_inference_result.model_provider_name,
                    model_provider_name
                );
                assert_eq!(model_inference_result.raw_request, raw_request);
            }
            InferenceResult::Chat(_) => panic!("Expected Json inference response"),
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
                usage: Some(usage),
                raw_response: "{\"name\":\"John\",".to_string(),
                latency: Duration::from_millis(100),
                finish_reason: None,
            }),
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                raw: Some(String::new()),
                thought: Some("Thought 2".to_string()),
                created,
                usage: None,
                raw_response: String::new(),
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
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "",
            variant_name: "",
            dynamic_output_schema: None,
            templates: &templates,
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            ticket_borrow: TicketBorrows::empty(),
        };
        let result = collect_chunks(collect_chunks_args).await.unwrap();
        assert_eq!(result.usage_considering_cached(), usage);
        match result {
            InferenceResult::Chat(chat_response) => {
                assert_eq!(chat_response.inference_id, inference_id);
                // We make a new timestamp for `chat_response.created`, so just check that it's at least
                // the timestamp of the first chunk.
                assert!(
                    chat_response.created >= created,
                    "Chat result was created at {:?}, before the first chunk was created at {:?}",
                    chat_response.created,
                    created
                );
                assert_eq!(
                    chat_response.content,
                    vec![
                        ContentBlockChatOutput::Text(Text {
                            text: "{\"name\":\"John\",\"age\":30}".to_string()
                        }),
                        ContentBlockChatOutput::Thought(Thought {
                            text: Some("Thought 2".to_string()),
                            signature: None,
                            provider_type: None,
                        }),
                    ]
                );
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
            }
            InferenceResult::Json(_) => panic!("Expected Chat inference response"),
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
        let output_schema = StaticJSONSchema::from_value(output_schema).unwrap();
        let json_function_config = Arc::new(FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            implicit_tool_call_config,
            output_schema,
            description: None,
            all_explicit_template_names: HashSet::new(),
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
                usage: Some(usage1),
                raw_response: "{\"name\":".to_string(),
                latency: Duration::from_millis(150),
                finish_reason: Some(FinishReason::ToolCall),
            }),
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                raw: Some("\"John\",\"age\":30}".to_string()),
                thought: Some("Thought 2".to_string()),
                created,
                usage: Some(usage2),
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
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "",
            variant_name: "",
            dynamic_output_schema: None,
            templates: &templates,
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            ticket_borrow: TicketBorrows::empty(),
        };
        let response = collect_chunks(collect_chunks_args).await.unwrap();
        assert_eq!(
            response.usage_considering_cached(),
            Usage {
                input_tokens: 15,
                output_tokens: 15,
            }
        );
        match response {
            InferenceResult::Json(json_result) => {
                assert_eq!(json_result.inference_id, inference_id);
                assert_eq!(
                    json_result.output.parsed,
                    Some(serde_json::json!({"name": "John", "age": 30}))
                );
                assert_eq!(
                    json_result.output.raw,
                    Some("{\"name\":\"John\",\"age\":30}".to_string())
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
            InferenceResult::Chat(_) => panic!("Expected Json inference response"),
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
        let output_schema = StaticJSONSchema::from_value(static_output_schema).unwrap();
        let json_function_config = Arc::new(FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            implicit_tool_call_config,
            output_schema,
            description: None,
            all_explicit_template_names: HashSet::new(),
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
                usage: Some(usage1),
                finish_reason: Some(FinishReason::Stop),
                raw_response: "{\"name\":".to_string(),
                latency: Duration::from_millis(150),
            }),
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                raw: Some("\"John\",\"age\":30}".to_string()),
                thought: Some("Thought 2".to_string()),
                created,
                usage: Some(usage2),
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
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "",
            variant_name: "",
            dynamic_output_schema: Some(dynamic_output_schema),
            templates: &templates,
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            ticket_borrow: TicketBorrows::empty(),
        };
        let response = collect_chunks(collect_chunks_args).await.unwrap();
        assert_eq!(
            response.usage_considering_cached(),
            Usage {
                input_tokens: 15,
                output_tokens: 15,
            }
        );
        match response {
            InferenceResult::Json(json_result) => {
                assert_eq!(json_result.inference_id, inference_id);
                assert_eq!(
                    json_result.output.parsed,
                    Some(serde_json::json!({"name": "John", "age": 30}))
                );
                assert_eq!(
                    json_result.output.raw,
                    Some("{\"name\":\"John\",\"age\":30}".to_string())
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
            InferenceResult::Chat(_) => panic!("Expected Json inference response"),
        }
    }

    #[tokio::test]
    async fn test_collect_interleaved_chunks() {
        let templates = TemplateConfig::default();
        let function_config = Arc::new(FunctionConfig::Chat(FunctionConfigChat::default()));
        let model_name = "test_model";
        let model_provider_name = "test_provider";
        let raw_request = "raw request".to_string();
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let created = current_timestamp();
        let latency = Duration::from_millis(150);
        let chunks = vec![
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![
                    ContentBlockChunk::Text(TextChunk {
                        text: "Hello ".to_string(),
                        id: "0".to_string(),
                    }),
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "0".to_string(),
                        raw_name: Some("my_tool_call".to_string()),
                        raw_arguments: "true".to_string(),
                    }),
                ],
                created,
                usage: None,
                raw_response: "{\"message\": \"Hello}".to_string(),
                latency,
                finish_reason: None,
            }),
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![
                    ContentBlockChunk::Thought(ThoughtChunk {
                        text: Some("Some thou".to_string()),
                        id: "0".to_string(),
                        signature: None,
                        provider_type: None,
                    }),
                    ContentBlockChunk::Thought(ThoughtChunk {
                        text: Some("My other interleaved thought".to_string()),
                        id: "1".to_string(),
                        signature: None,
                        provider_type: None,
                    }),
                ],
                created,
                usage: None,
                raw_response: "my raw thought".to_string(),
                latency,
                finish_reason: None,
            }),
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![ContentBlockChunk::Text(TextChunk {
                    text: "world!".to_string(),
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
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![ContentBlockChunk::Thought(ThoughtChunk {
                    text: Some("ght".to_string()),
                    id: "0".to_string(),
                    signature: None,
                    provider_type: None,
                })],
                created,
                usage: None,
                raw_response: "my other raw thought".to_string(),
                latency,
                finish_reason: None,
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
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "",
            variant_name: "",
            dynamic_output_schema: None,
            templates: &templates,
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            ticket_borrow: TicketBorrows::empty(),
        };
        let result = collect_chunks(collect_chunks_args).await.unwrap();
        assert_eq!(
            result.usage_considering_cached(),
            Usage {
                input_tokens: 2,
                output_tokens: 4,
            }
        );
        let chat_result = match result {
            InferenceResult::Chat(chat_result) => chat_result,
            InferenceResult::Json(_) => panic!("Expected Chat inference response"),
        };
        assert_eq!(chat_result.inference_id, inference_id);
        // We make a new timestamp for `chat_result.created`, so just check that it's at least
        // the timestamp of the first chunk.
        assert!(
            chat_result.created >= created,
            "Chat result was created at {:?}, before the first chunk was created at {:?}",
            chat_result.created,
            created
        );
        assert_eq!(chat_result.finish_reason, Some(FinishReason::Stop));

        let expected_content = vec![
            ContentBlockChatOutput::Text(Text {
                text: "Hello world!".to_string(),
            }),
            ContentBlockChatOutput::ToolCall(ToolCallOutput {
                name: None,
                raw_name: "my_tool_call".to_string(),
                raw_arguments: "true".to_string(),
                arguments: None,
                id: "0".to_string(),
            }),
            ContentBlockChatOutput::Thought(Thought {
                text: Some("Some thought".to_string()),
                signature: None,
                provider_type: None,
            }),
            ContentBlockChatOutput::Thought(Thought {
                text: Some("My other interleaved thought".to_string()),
                signature: None,
                provider_type: None,
            }),
        ];
        assert_eq!(chat_result.content, expected_content);

        assert_eq!(chat_result.model_inference_results.len(), 1);
        let model_inference_result = chat_result.model_inference_results.first().unwrap();
        assert_eq!(&*model_inference_result.model_name, model_name);
        assert_eq!(
            &*model_inference_result.model_provider_name,
            model_provider_name
        );
        assert_eq!(model_inference_result.raw_request, raw_request);
    }

    #[tokio::test]
    async fn test_collect_chunks_tool_name_accumulation() {
        let templates = TemplateConfig::default();
        let function_config = Arc::new(FunctionConfig::Chat(FunctionConfigChat::default()));
        let model_name = "test_model";
        let model_provider_name = "test_provider";
        let raw_request = "raw request".to_string();
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let created = current_timestamp();
        let latency = Duration::from_millis(150);

        // Test case 1: Tool name sent in first chunk, then arguments accumulated
        let chunks_case1 = vec![
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                    id: "tool_1".to_string(),
                    raw_name: Some("get_weather".to_string()),
                    raw_arguments: "{\"loca".to_string(),
                })],
                created,
                usage: None,
                raw_response: "chunk1".to_string(),
                latency,
                finish_reason: None,
            }),
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                    id: "tool_1".to_string(),
                    raw_name: None, // No name in subsequent chunks
                    raw_arguments: "tion\": \"San Francisco\", \"unit\": \"celsius\"}".to_string(),
                })],
                created,
                usage: Some(Usage {
                    input_tokens: 10,
                    output_tokens: 20,
                }),
                raw_response: "chunk2".to_string(),
                latency: Duration::from_millis(250),
                finish_reason: Some(FinishReason::ToolCall),
            }),
        ];

        let collect_chunks_args = CollectChunksArgs {
            inference_id,
            episode_id,
            value: chunks_case1,
            system: None,
            input_messages: vec![],
            function: function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "",
            variant_name: "",
            dynamic_output_schema: None,
            templates: &templates,
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            ticket_borrow: TicketBorrows::empty(),
        };

        let result = collect_chunks(collect_chunks_args).await.unwrap();
        let chat_result = match result {
            InferenceResult::Chat(chat_result) => chat_result,
            InferenceResult::Json(_) => panic!("Expected Chat inference response"),
        };

        assert_eq!(chat_result.content.len(), 1);
        match &chat_result.content[0] {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "get_weather");
                assert_eq!(
                    tool_call.raw_arguments,
                    r#"{"location": "San Francisco", "unit": "celsius"}"#
                );
                assert_eq!(tool_call.id, "tool_1");
            }
            _ => panic!("Expected tool call block"),
        }

        // Test case 2: Multiple tool calls with different IDs and name accumulation
        let chunks_case2 = vec![
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_1".to_string(),
                        raw_name: Some("get_wea".to_string()),
                        raw_arguments: "{\"loc".to_string(),
                    }),
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_2".to_string(),
                        raw_name: Some("calculate".to_string()),
                        raw_arguments: "{\"expr".to_string(),
                    }),
                ],
                created,
                usage: None,
                raw_response: "chunk1".to_string(),
                latency,
                finish_reason: None,
            }),
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_1".to_string(),
                        raw_name: Some("ther".to_string()), // Continue accumulating name
                        raw_arguments: "ation\": \"NYC\"}".to_string(),
                    }),
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_2".to_string(),
                        raw_name: None, // No more name for tool_2
                        raw_arguments: "ession\": \"2+2\"}".to_string(),
                    }),
                ],
                created,
                usage: Some(Usage {
                    input_tokens: 15,
                    output_tokens: 25,
                }),
                raw_response: "chunk2".to_string(),
                latency: Duration::from_millis(250),
                finish_reason: Some(FinishReason::ToolCall),
            }),
        ];

        let collect_chunks_args = CollectChunksArgs {
            inference_id,
            episode_id,
            value: chunks_case2,
            system: None,
            input_messages: vec![],
            function: function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "",
            variant_name: "",
            dynamic_output_schema: None,
            templates: &templates,
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            ticket_borrow: TicketBorrows::empty(),
        };

        let result = collect_chunks(collect_chunks_args).await.unwrap();
        let chat_result = match result {
            InferenceResult::Chat(chat_result) => chat_result,
            InferenceResult::Json(_) => panic!("Expected Chat inference response"),
        };

        assert_eq!(chat_result.content.len(), 2);
        match &chat_result.content[0] {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "get_weather");
                assert_eq!(tool_call.raw_arguments, r#"{"location": "NYC"}"#);
                assert_eq!(tool_call.id, "tool_1");
            }
            _ => panic!("Expected first tool call block"),
        }
        match &chat_result.content[1] {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "calculate");
                assert_eq!(tool_call.raw_arguments, r#"{"expression": "2+2"}"#);
                assert_eq!(tool_call.id, "tool_2");
            }
            _ => panic!("Expected second tool call block"),
        }

        // Test case 3: Tool call with no name in first chunk (should start with empty name)
        let chunks_case3 = vec![
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                    id: "tool_1".to_string(),
                    raw_name: None, // No name in first chunk
                    raw_arguments: "{\"key\":".to_string(),
                })],
                created,
                usage: None,
                raw_response: "chunk1".to_string(),
                latency,
                finish_reason: None,
            }),
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                    id: "tool_1".to_string(),
                    raw_name: Some("my_function".to_string()), // Name comes later
                    raw_arguments: " \"value\"}".to_string(),
                })],
                created,
                usage: Some(Usage {
                    input_tokens: 5,
                    output_tokens: 10,
                }),
                raw_response: "chunk2".to_string(),
                latency: Duration::from_millis(250),
                finish_reason: Some(FinishReason::ToolCall),
            }),
        ];

        let collect_chunks_args = CollectChunksArgs {
            inference_id,
            episode_id,
            value: chunks_case3,
            system: None,
            input_messages: vec![],
            function: function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "",
            variant_name: "",
            dynamic_output_schema: None,
            templates: &templates,
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            ticket_borrow: TicketBorrows::empty(),
        };

        let result = collect_chunks(collect_chunks_args).await.unwrap();
        let chat_result = match result {
            InferenceResult::Chat(chat_result) => chat_result,
            InferenceResult::Json(_) => panic!("Expected Chat inference response"),
        };

        assert_eq!(chat_result.content.len(), 1);
        match &chat_result.content[0] {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "my_function"); // Should accumulate to the full name
                assert_eq!(tool_call.raw_arguments, r#"{"key": "value"}"#);
                assert_eq!(tool_call.id, "tool_1");
            }
            _ => panic!("Expected tool call block"),
        }

        // Test case 4: Mixed content with text and tool calls preserving order
        let chunks_case4 = vec![
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![
                    ContentBlockChunk::Text(TextChunk {
                        text: "I'll help you with that. ".to_string(),
                        id: "0".to_string(),
                    }),
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_1".to_string(),
                        raw_name: Some("search".to_string()),
                        raw_arguments: "{\"query\"".to_string(),
                    }),
                ],
                created,
                usage: None,
                raw_response: "chunk1".to_string(),
                latency,
                finish_reason: None,
            }),
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![
                    ContentBlockChunk::Text(TextChunk {
                        text: "Let me search for information.".to_string(),
                        id: "0".to_string(),
                    }),
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_1".to_string(),
                        raw_name: None,
                        raw_arguments: ": \"weather today\"}".to_string(),
                    }),
                ],
                created,
                usage: Some(Usage {
                    input_tokens: 20,
                    output_tokens: 15,
                }),
                raw_response: "chunk2".to_string(),
                latency: Duration::from_millis(250),
                finish_reason: Some(FinishReason::ToolCall),
            }),
        ];

        let collect_chunks_args = CollectChunksArgs {
            inference_id,
            episode_id,
            value: chunks_case4,
            system: None,
            input_messages: vec![],
            function: function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "",
            variant_name: "",
            dynamic_output_schema: None,
            templates: &templates,
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            ticket_borrow: TicketBorrows::empty(),
        };

        let result = collect_chunks(collect_chunks_args).await.unwrap();
        let chat_result = match result {
            InferenceResult::Chat(chat_result) => chat_result,
            InferenceResult::Json(_) => panic!("Expected Chat inference response"),
        };

        assert_eq!(chat_result.content.len(), 2);
        // Order should be preserved: text first, then tool call
        match &chat_result.content[0] {
            ContentBlockChatOutput::Text(text) => {
                assert_eq!(
                    text.text,
                    "I'll help you with that. Let me search for information."
                );
            }
            _ => panic!("Expected text block first"),
        }
        match &chat_result.content[1] {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "search");
                assert_eq!(tool_call.raw_arguments, r#"{"query": "weather today"}"#);
                assert_eq!(tool_call.id, "tool_1");
            }
            _ => panic!("Expected tool call block second"),
        }

        // Test case 5: Tool call with empty name parts that should result in empty final name
        let chunks_case5 = vec![InferenceResultChunk::Chat(ChatInferenceResultChunk {
            content: vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "tool_1".to_string(),
                raw_name: None,
                raw_arguments: "{\"test\": true}".to_string(),
            })],
            created,
            usage: Some(Usage {
                input_tokens: 5,
                output_tokens: 5,
            }),
            raw_response: "chunk1".to_string(),
            latency,
            finish_reason: Some(FinishReason::ToolCall),
        })];

        let collect_chunks_args = CollectChunksArgs {
            inference_id,
            episode_id,
            value: chunks_case5,
            system: None,
            input_messages: vec![],
            function: function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "",
            variant_name: "",
            dynamic_output_schema: None,
            templates: &templates,
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            ticket_borrow: TicketBorrows::empty(),
        };

        let result = collect_chunks(collect_chunks_args).await.unwrap();
        let chat_result = match result {
            InferenceResult::Chat(chat_result) => chat_result,
            InferenceResult::Json(_) => panic!("Expected Chat inference response"),
        };

        assert_eq!(chat_result.content.len(), 1);
        match &chat_result.content[0] {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, ""); // Should be empty string when no name provided
                assert_eq!(tool_call.raw_arguments, r#"{"test": true}"#);
                assert_eq!(tool_call.id, "tool_1");
            }
            _ => panic!("Expected tool call block"),
        }

        // Test case 6: Complex multi-tool name accumulation across multiple chunks
        let chunks_case6 = vec![
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_1".to_string(),
                        raw_name: Some("get_".to_string()),
                        raw_arguments: "{\"lo".to_string(),
                    }),
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_2".to_string(),
                        raw_name: Some("cal".to_string()),
                        raw_arguments: "{\"op".to_string(),
                    }),
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_3".to_string(),
                        raw_name: Some("send_".to_string()),
                        raw_arguments: "{\"me".to_string(),
                    }),
                ],
                created,
                usage: None,
                raw_response: "chunk1".to_string(),
                latency,
                finish_reason: None,
            }),
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_1".to_string(),
                        raw_name: Some("wea".to_string()),
                        raw_arguments: "cation\": ".to_string(),
                    }),
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_2".to_string(),
                        raw_name: Some("cul".to_string()),
                        raw_arguments: "eration\": ".to_string(),
                    }),
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_3".to_string(),
                        raw_name: Some("email".to_string()),
                        raw_arguments: "ssage\": ".to_string(),
                    }),
                ],
                created,
                usage: None,
                raw_response: "chunk2".to_string(),
                latency,
                finish_reason: None,
            }),
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_1".to_string(),
                        raw_name: Some("ther".to_string()),
                        raw_arguments: "\"Paris\"}".to_string(),
                    }),
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_2".to_string(),
                        raw_name: Some("ate".to_string()),
                        raw_arguments: "\"5*5\"}".to_string(),
                    }),
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_3".to_string(),
                        raw_name: None, // No more name parts
                        raw_arguments: "\"Hello world\"}".to_string(),
                    }),
                ],
                created,
                usage: Some(Usage {
                    input_tokens: 20,
                    output_tokens: 30,
                }),
                raw_response: "chunk3".to_string(),
                latency: Duration::from_millis(250),
                finish_reason: Some(FinishReason::ToolCall),
            }),
        ];

        let collect_chunks_args = CollectChunksArgs {
            inference_id,
            episode_id,
            value: chunks_case6,
            system: None,
            input_messages: vec![],
            function: function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "",
            variant_name: "",
            dynamic_output_schema: None,
            templates: &templates,
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            ticket_borrow: TicketBorrows::empty(),
        };

        let result = collect_chunks(collect_chunks_args).await.unwrap();
        let chat_result = match result {
            InferenceResult::Chat(chat_result) => chat_result,
            InferenceResult::Json(_) => panic!("Expected Chat inference response"),
        };

        assert_eq!(chat_result.content.len(), 3);
        match &chat_result.content[0] {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "get_weather"); // "get_" + "wea" + "ther"
                assert_eq!(tool_call.raw_arguments, r#"{"location": "Paris"}"#);
                assert_eq!(tool_call.id, "tool_1");
            }
            _ => panic!("Expected first tool call block"),
        }
        match &chat_result.content[1] {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "calculate"); // "cal" + "cul" + "ate"
                assert_eq!(tool_call.raw_arguments, r#"{"operation": "5*5"}"#);
                assert_eq!(tool_call.id, "tool_2");
            }
            _ => panic!("Expected second tool call block"),
        }
        match &chat_result.content[2] {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "send_email"); // "send_" + "email"
                assert_eq!(tool_call.raw_arguments, r#"{"message": "Hello world"}"#);
                assert_eq!(tool_call.id, "tool_3");
            }
            _ => panic!("Expected third tool call block"),
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
                assert_eq!(text, "Hello, world!");
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
                assert_eq!(arguments, json!({"key": "value"}).as_object().unwrap());
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
                assert_eq!(value, "Hello");
            }
            _ => panic!("Expected Text content"),
        }
        match &message.content[1] {
            InputMessageContent::ToolCall(tool_call) => {
                assert_eq!(tool_call.id, "123");
                assert_eq!(tool_call.name, Some("test_tool".to_string()));
                assert_eq!(tool_call.arguments, Some(json!("{}")));
                assert_eq!(tool_call.raw_name, None);
                assert_eq!(tool_call.raw_arguments, None);
            }
            _ => panic!("Expected ToolCall content"),
        }
        // Test case for multiple content items with JSON object in text block
        let input = json!({
            "role": "user",
            "content": [
                {"type": "template", "name": "user", "arguments": {"complex": "json", "with": ["nested", "array"]}},
                {"type": "tool_call", "id": "456", "name": "another_tool", "arguments": {"key": "value"}}
            ]
        });
        let message: InputMessage = serde_json::from_value(input).unwrap();
        assert_eq!(message.role, Role::User);
        assert_eq!(message.content.len(), 2);
        match &message.content[0] {
            InputMessageContent::Template(TemplateInput { name, arguments }) => {
                assert_eq!(name, "user");
                assert_eq!(
                    arguments,
                    json!({"complex": "json", "with": ["nested", "array"]})
                        .as_object()
                        .unwrap()
                );
            }
            _ => panic!("Expected Text content with JSON object"),
        }
        match &message.content[1] {
            InputMessageContent::ToolCall(tool_call) => {
                assert_eq!(tool_call.id, "456");
                assert_eq!(tool_call.name, Some("another_tool".to_string()));
                assert_eq!(tool_call.arguments, Some(json!({"key":"value"})));
                assert_eq!(tool_call.raw_name, None);
                assert_eq!(tool_call.raw_arguments, None,);
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
                raw_name: Some("test_tool".to_string()),
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
                provider_type: None,
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
                    raw_name: Some("test_tool".to_string()),
                }),
                ContentBlockChunk::Thought(ThoughtChunk {
                    id: "789".to_string(),
                    text: Some("final thought".to_string()),
                    signature: None,
                    provider_type: None,
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
        let mut blocks: IndexMap<(ContentBlockOutputType, String), ContentBlockOutput> =
            IndexMap::new();
        let mut ttft: Option<Duration> = None;
        let chunk_latency = Duration::from_millis(100);

        // Test case 1: Create new text block
        handle_textual_content_block(
            &mut blocks,
            (ContentBlockOutputType::Text, "1".to_string()),
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
        match blocks
            .get(&(ContentBlockOutputType::Text, "1".to_string()))
            .unwrap()
        {
            ContentBlockOutput::Text(Text { text }) => assert_eq!(text, "Hello"),
            _ => panic!("Expected text block"),
        }

        // Test case 2: Append to existing text block
        handle_textual_content_block(
            &mut blocks,
            (ContentBlockOutputType::Text, "1".to_string()),
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
        match blocks
            .get(&(ContentBlockOutputType::Text, "1".to_string()))
            .unwrap()
        {
            ContentBlockOutput::Text(Text { text }) => assert_eq!(text, "Hello World"),
            _ => panic!("Expected text block"),
        }

        // Test case 3: Empty text should not create block
        handle_textual_content_block(
            &mut blocks,
            (ContentBlockOutputType::Text, "2".to_string()),
            String::new(),
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
            (ContentBlockOutputType::Thought, "3".to_string()),
            "Thinking...".to_string(),
            &mut ttft,
            chunk_latency,
            |text| {
                ContentBlockOutput::Thought(Thought {
                    text: Some(text),
                    signature: None,
                    provider_type: None,
                })
            },
            |block, text| {
                if let ContentBlockOutput::Thought(thought) = block {
                    thought.text.get_or_insert_default().push_str(text);
                }
            },
        );

        assert_eq!(blocks.len(), 2);
        match blocks
            .get(&(ContentBlockOutputType::Thought, "3".to_string()))
            .unwrap()
        {
            ContentBlockOutput::Thought(Thought {
                text,
                signature: _,
                provider_type: _,
            }) => {
                assert_eq!(text, &Some("Thinking...".to_string()));
            }
            _ => panic!("Expected thought block"),
        }
    }
}
