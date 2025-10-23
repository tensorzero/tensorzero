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
use crate::endpoints::object_storage::get_object;
use crate::http::TensorzeroHttpClient;
use crate::inference::types::file::Base64FileMetadata;
use crate::inference::types::resolved_input::{
    write_file, FileUrl, LazyFile, LazyResolvedInput, LazyResolvedInputMessage,
    LazyResolvedInputMessageContent,
};
use crate::inference::types::storage::StorageKind;
use crate::inference::types::stored_input::StoredFile;
use crate::rate_limiting::{
    get_estimated_tokens, EstimatedRateLimitResourceUsage, RateLimitResource,
    RateLimitResourceUsage, RateLimitedInputContent, RateLimitedRequest,
};
use crate::serde_util::{
    deserialize_defaulted_json_string, deserialize_json_string, deserialize_optional_json_string,
};
use crate::tool::ToolCallInput;
use crate::variant::chat_completion::{ASSISTANT_TEXT_TEMPLATE_VAR, USER_TEXT_TEMPLATE_VAR};
use derive_builder::Builder;
use extra_body::{FullExtraBodyConfig, UnfilteredInferenceExtraBody};
use extra_headers::FullExtraHeadersConfig;
use file::sanitize_raw_request;
pub use file::{Base64File, File};
use futures::future::{join_all, try_join_all};
use futures::FutureExt;
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
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use uuid::Uuid;

use crate::cache::NonStreamingCacheData;
use crate::function::FunctionConfigType;
use crate::tool::ToolCallConfigDatabaseInsert;
use crate::tool::{ToolCall, ToolCallConfig, ToolCallOutput, ToolResult};
use crate::{cache::CacheData, config::ObjectStoreInfo};
use crate::{endpoints::inference::InferenceDatabaseInsertMetadata, variant::InferenceConfig};
use crate::{
    endpoints::inference::InferenceParams,
    error::{ErrorDetails, ErrorDetails::RateLimitMissingMaxTokens},
};
use crate::{error::Error, variant::JsonMode};
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
pub mod streams;

pub use resolved_input::ResolvedRequestMessage;
pub use stored_input::{
    StoredInput, StoredInputMessage, StoredInputMessageContent, StoredRequestMessage,
};
pub use streams::{
    collect_chunks, ChatInferenceResultChunk, CollectChunksArgs, ContentBlockChunk,
    InferenceResultChunk, InferenceResultStream, JsonInferenceResultChunk,
    PeekableProviderInferenceResponseStream, ProviderInferenceResponseChunk,
    ProviderInferenceResponseStreamInner, TextChunk, ThoughtChunk,
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
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
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

    /// Turns the input into a StoredInput, avoiding resolving network resources if possible.
    pub async fn into_stored_input(
        self,
        object_store_info: &Option<ObjectStoreInfo>,
    ) -> Result<StoredInput, Error> {
        let stored_messages = futures::future::try_join_all(
            self.messages
                .into_iter()
                .map(|message| message.into_stored_input_message(object_store_info)),
        )
        .await?;

        Ok(StoredInput {
            system: self.system,
            messages: stored_messages,
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
    /// Turns the input into a ResolvedInputMessage by fetching network resources for Files.
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

    /// Turns the input into a StoredInputMessage by converting Files to StoredFiles, bypassing resolving network resources if possible.
    pub async fn into_stored_input_message(
        self,
        object_store_info: &Option<ObjectStoreInfo>,
    ) -> Result<StoredInputMessage, Error> {
        let content = futures::future::try_join_all(
            self.content
                .into_iter()
                .map(|content| content.into_stored_input_message_content(object_store_info)),
        )
        .await?;

        Ok(StoredInputMessage {
            role: self.role,
            content,
        })
    }
}

/// Extracts the StorageKind from the FetchContext, or returns an error if the object store is not configured.
fn get_storage_kind(context: &FetchContext<'_>) -> Result<StorageKind, Error> {
    let object_store_info = context.object_store_info.as_ref().ok_or_else(|| {
        Error::new(ErrorDetails::ObjectStoreUnconfigured {
            block_type: "file".to_string(),
        })
    })?;
    Ok(object_store_info.kind.clone())
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
                match &file {
                    File::Url { url, mime_type } => {
                        // Check that we have an object store *outside* of the future that we're going to store in
                        // `LazyResolvedInputMessageContent::File`. We want to error immediately if the user tries
                        // to use a file input without explicitly configuring an object store (either explicit enabled or disabled)
                        let storage_kind = get_storage_kind(&context)?;
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
                        let storage_kind = get_storage_kind(&context)?;
                        let path = storage_kind.file_path(&file)?;

                        LazyResolvedInputMessageContent::File(Box::new(LazyFile::FileWithPath(
                            FileWithPath {
                                file,
                                storage_path: path,
                            },
                        )))
                    }
                    File::ObjectStorage {
                        source_url,
                        mime_type,
                        storage_path,
                    } => {
                        let source_url = source_url.clone();
                        let object_store_info = context.object_store_info.clone();
                        let owned_storage_path = storage_path.clone();
                        let mime_type_for_closure = mime_type.clone();
                        // Construct a future that will fetch the file from the object store.
                        // Important - the future will not actually begin executing (including opening the network connection)
                        // until the first time the `Shared` wrapper is `.await`ed.
                        let delayed_file_future = async move {
                            let object_response =
                                get_object(object_store_info.as_ref(), owned_storage_path.clone())
                                    .await?;
                            let file = Base64File {
                                url: None,
                                mime_type: mime_type_for_closure,
                                data: object_response.data,
                            };
                            Ok(FileWithPath {
                                file,
                                storage_path: owned_storage_path,
                            })
                        };
                        LazyResolvedInputMessageContent::File(Box::new(LazyFile::ObjectStorage {
                            metadata: Base64FileMetadata {
                                url: source_url,
                                mime_type: mime_type.clone(),
                            },
                            storage_path: storage_path.clone(),
                            future: delayed_file_future.boxed().shared(),
                        }))
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
                LazyFile::ObjectStorage { future, .. } => {
                    ResolvedInputMessageContent::File(Box::new(future.await?))
                }
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

    /// Converts the message content into a StoredInputMessageContent,
    /// bypassing fetching files / resolving network resources if possible.
    pub async fn into_stored_input_message_content(
        self,
        object_store_info: &Option<ObjectStoreInfo>,
    ) -> Result<StoredInputMessageContent, Error> {
        Ok(match self {
            // When the input file already contains the ObjectStorage path, we discard the future without awaiting
            // (which doesn't trigger the pending network request), and directly convert the metadata and
            // storage_path into a StoredFile.
            LazyResolvedInputMessageContent::File(file) => match *file {
                LazyFile::ObjectStorage {
                    metadata,
                    storage_path,
                    future: _,
                } => StoredInputMessageContent::File(Box::new(StoredFile {
                    file: metadata,
                    storage_path,
                })),
                // All other file types need to be resolved first.
                other => {
                    let file = other.resolve().await?.into_owned();
                    // Because this may trigger during a datapoint update,
                    // we need to try and write the file to object storage first.
                    write_file(
                        object_store_info,
                        file.file.clone(),
                        file.storage_path.clone(),
                    )
                    .await?;

                    StoredInputMessageContent::File(Box::new(file.into_stored_file()))
                }
            },
            // All other cases delegate to the "resolve" case, which is mostly just a type conversion.
            other => other.resolve().await?.into_stored_input_message_content(),
        })
    }
}

/// InputMessage and Role are our representation of the input sent by the client
/// prior to any processing into LLM representations below.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
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
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, tag = "type", rename_all = "snake_case"))]
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

#[derive(ts_rs::TS, Clone, Debug, Deserialize, PartialEq, Serialize)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(get_all))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ThoughtSummaryBlock {
    SummaryText { text: String },
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
    #[ts(optional)]
    pub signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub summary: Option<Vec<ThoughtSummaryBlock>>,
    /// When set, this 'Thought' block will only be used for providers
    /// matching this type (e.g. `anthropic`). Other providers will emit
    /// a warning and discard the block.
    #[serde(
        rename = "_internal_provider_type",
        skip_serializing_if = "Option::is_none"
    )]
    #[ts(optional)]
    pub provider_type: Option<String>,
}

impl RateLimitedInputContent for Thought {
    fn estimated_input_token_usage(&self) -> u64 {
        let Thought {
            text,
            signature,
            // We intentionally do *not* count the summary towards the token usage
            // Even though OpenAI responses requires passing the summaries back in a multi-turn
            // conversation, we expect that the actual model will ignore them, since they're
            // not the internal model thoughts.
            summary: _,
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
    pub fetch_and_encode_input_files_before_inference: bool,
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
    fn estimated_resource_usage(
        &self,
        resources: &[RateLimitResource],
    ) -> Result<EstimatedRateLimitResourceUsage, Error> {
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
            fetch_and_encode_input_files_before_inference: _,
            extra_headers: _,
            extra_cache_key: _,
        } = self;

        let tokens = if resources.contains(&RateLimitResource::Token) {
            let system_tokens = system
                .as_ref()
                .map(|s| get_estimated_tokens(s))
                .unwrap_or(0);
            let messages_tokens: u64 = messages
                .iter()
                .map(RateLimitedInputContent::estimated_input_token_usage)
                .sum();
            let output_tokens =
                max_tokens.ok_or_else(|| Error::new(RateLimitMissingMaxTokens))? as u64;
            Some(system_tokens + messages_tokens + output_tokens)
        } else {
            None
        };

        let model_inferences = if resources.contains(&RateLimitResource::ModelInference) {
            Some(1)
        } else {
            None
        };

        Ok(EstimatedRateLimitResourceUsage {
            model_inferences,
            tokens,
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
        Ok(RateLimitResourceUsage::Exact {
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
    /// This is never omitted from the response even if it's None. A `null` value indicates no output from the model.
    /// It's rare and unexpected from the model, but it's possible.
    pub raw: Option<String>,
    /// This is never omitted from the response even if it's None.
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

/*
In the streaming case we convert ProviderInferenceResponseChunks into a InferenceResultChunk, which is then
converted into an InferenceResponseChunk and sent to the client.
We then collect all the InferenceResultChunks into an InferenceResult for validation and storage after the fact.

Alongside the response, we also store information about what happened during the request.
For this we convert the InferenceResult into a ChatInferenceDatabaseInsert or JsonInferenceDatabaseInsert and ModelInferenceDatabaseInserts,
which are written to ClickHouse tables of the same name asynchronously.
*/
#[derive(Deserialize)]
#[serde(deny_unknown_fields, tag = "function_type", rename_all = "snake_case")]
pub enum TaggedInferenceDatabaseInsert {
    Chat(ChatInferenceDatabaseInsert),
    Json(JsonInferenceDatabaseInsert),
}

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
    use super::*;
    use crate::jsonschema_util::DynamicJSONSchema;
    use crate::providers::test_helpers::get_temperature_tool_config;
    use crate::tool::{DynamicToolConfig, ToolChoice, ToolConfig};
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
            provider_tools: None,
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
            provider_tools: None,
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
}
