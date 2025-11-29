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

use derive_builder::Builder;
use extra_body::{FullExtraBodyConfig, UnfilteredInferenceExtraBody};
use extra_headers::FullExtraHeadersConfig;
use file::sanitize_raw_request;
pub use file::{
    Base64File, File, ObjectStorageError, ObjectStorageFile, ObjectStoragePointer,
    PendingObjectStoreFile, UrlFile,
};
use futures::future::{join_all, try_join_all};
use futures::FutureExt;
use itertools::Itertools;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::types::PyAny;
#[cfg(feature = "pyo3")]
use pyo3_helpers::serialize_to_dict;
pub use resolved_input::{ResolvedInput, ResolvedInputMessage, ResolvedInputMessageContent};
use schemars::JsonSchema;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};
use std::borrow::Borrow;
use std::{
    borrow::Cow,
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tensorzero_derive::export_schema;
use uuid::Uuid;

use crate::cache::{CacheData, NonStreamingCacheData};
use crate::config::ObjectStoreInfo;
use crate::endpoints::inference::{InferenceDatabaseInsertMetadata, InferenceParams};
use crate::endpoints::object_storage::get_object;
use crate::error::{Error, ErrorDetails, ErrorDetails::RateLimitMissingMaxTokens};
use crate::function::FunctionConfigType;
use crate::http::TensorzeroHttpClient;
use crate::inference::types::chat_completion_inference_params::ChatCompletionInferenceParamsV2;
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
use crate::serde_util::{deserialize_defaulted_json_string, deserialize_json_string};
use crate::tool::{
    deserialize_optional_tool_info, InferenceResponseToolCall, ToolCall, ToolCallConfig,
    ToolCallConfigDatabaseInsert, ToolCallWrapper, ToolResult,
};
use crate::variant::{InferenceConfig, JsonMode};

pub mod batch;
pub mod chat_completion_inference_params;
pub mod extra_body;
pub mod extra_headers;
pub mod extra_stuff;
pub mod file;
mod input_message;
#[cfg(feature = "pyo3")]
pub mod pyo3_helpers;
pub mod resolved_input;
mod role;
pub mod storage;
pub mod stored_input;
pub mod streams;
pub mod usage;

pub use resolved_input::ResolvedRequestMessage;
pub use role::Role;
pub use stored_input::{
    StoredInput, StoredInputMessage, StoredInputMessageContent, StoredRequestMessage,
};
pub use streams::{
    collect_chunks, ChatInferenceResultChunk, CollectChunksArgs, ContentBlockChunk,
    InferenceResultChunk, InferenceResultStream, JsonInferenceResultChunk,
    PeekableProviderInferenceResponseStream, ProviderInferenceResponseChunk,
    ProviderInferenceResponseStreamInner, TextChunk, ThoughtChunk, UnknownChunk,
};
pub use usage::Usage;

/*
 * Data flow in TensorZero
 *
 * The flow of an inference request through TensorZero can be viewed as a series of transformations between types.
 * Most of them are defined below.
 */

/// API representation of an input to a model.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Default, ts_rs::TS, JsonSchema)]
#[serde(deny_unknown_fields)]
#[ts(export, optional_fields)]
#[export_schema]
pub struct Input {
    /// System prompt of the input.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub system: Option<System>,

    /// Messages in the input.
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
        context: &FetchContext<'_>,
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

    /// Turns the input into a StoredInput, without resolving network resources for files.
    /// Returns an error if any files are present.
    pub fn into_stored_input_without_file_handling(self) -> Result<StoredInput, Error> {
        Ok(StoredInput {
            system: self.system,
            messages: self
                .messages
                .into_iter()
                .map(InputMessage::into_stored_input_message_without_file_handling)
                .try_collect()?,
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
        context: &FetchContext<'_>,
    ) -> Result<LazyResolvedInputMessage, Error> {
        Ok(LazyResolvedInputMessage {
            role: self.role,
            content: self
                .content
                .into_iter()
                .map(|content| content.into_lazy_resolved_input_message(context))
                .collect::<Result<Vec<LazyResolvedInputMessageContent>, Error>>()?,
        })
    }

    /// Turns the input message into a StoredInputMessage, without resolving network resources for files.
    /// Returns an error if the message contains any files that require storage (e.g. external URLs, Base64).
    pub fn into_stored_input_message_without_file_handling(
        self,
    ) -> Result<StoredInputMessage, Error> {
        Ok(StoredInputMessage {
            role: self.role,
            content: self
                .content
                .into_iter()
                .map(InputMessageContent::into_stored_input_message_content_without_file_handling)
                .try_collect()?,
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

/// Extracts the `StorageKind` from the `FetchContext`, or returns an error if the object store is not configured.
fn get_storage_kind(context: &FetchContext<'_>) -> Result<StorageKind, Error> {
    let object_store_info = context.object_store_info.as_ref().ok_or_else(|| {
        Error::new(ErrorDetails::ObjectStoreUnconfigured {
            block_type: "file".to_string(),
        })
    })?;
    Ok(object_store_info.kind.clone())
}

impl InputMessageContent {
    /// The `role` parameter is only used to handle legacy role-based templates (`{"type": "text", "value": ...}`).
    /// Once we removed support for these input blocks (and only support `{"type": "template", "name": "...", "arguments": ...}`),
    /// we can remove the `role` parameter.
    pub fn into_lazy_resolved_input_message(
        self,
        context: &FetchContext<'_>,
    ) -> Result<LazyResolvedInputMessageContent, Error> {
        Ok(match self {
            InputMessageContent::Text(Text { text }) => {
                LazyResolvedInputMessageContent::Text(Text { text })
            }
            InputMessageContent::RawText(raw_text) => {
                LazyResolvedInputMessageContent::RawText(raw_text)
            }
            InputMessageContent::Thought(thought) => {
                LazyResolvedInputMessageContent::Thought(thought)
            }
            InputMessageContent::Template(template) => {
                LazyResolvedInputMessageContent::Template(template)
            }
            InputMessageContent::ToolCall(tool_call) => {
                LazyResolvedInputMessageContent::ToolCall(tool_call.try_into()?)
            }
            InputMessageContent::ToolResult(tool_result) => {
                LazyResolvedInputMessageContent::ToolResult(tool_result)
            }
            InputMessageContent::File(file) => {
                match &file {
                    // User provided a file URL.
                    // We create a lazy future that will fetch the file when needed.
                    // The future is not executed immediately - it only runs when awaited.
                    // This allows model providers that support URL forwarding to skip the fetch entirely.
                    // When the future does run, it:
                    // 1. Fetches the file from the URL
                    // 2. Computes a content-addressed `storage_path`
                    // 3. Returns a `ObjectStorageFile` with the data
                    File::Url(UrlFile {
                        url,
                        mime_type,
                        detail,
                        filename,
                    }) => {
                        // Check that we have an object store *outside* of the future that we're going to store in
                        // `LazyResolvedInputMessageContent::File`. We want to error immediately if the user tries
                        // to use a file input without explicitly configuring an object store (either explicit enabled or disabled)
                        let storage_kind = get_storage_kind(context)?;
                        let client = context.client.clone();
                        // Construct a future that will actually fetch the file URL from the network.
                        // Important: we do *not* use `tokio::spawn` here. As a result, the future
                        // will not actually begin executing (including opening the network connection)
                        // until the first time the `Shared` wrapper is `.await`ed.
                        // This ensures that if we never actually need to download the file
                        // (due to model providers forwarding image urls, and object store observability being disabled),
                        // we will skip downloading the file entirely.
                        let url = url.clone();
                        let mime_type = mime_type.clone();
                        let detail_clone = detail.clone();
                        let detail_for_future = detail.clone();
                        let filename_for_future = filename.clone();
                        let delayed_file_future = async move {
                            let base64_file = file.take_or_fetch(&client).await?;
                            let path = storage_kind.file_path(&base64_file)?;
                            Ok(ObjectStorageFile {
                                file: ObjectStoragePointer {
                                    source_url: base64_file.source_url.clone(),
                                    mime_type: base64_file.mime_type.clone(),
                                    storage_path: path,
                                    detail: detail_for_future,
                                    filename: filename_for_future,
                                },
                                data: base64_file.data().to_string(),
                            })
                        };
                        LazyResolvedInputMessageContent::File(Box::new(LazyFile::Url {
                            file_url: FileUrl {
                                url,
                                mime_type,
                                detail: detail_clone,
                            },
                            future: delayed_file_future.boxed().shared(),
                        }))
                    }
                    // User provided base64-encoded file data.
                    // We immediately:
                    // 1. Compute a content-addressed `storage_path` from the data
                    // 2. Wrap the data in `PendingObjectStoreFile` to signal it needs writing
                    // The data is ready to use but not yet persisted to object storage.
                    // The write will happen later in `into_stored_input_message_content`.
                    File::Base64(base64_file) => {
                        let source_url = &base64_file.source_url;
                        let mime_type = &base64_file.mime_type;
                        let data = base64_file.data();
                        let detail = &base64_file.detail;
                        let filename = &base64_file.filename;

                        let storage_kind = get_storage_kind(context)?;
                        let base64_file_for_path = Base64File::new(
                            source_url.clone(),
                            mime_type.clone(),
                            data.to_string(),
                            // We explicitly set detail to None when computing the storage path.
                            // This is intentional for content-addressing: the detail parameter controls
                            // how providers process the image (resolution/token cost), but shouldn't
                            // affect the file's hash or storage location. The same image file with
                            // different detail values should map to the same storage path for deduplication.
                            None,
                            // We also set filename to None when computing the storage path for the same reason.
                            // The filename is metadata that shouldn't affect the file's hash or storage location.
                            None,
                        )?;
                        let path = storage_kind.file_path(&base64_file_for_path)?;

                        LazyResolvedInputMessageContent::File(Box::new(LazyFile::Base64(
                            PendingObjectStoreFile(ObjectStorageFile {
                                file: ObjectStoragePointer {
                                    source_url: source_url.clone(),
                                    mime_type: mime_type.clone(),
                                    storage_path: path,
                                    detail: detail.clone(),
                                    filename: filename.clone(),
                                },
                                data: data.to_string(),
                            }),
                        )))
                    }
                    // # File::ObjectStoragePointer
                    //
                    // User provided a reference to a file already in object storage.
                    // We create a lazy future that will fetch the file data when needed.
                    // The future is not executed immediately - it only runs when awaited.
                    // This allows us to skip fetching if the file data isn't needed (e.g., just storing metadata).
                    // When the future does run, it fetches the file from object storage.
                    //
                    // # File::ObjectStorageError
                    //
                    // User provided a failed that we previously attempted and failed to fetch from object storage.
                    // Here, we disregard the previous attempt and retry, as if the user had only sent the pointer.
                    File::ObjectStoragePointer(file)
                    | File::ObjectStorageError(ObjectStorageError { file, .. }) => {
                        let source_url_for_future = file.source_url.clone();
                        let object_store_info = context.object_store_info.clone();
                        let owned_storage_path = file.storage_path.clone();
                        let mime_type_for_closure = file.mime_type.clone();
                        let detail_for_future = file.detail.clone();
                        let filename_for_future = file.filename.clone();
                        // Construct a future that will fetch the file from the object store.
                        // Important: the future will not actually begin executing (including opening the network connection)
                        // until the first time the `Shared` wrapper is `.await`ed.
                        let delayed_file_future = async move {
                            let object_response =
                                get_object(object_store_info.as_ref(), owned_storage_path.clone())
                                    .await?;
                            Ok(ObjectStorageFile {
                                file: ObjectStoragePointer {
                                    source_url: source_url_for_future,
                                    mime_type: mime_type_for_closure,
                                    storage_path: owned_storage_path,
                                    detail: detail_for_future,
                                    filename: filename_for_future,
                                },
                                data: object_response.data,
                            })
                        };
                        LazyResolvedInputMessageContent::File(Box::new(
                            LazyFile::ObjectStoragePointer {
                                metadata: Base64FileMetadata {
                                    source_url: file.source_url.clone(),
                                    mime_type: file.mime_type.clone(),
                                    detail: file.detail.clone(),
                                    filename: file.filename.clone(),
                                },
                                storage_path: file.storage_path.clone(),
                                future: delayed_file_future.boxed().shared(),
                            },
                        ))
                    }
                    // User provided a file reference with data already in memory.
                    // This typically comes from roundtripping (e.g., `get_datapoints` → `update_datapoints`).
                    // The file is already persisted at `storage_path`, and we have the data available.
                    // No fetch or write is needed - we can use it directly.
                    File::ObjectStorage(resolved_file) => LazyResolvedInputMessageContent::File(
                        Box::new(LazyFile::ObjectStorage(resolved_file.clone())),
                    ),
                }
            }
            InputMessageContent::Unknown(unknown) => {
                LazyResolvedInputMessageContent::Unknown(unknown)
            }
        })
    }

    /// Convert the input message content into a StoredInputMessageContent, but without loading or storing any files.
    pub fn into_stored_input_message_content_without_file_handling(
        self,
    ) -> Result<StoredInputMessageContent, Error> {
        Ok(match self {
            InputMessageContent::Text(Text { text }) => {
                StoredInputMessageContent::Text(Text { text })
            }
            InputMessageContent::RawText(raw_text) => StoredInputMessageContent::RawText(raw_text),
            InputMessageContent::Thought(thought) => StoredInputMessageContent::Thought(thought),
            InputMessageContent::Template(template) => {
                StoredInputMessageContent::Template(template)
            }
            InputMessageContent::ToolCall(tool_call) => {
                StoredInputMessageContent::ToolCall(tool_call.try_into()?)
            }
            InputMessageContent::ToolResult(tool_result) => {
                StoredInputMessageContent::ToolResult(tool_result)
            }
            InputMessageContent::File(file) => {
                match file {
                    // If `file` is external, we cannot convert it directly to a StoredFile.
                    File::Url(_) | File::Base64(_) => {
                        return Err(Error::new(ErrorDetails::InvalidRequest {
                            message: "Cannot resolve file input without a fetch context. Please provide a fetch context when creating the input.".to_string()
                        }));
                    }
                    // If `file` is present in our object storage, even if it's an error, we can represent it as a StoredFile.
                    File::ObjectStorage(ObjectStorageFile { file, .. })
                    | File::ObjectStoragePointer(file)
                    | File::ObjectStorageError(ObjectStorageError { file, .. }) => {
                        StoredInputMessageContent::File(Box::new(StoredFile(file)))
                    }
                }
            }
            InputMessageContent::Unknown(unknown) => StoredInputMessageContent::Unknown(unknown),
        })
    }
}

impl LazyResolvedInputMessageContent {
    /// Converts lazy content into fully resolved content by executing any pending operations.
    /// For files, this means fetching data if needed (from URLs or object storage).
    /// This is used when we need the actual file data (e.g., for inference with providers
    /// that don't support URL forwarding, or for observability).
    pub async fn resolve(self) -> Result<ResolvedInputMessageContent, Error> {
        Ok(match self {
            LazyResolvedInputMessageContent::Text(text) => ResolvedInputMessageContent::Text(text),
            LazyResolvedInputMessageContent::Template(template) => {
                ResolvedInputMessageContent::Template(template)
            }
            LazyResolvedInputMessageContent::ToolCall(tool_call) => {
                ResolvedInputMessageContent::ToolCall(tool_call)
            }
            LazyResolvedInputMessageContent::ToolResult(tool_result) => {
                ResolvedInputMessageContent::ToolResult(tool_result)
            }
            LazyResolvedInputMessageContent::RawText(raw_text) => {
                ResolvedInputMessageContent::RawText(raw_text)
            }
            LazyResolvedInputMessageContent::Thought(thought) => {
                ResolvedInputMessageContent::Thought(thought)
            }
            LazyResolvedInputMessageContent::File(file) => match *file {
                // File from URL: await the fetch future to get the data.
                // This performs the network request and returns the file data.
                LazyFile::Url {
                    future,
                    file_url: _,
                } => ResolvedInputMessageContent::File(Box::new(future.await?)),
                // Base64 file: unwrap from `PendingObjectStoreFile`.
                // The data is already in memory, just needs type conversion.
                LazyFile::Base64(pending) => ResolvedInputMessageContent::File(Box::new(pending.0)),
                // File from object storage: await the fetch future to get the data.
                // This fetches the file from object storage and returns the file data.
                LazyFile::ObjectStoragePointer { future, .. } => {
                    ResolvedInputMessageContent::File(Box::new(future.await?))
                }
                // Already resolved file: data is in memory, return it directly.
                LazyFile::ObjectStorage(resolved) => {
                    ResolvedInputMessageContent::File(Box::new(resolved))
                }
            },
            LazyResolvedInputMessageContent::Unknown(unknown) => {
                ResolvedInputMessageContent::Unknown(unknown)
            }
        })
    }

    /// Converts the message content into a StoredInputMessageContent for database storage.
    /// This method optimizes file handling by:
    /// - Skipping fetches when the file is already in object storage (`ObjectStoragePointer`, `ObjectStorageFile`)
    /// - Only writing files that aren't already persisted (`Url`, `Base64`)
    /// This enables efficient roundtripping: data from the database can be passed back
    /// to new requests without re-fetching or re-writing files.
    pub async fn into_stored_input_message_content(
        self,
        object_store_info: &Option<ObjectStoreInfo>,
    ) -> Result<StoredInputMessageContent, Error> {
        Ok(match self {
            LazyResolvedInputMessageContent::File(file) => match *file {
                // File reference to object storage without data in memory.
                // Origin: User provided File::ObjectStorage (e.g., from list_datapoints)
                // The file is already persisted at storage_path, so we skip both fetch and write.
                // We discard the future without awaiting (avoiding the pending fetch operation)
                // and directly convert the metadata and storage_path into a StoredFile.
                LazyFile::ObjectStoragePointer {
                    metadata,
                    storage_path,
                    future: _,
                } => StoredInputMessageContent::File(Box::new(StoredFile(ObjectStoragePointer {
                    source_url: metadata.source_url,
                    mime_type: metadata.mime_type,
                    storage_path,
                    detail: metadata.detail,
                    filename: metadata.filename,
                }))),
                // File reference to object storage with data in memory.
                // Origin: Roundtripping from database (e.g., list_inferences → update_datapoints)
                //         User provided File::ObjectStorageFile with file data attached
                // The file is already persisted at storage_path, so we skip the write.
                // We drop the in-memory data and keep only the metadata for storage.
                LazyFile::ObjectStorage(resolved) => {
                    StoredInputMessageContent::File(Box::new(StoredFile(resolved.file)))
                }
                // File from a URL that needs to be fetched and stored.
                // Origin: User provided File::Url
                // We fetch the file from the URL, then write it to object storage.
                LazyFile::Url {
                    future,
                    file_url: _,
                } => {
                    let resolved_file = future.await?;
                    let base64_file = Base64File::new(
                        resolved_file.file.source_url.clone(),
                        resolved_file.file.mime_type.clone(),
                        resolved_file.data.clone(),
                        resolved_file.file.detail.clone(),
                        resolved_file.file.filename.clone(),
                    )?;
                    write_file(
                        object_store_info,
                        base64_file,
                        resolved_file.file.storage_path.clone(),
                    )
                    .await?;

                    StoredInputMessageContent::File(Box::new(StoredFile(resolved_file.file)))
                }
                // Base64-encoded file data that needs to be written to storage.
                // Origin: User provided File::Base64 (fresh file data)
                // We write the base64 data to object storage.
                // The PendingObjectStoreFile wrapper type signals this write requirement.
                LazyFile::Base64(pending) => {
                    let base64_file = Base64File::new(
                        pending.0.file.source_url.clone(),
                        pending.0.file.mime_type.clone(),
                        pending.0.data.clone(),
                        pending.0.file.detail.clone(),
                        pending.0.file.filename.clone(),
                    )?;
                    write_file(
                        object_store_info,
                        base64_file,
                        pending.0.file.storage_path.clone(),
                    )
                    .await?;

                    StoredInputMessageContent::File(Box::new(StoredFile(pending.0.file)))
                }
            },
            // All other cases delegate to the "resolve" case, which is mostly just a type conversion.
            other => other.resolve().await?.into_stored_input_message_content(),
        })
    }
}

/// InputMessage and Role are our representation of the input sent by the client
/// prior to any processing into LLM representations below.
/// `InputMessage` has a custom deserializer that addresses legacy data formats that we used to support (see input_message.rs).
#[derive(Clone, Debug, Serialize, PartialEq, ts_rs::TS, JsonSchema)]
#[ts(export, optional_fields)]
#[export_schema]
pub struct InputMessage {
    pub role: Role,
    pub content: Vec<InputMessageContent>,
}

impl From<StoredInputMessage> for InputMessage {
    fn from(stored_input_message: StoredInputMessage) -> Self {
        InputMessage {
            role: stored_input_message.role,
            content: stored_input_message
                .content
                .into_iter()
                .map(StoredInputMessageContent::into_input_message_content)
                .collect(),
        }
    }
}

/// A newtype wrapper around Map<String, Value> for template and system arguments
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, ts_rs::TS)]
#[ts(export)]
#[serde(transparent)]
pub struct Arguments(
    // This type cannot be a Python dataclass because it's equivalent to a Map with arbitrary keys, and Python dataclasses
    // need its slots specified. So all references to this type need to be `Map<String, Value>` in JSON schemas.
    pub Map<String, Value>,
);

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, ts_rs::TS, JsonSchema)]
#[ts(export)]
#[serde(deny_unknown_fields)]
#[export_schema]
pub struct Template {
    pub name: String,
    #[schemars(with = "Map<String, Value>")]
    pub arguments: Arguments,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, ts_rs::TS, JsonSchema)]
#[serde(untagged)]
#[ts(export)]
#[export_schema]
pub enum System {
    Text(String),
    #[schemars(with = "Map<String, Value>")]
    Template(Arguments),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, ts_rs::TS, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
#[ts(export, tag = "type", rename_all = "snake_case")]
#[export_schema]
pub enum InputMessageContent {
    #[schemars(title = "InputMessageContentText")]
    Text(Text),
    #[schemars(title = "InputMessageContentTemplate")]
    Template(Template),
    #[schemars(title = "InputMessageContentToolCall")]
    ToolCall(ToolCallWrapper),
    #[schemars(title = "InputMessageContentToolResult")]
    ToolResult(ToolResult),
    #[schemars(title = "InputMessageContentRawText")]
    RawText(RawText),
    #[schemars(title = "InputMessageContentThought")]
    Thought(Thought),
    #[serde(alias = "image")]
    #[schemars(title = "InputMessageContentFile")]
    File(File),
    /// An unknown content block type, used to allow passing provider-specific
    /// content blocks (e.g. Anthropic's `redacted_thinking`) in and out
    /// of TensorZero.
    /// The `data` field holds the original content block from the provider,
    /// without any validation or transformation by TensorZero.
    #[schemars(title = "InputMessageContentUnknown")]
    Unknown(Unknown),
}

#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(untagged, deny_unknown_fields)]
#[derive(ts_rs::TS)]
#[ts(export)]
pub enum TextKind {
    Text { text: String },
    Arguments { arguments: Arguments },
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
                    serde::de::Error::custom(format!("Error deserializing `text`: {e}"))
                })?,
            }),
            "arguments" => Ok(TextKind::Arguments {
                arguments: Arguments(serde_json::from_value(value).map_err(|e| {
                    serde::de::Error::custom(format!("Error deserializing `arguments`: {e}"))
                })?),
            }),
            _ => Err(serde::de::Error::custom(format!(
                "Unknown key `{key}` in text content"
            ))),
        }
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
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS, JsonSchema)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(get_all, str))]
#[serde(deny_unknown_fields)]
#[export_schema]
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

/// Struct that represents raw text content that should be passed directly to the model
/// without any template processing or validation
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS, JsonSchema)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(get_all, str))]
#[serde(deny_unknown_fields)]
#[export_schema]
pub struct RawText {
    pub value: String,
}

impl std::fmt::Display for RawText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl RateLimitedInputContent for RawText {
    fn estimated_input_token_usage(&self) -> u64 {
        get_estimated_tokens(&self.value)
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl RawText {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}

/// Struct that represents an unknown provider-specific content block.
/// We pass this along as-is without any validation or transformation.
#[derive(Clone, Debug, PartialEq, Serialize, ts_rs::TS, JsonSchema)]
#[ts(export, optional_fields)]
#[cfg_attr(feature = "pyo3", pyclass)]
#[export_schema]
pub struct Unknown {
    /// The underlying content block to be passed to the model provider.
    pub data: Value,
    /// A model name in your configuration (e.g. `my_gpt_5`) or a short-hand model name (e.g. `openai::gpt-5`)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    /// A provider name for the model you specified (e.g. `my_openai`)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_name: Option<String>,
}

/// Custom deserializer to handle legacy `model_provider_name` field.
///
/// Legacy format: `tensorzero::model_name::{model}::provider_name::{provider}`
/// Current format: separate `model_name` and `provider_name` fields
///
/// If both old and new fields are present, return an error.
/// If parsing the legacy format fails (e.g. the expected prefix/suffix markers are missing), a deserialization error is returned.
impl<'de> Deserialize<'de> for Unknown {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct UnknownDeserialize {
            data: Value,
            model_provider_name: Option<String>,
            model_name: Option<String>,
            provider_name: Option<String>,
        }

        /// Parse legacy FQN format: `tensorzero::model_name::XXX::provider_name::YYY`
        /// Uses best-effort parsing: everything between prefix and `::provider_name::` is model_name,
        /// everything after `::provider_name::` is provider_name.
        fn parse_fully_qualified_model_provider_name(
            fqn: &str,
        ) -> Result<(String, String), String> {
            const PREFIX: &str = "tensorzero::model_name::";
            const SUFFIX: &str = "::provider_name::";

            let Some(rest) = fqn.strip_prefix(PREFIX) else {
                return Err(format!(
                    "Invalid legacy `model_provider_name` format (missing prefix): {fqn}"
                ));
            };

            let Some(suffix_pos) = rest.find(SUFFIX) else {
                return Err(format!(
                    "Invalid legacy `model_provider_name` format (missing provider_name): {fqn}"
                ));
            };

            let model_name = &rest[..suffix_pos];
            let provider_name = &rest[suffix_pos + SUFFIX.len()..];

            Ok((model_name.to_string(), provider_name.to_string()))
        }

        let helper = UnknownDeserialize::deserialize(deserializer)?;

        // If new fields are present, use them directly
        if helper.model_name.is_some() || helper.provider_name.is_some() {
            if helper.model_provider_name.is_some() {
                return Err(serde::de::Error::custom(
                    "Cannot specify both `model_provider_name` and `model_name`/`provider_name`",
                ));
            }
            return Ok(Unknown {
                data: helper.data,
                model_name: helper.model_name,
                provider_name: helper.provider_name,
            });
        }

        // Parse legacy format if present
        let (model_name, provider_name) = match helper.model_provider_name {
            Some(ref fqn) => {
                let (m, p) = parse_fully_qualified_model_provider_name(fqn)
                    .map_err(serde::de::Error::custom)?;
                (Some(m), Some(p))
            }
            None => (None, None),
        };

        Ok(Unknown {
            data: helper.data,
            model_name,
            provider_name,
        })
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl Unknown {
    #[getter]
    pub fn data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use crate::inference::types::pyo3_helpers::serialize_to_dict;
        serialize_to_dict(py, &self.data).map(|p| p.into_bound(py))
    }

    #[getter]
    pub fn model_name(&self) -> Option<String> {
        self.model_name.clone()
    }

    #[getter]
    pub fn provider_name(&self) -> Option<String> {
        self.provider_name.clone()
    }
}

#[derive(ts_rs::TS, Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(get_all))]
#[serde(tag = "type", rename_all = "snake_case")]
#[export_schema]
pub enum ThoughtSummaryBlock {
    #[schemars(title = "ThoughtSummaryBlockSummaryText")]
    SummaryText { text: String },
}

/// Struct that represents a model's reasoning
#[derive(ts_rs::TS, Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(get_all))]
#[export_schema]
pub struct Thought {
    #[ts(optional)]
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
    Unknown(Unknown),
}

impl ContentBlock {
    pub async fn into_stored_content_block(self) -> Result<StoredContentBlock, Error> {
        match self {
            ContentBlock::Text(text) => Ok(StoredContentBlock::Text(text)),
            ContentBlock::ToolCall(tool_call) => Ok(StoredContentBlock::ToolCall(tool_call)),
            ContentBlock::ToolResult(tool_result) => {
                Ok(StoredContentBlock::ToolResult(tool_result))
            }
            ContentBlock::File(file) => {
                let resolved_file = file.resolve().await?.clone().into_owned();
                Ok(StoredContentBlock::File(Box::new(
                    File::ObjectStorage(resolved_file).into_stored_file()?,
                )))
            }
            ContentBlock::Thought(thought) => Ok(StoredContentBlock::Thought(thought)),
            ContentBlock::Unknown(unknown) => Ok(StoredContentBlock::Unknown(unknown)),
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
            ContentBlock::Unknown(unknown) => Ok(ResolvedContentBlock::Unknown(unknown)),
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
            ContentBlock::Unknown(_) => 0,
        }
    }
}

/// The version of `ContentBlock` that is stored in ClickHouse.
/// This is almost identical to `ContentBlock`, but without `File` data.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StoredContentBlock {
    Text(Text),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    #[serde(alias = "image")]
    File(Box<StoredFile>),
    Thought(Thought),
    Unknown(Unknown),
}

/// Like `ContentBlock`, but stores an in-memory `ObjectStorageFile` instead of a `LazyFile`
/// As a result, it can implement both `Serialize` and `Deserialize`
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResolvedContentBlock {
    Text(Text),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    File(Box<ObjectStorageFile>),
    Thought(Thought),
    Unknown(Unknown),
}

impl ResolvedContentBlock {
    /// Converts a `ResolvedContentBlock` into a `ContentBlock`.
    pub fn into_content_block(self) -> ContentBlock {
        match self {
            ResolvedContentBlock::Text(text) => ContentBlock::Text(text),
            ResolvedContentBlock::ToolCall(tool_call) => ContentBlock::ToolCall(tool_call),
            ResolvedContentBlock::ToolResult(tool_result) => ContentBlock::ToolResult(tool_result),
            ResolvedContentBlock::File(resolved) => {
                ContentBlock::File(Box::new(LazyFile::ObjectStorage(*resolved)))
            }
            ResolvedContentBlock::Thought(thought) => ContentBlock::Thought(thought),
            ResolvedContentBlock::Unknown(unknown) => ContentBlock::Unknown(unknown),
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
    Unknown,
}

/// Defines the types of content block that can come out of a model provider
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockOutput {
    Text(Text),
    ToolCall(ToolCall),
    Thought(Thought),
    Unknown(Unknown),
}

/// Defines the types of content block that can come from a `chat` function
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS, JsonSchema)]
#[ts(export, optional_fields)]
#[serde(tag = "type", rename_all = "snake_case")]
#[export_schema]
pub enum ContentBlockChatOutput {
    #[schemars(title = "ContentBlockChatOutputText")]
    Text(Text),
    #[schemars(title = "ContentBlockChatOutputToolCall")]
    ToolCall(InferenceResponseToolCall),
    #[schemars(title = "ContentBlockChatOutputThought")]
    Thought(Thought),
    #[schemars(title = "ContentBlockChatOutputUnknown")]
    Unknown(Unknown),
}

impl ContentBlockChatOutput {
    /// Validates a `ContentBlockChatOutput` and re-validate and re-parse structured fields.
    /// (e.g. ToolCallOutput.name and .arguments). Returns a new `ContentBlockChatOutput` with the validated fields.
    ///
    /// This is used in CreateChatDatapointRequest, which accepts a ContentBlockChatOutput. In these cases where a
    /// user specifies it, we cannot trust raw and parsed values agree, and we use the raw fields as the source of truth
    /// and re-validate.
    pub async fn into_validated(
        self,
        tool_call_config: Option<&ToolCallConfig>,
    ) -> ContentBlockChatOutput {
        if let ContentBlockChatOutput::ToolCall(input_tool_call) = self {
            let unvalidated_tool_call = ToolCall {
                name: input_tool_call.raw_name,
                arguments: input_tool_call.raw_arguments,
                id: input_tool_call.id,
            };
            let validated_tool_call =
                InferenceResponseToolCall::new(unvalidated_tool_call, tool_call_config).await;
            ContentBlockChatOutput::ToolCall(validated_tool_call)
        } else {
            self
        }
    }
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
    #[serde(flatten)]
    pub inference_params_v2: ChatCompletionInferenceParamsV2,
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
            inference_params_v2: _,
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
#[derive(Clone, Debug, Serialize, Deserialize, ts_rs::TS)]
#[cfg_attr(any(feature = "e2e_tests", test), derive(PartialEq))]
#[ts(export)]
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
        match self.usage.total_tokens() {
            Some(tokens) => Ok(RateLimitResourceUsage::Exact {
                model_inferences: 1,
                tokens: tokens as u64,
            }),
            None => Ok(RateLimitResourceUsage::UnderEstimate {
                model_inferences: 1,
                tokens: 0,
            }),
        }
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
                input_tokens: Some(0),
                output_tokens: Some(0),
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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, ts_rs::TS, JsonSchema)]
#[export_schema]
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
    #[serde(deserialize_with = "deserialize_optional_tool_info")]
    #[serde(flatten)]
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
        InputMessageContent::Text(Text { text })
    }
}

#[cfg(test)]
impl From<String> for ResolvedInputMessageContent {
    fn from(text: String) -> Self {
        ResolvedInputMessageContent::Text(Text { text })
    }
}

#[cfg(test)]
impl From<String> for LazyResolvedInputMessageContent {
    fn from(text: String) -> Self {
        LazyResolvedInputMessageContent::Text(Text { text })
    }
}

#[cfg(any(test, feature = "e2e_tests"))]
impl From<String> for ContentBlockChatOutput {
    fn from(text: String) -> Self {
        ContentBlockChatOutput::Text(Text { text })
    }
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
        let input_tokens = match result.usage.input_tokens {
            Some(tokens) if tokens > 0 => Some(tokens),
            _ => None,
        };
        let output_tokens = match result.usage.output_tokens {
            Some(tokens) if tokens > 0 => Some(tokens),
            _ => None,
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

    /// Aggregates the usage of all model inference results, considering cached results.
    /// If any of the values are None, the total usage is considered as None (via `sum_usage_strict`).
    pub fn usage_considering_cached(&self) -> Usage {
        Usage::sum_iter_strict(
            self.model_inference_results()
                .iter()
                .map(ModelInferenceResponseWithMetadata::usage_considering_cached),
        )
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
        json_mode: Option<JsonMode>,
    ) -> Self {
        let created = current_timestamp();
        let content = parse_chat_output(raw_content, tool_config, json_mode).await;
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
    json_mode: Option<JsonMode>,
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
                // If using json_mode="tool", convert tool call arguments to text
                if json_mode == Some(JsonMode::Tool) {
                    output.push(ContentBlockChatOutput::Text(Text {
                        text: tool_call.arguments,
                    }));
                } else {
                    // Normal tool call handling
                    let inference_response_tool_call =
                        InferenceResponseToolCall::new(tool_call, tool_config).await;
                    output.push(ContentBlockChatOutput::ToolCall(
                        inference_response_tool_call,
                    ));
                }
            }
            ContentBlockOutput::Thought(thought) => {
                output.push(ContentBlockChatOutput::Thought(thought));
            }
            ContentBlockOutput::Unknown(unknown) => {
                output.push(ContentBlockChatOutput::Unknown(unknown));
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

impl From<InferenceResponseToolCall> for ToolCall {
    fn from(output: InferenceResponseToolCall) -> Self {
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
            ContentBlockChatOutput::ToolCall(inference_response_tool_call) => {
                ContentBlock::ToolCall(inference_response_tool_call.into())
            }
            ContentBlockChatOutput::Thought(thought) => ContentBlock::Thought(thought),
            ContentBlockChatOutput::Unknown(unknown) => ContentBlock::Unknown(unknown),
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
            ContentBlockChatOutput::Unknown(unknown) => ContentBlockOutput::Unknown(unknown),
        }
    }
}

impl From<JsonMode> for ModelInferenceRequestJsonMode {
    fn from(json_enforcement: JsonMode) -> Self {
        match json_enforcement {
            JsonMode::On => ModelInferenceRequestJsonMode::On,
            JsonMode::Strict => ModelInferenceRequestJsonMode::Strict,
            JsonMode::Tool => ModelInferenceRequestJsonMode::Off,
            JsonMode::Off => ModelInferenceRequestJsonMode::Off,
        }
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
            "Error deserializing replacement config: `delete` must be `true`, or not set",
        ));
    }
    Ok(())
}

// Field-aware versions for struct fields (not enum variants)
#[expect(clippy::trivially_copy_pass_by_ref)]
pub(super) fn serialize_delete_field<S>(_: &(), s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    true.serialize(s)
}

pub(super) fn deserialize_delete_field<'de, D>(d: D) -> Result<(), D::Error>
where
    D: Deserializer<'de>,
{
    let val = bool::deserialize(d)?;
    if !val {
        return Err(D::Error::custom(
            "Error deserializing replacement config: `delete` must be `true`, or not set",
        ));
    }
    Ok(())
}

pub(super) fn schema_for_delete_field(_gen: &mut schemars::SchemaGenerator) -> schemars::Schema {
    let mut map = Map::new();
    map.insert("type".to_owned(), Value::String("boolean".to_owned()));
    map.insert("const".to_owned(), Value::Bool(true));
    schemars::Schema::from(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jsonschema_util::DynamicJSONSchema;
    use crate::providers::test_helpers::get_temperature_tool_config;
    use crate::tool::{DynamicToolConfig, FunctionToolConfig, ToolChoice};
    use serde_json::json;
    use tokio::time::Instant;

    #[tokio::test]
    async fn test_create_chat_inference_response() {
        // Case 1: No output schema
        let inference_id = Uuid::now_v7();
        let content = vec!["Hello, world!".to_string().into()];
        let usage = Usage {
            input_tokens: Some(10),
            output_tokens: Some(20),
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
            tool_choice: ToolChoice::None,
            ..ToolCallConfig::with_tools_available(
                vec![],
                vec![FunctionToolConfig::Dynamic(DynamicToolConfig {
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
            )
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
            tool_choice: ToolChoice::None,
            ..ToolCallConfig::with_tools_available(
                vec![],
                vec![FunctionToolConfig::Dynamic(DynamicToolConfig {
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
            )
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
            InputMessageContent::Text(text) => {
                assert_eq!(&text.text, "Hello, world!");
            }
            _ => panic!("Expected Text content: {message:?}"),
        }

        // Test case for object content (should be converted to Template)
        let input = json!({
            "role": "assistant",
            "content": {"key": "value"}
        });
        let message: InputMessage = serde_json::from_value(input).unwrap();
        assert_eq!(message.role, Role::Assistant);
        assert_eq!(message.content.len(), 1);
        match &message.content[0] {
            InputMessageContent::Template(template) => {
                assert_eq!(template.name, "assistant");
                assert_eq!(
                    &template.arguments.0,
                    json!({"key": "value"}).as_object().unwrap()
                );
            }
            _ => panic!("Expected Template content"),
        }

        // Test case for multiple content items
        let input = json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "tool_call", "id": "123", "name": "test_tool", "arguments": "{}"}
            ]
        });
        let message: InputMessage = serde_json::from_value(input).unwrap();
        assert_eq!(message.role, Role::User);
        assert_eq!(message.content.len(), 2);
        match &message.content[0] {
            InputMessageContent::Text(text) => {
                assert_eq!(&text.text, "Hello");
            }
            _ => panic!("Expected Text content"),
        }
        match &message.content[1] {
            InputMessageContent::ToolCall(wrapper) => match wrapper {
                ToolCallWrapper::ToolCall(tc) => {
                    assert_eq!(tc.id, "123");
                    assert_eq!(tc.name, "test_tool");
                    assert_eq!(tc.arguments, "{}");
                }
                ToolCallWrapper::InferenceResponseToolCall(tc) => {
                    assert_eq!(tc.id, "123");
                    assert_eq!(tc.name, Some("test_tool".to_string()));
                    assert_eq!(tc.arguments, Some(json!("{}")));
                    assert_eq!(tc.raw_name, "test_tool");
                    assert_eq!(tc.raw_arguments, "{}");
                }
            },
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
            InputMessageContent::Template(Template { name, arguments }) => {
                assert_eq!(name, "user");
                assert_eq!(
                    &arguments.0,
                    json!({"complex": "json", "with": ["nested", "array"]})
                        .as_object()
                        .unwrap()
                );
            }
            _ => panic!("Expected Text content with JSON object"),
        }
        match &message.content[1] {
            InputMessageContent::ToolCall(wrapper) => match wrapper {
                ToolCallWrapper::ToolCall(tc) => {
                    assert_eq!(tc.id, "456");
                    assert_eq!(tc.name, "another_tool");
                    assert_eq!(tc.arguments, json!({"key":"value"}).to_string());
                }
                ToolCallWrapper::InferenceResponseToolCall(tc) => {
                    assert_eq!(tc.id, "456");
                    assert_eq!(tc.name, Some("another_tool".to_string()));
                    assert_eq!(tc.arguments, Some(json!({"key":"value"})));
                }
            },
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

    /// Test that usage_considering_cached properly propagates None values
    /// If any of the model inference results have None for input_tokens or output_tokens,
    /// the aggregated result should also have None for those fields
    #[tokio::test]
    async fn test_usage_considering_cached_none_propagation() {
        let inference_id = Uuid::now_v7();

        // Helper function to create a ModelInferenceResponseWithMetadata with specified usage
        let create_model_response =
            |usage: Usage, cached: bool| ModelInferenceResponseWithMetadata {
                id: Uuid::now_v7(),
                created: Instant::now().elapsed().as_secs(),
                system: None,
                input_messages: RequestMessagesOrBatch::Message(vec![]),
                output: vec!["test".to_string().into()],
                raw_request: String::new(),
                raw_response: String::new(),
                usage,
                latency: Latency::NonStreaming {
                    response_time: Duration::default(),
                },
                finish_reason: None,
                model_provider_name: "test_provider".into(),
                model_name: "test_model".into(),
                cached,
            };

        // Test Case 1: All values are Some() - should aggregate correctly
        let model_responses_all_some = vec![
            create_model_response(
                Usage {
                    input_tokens: Some(10),
                    output_tokens: Some(20),
                },
                false,
            ),
            create_model_response(
                Usage {
                    input_tokens: Some(15),
                    output_tokens: Some(25),
                },
                false,
            ),
        ];
        let chat_result_all_some = ChatInferenceResult::new(
            inference_id,
            vec!["test".to_string().into()],
            model_responses_all_some,
            None,
            InferenceParams::default(),
            None,
            None,
        )
        .await;
        let result_all_some = InferenceResult::Chat(chat_result_all_some);
        let usage_all_some = result_all_some.usage_considering_cached();
        assert_eq!(usage_all_some.input_tokens, Some(25));
        assert_eq!(usage_all_some.output_tokens, Some(45));

        // Test Case 2: Some input_tokens are None - should propagate None for input_tokens
        let model_responses_input_none = vec![
            create_model_response(
                Usage {
                    input_tokens: Some(10),
                    output_tokens: Some(20),
                },
                false,
            ),
            create_model_response(
                Usage {
                    input_tokens: None,
                    output_tokens: Some(25),
                },
                false,
            ),
        ];
        let chat_result_input_none = ChatInferenceResult::new(
            inference_id,
            vec!["test".to_string().into()],
            model_responses_input_none,
            None,
            InferenceParams::default(),
            None,
            None,
        )
        .await;
        let result_input_none = InferenceResult::Chat(chat_result_input_none);
        let usage_input_none = result_input_none.usage_considering_cached();
        assert_eq!(usage_input_none.input_tokens, None);
        assert_eq!(usage_input_none.output_tokens, Some(45));

        // Test Case 3: Some output_tokens are None - should propagate None for output_tokens
        let model_responses_output_none = vec![
            create_model_response(
                Usage {
                    input_tokens: Some(10),
                    output_tokens: Some(20),
                },
                false,
            ),
            create_model_response(
                Usage {
                    input_tokens: Some(15),
                    output_tokens: None,
                },
                false,
            ),
        ];
        let chat_result_output_none = ChatInferenceResult::new(
            inference_id,
            vec!["test".to_string().into()],
            model_responses_output_none,
            None,
            InferenceParams::default(),
            None,
            None,
        )
        .await;
        let result_output_none = InferenceResult::Chat(chat_result_output_none);
        let usage_output_none = result_output_none.usage_considering_cached();
        assert_eq!(usage_output_none.input_tokens, Some(25));
        assert_eq!(usage_output_none.output_tokens, None);

        // Test Case 4: All values are None - should result in all None
        let model_responses_all_none = vec![
            create_model_response(
                Usage {
                    input_tokens: None,
                    output_tokens: None,
                },
                false,
            ),
            create_model_response(
                Usage {
                    input_tokens: None,
                    output_tokens: None,
                },
                false,
            ),
        ];
        let chat_result_all_none = ChatInferenceResult::new(
            inference_id,
            vec!["test".to_string().into()],
            model_responses_all_none,
            None,
            InferenceParams::default(),
            None,
            None,
        )
        .await;
        let result_all_none = InferenceResult::Chat(chat_result_all_none);
        let usage_all_none = result_all_none.usage_considering_cached();
        assert_eq!(usage_all_none.input_tokens, None);
        assert_eq!(usage_all_none.output_tokens, None);

        // Test Case 5: Mixed cached and non-cached with None values
        // Cached results return Usage { input_tokens: Some(0), output_tokens: Some(0) }
        let model_responses_mixed = vec![
            create_model_response(
                Usage {
                    input_tokens: Some(10),
                    output_tokens: Some(20),
                },
                true,
            ), // This will be treated as 0/0 due to cached=true
            create_model_response(
                Usage {
                    input_tokens: None,
                    output_tokens: Some(25),
                },
                false,
            ),
        ];
        let chat_result_mixed = ChatInferenceResult::new(
            inference_id,
            vec!["test".to_string().into()],
            model_responses_mixed,
            None,
            InferenceParams::default(),
            None,
            None,
        )
        .await;
        let result_mixed = InferenceResult::Chat(chat_result_mixed);
        let usage_mixed = result_mixed.usage_considering_cached();
        assert_eq!(usage_mixed.input_tokens, None); // None propagates
        assert_eq!(usage_mixed.output_tokens, Some(25)); // 0 (cached) + 25
    }

    #[test]
    fn test_unknown_deserialize() {
        // New format
        let u: Unknown =
            serde_json::from_value(json!({"data": {}, "model_name": "m", "provider_name": "p"}))
                .unwrap();
        assert_eq!(u.model_name.as_deref(), Some("m"));
        assert_eq!(u.provider_name.as_deref(), Some("p"));

        // Provider-agnostic (no targeting)
        let u: Unknown = serde_json::from_value(json!({"data": {}})).unwrap();
        assert!(u.model_name.is_none() && u.provider_name.is_none());

        // Legacy FQN - simple
        let u: Unknown = serde_json::from_value(json!({"data": {}, "model_provider_name": "tensorzero::model_name::m::provider_name::p"})).unwrap();
        assert_eq!(u.model_name.as_deref(), Some("m"));
        assert_eq!(u.provider_name.as_deref(), Some("p"));

        // Legacy FQN - model with colons (e.g. dummy::echo)
        let u: Unknown = serde_json::from_value(json!({"data": {}, "model_provider_name": "tensorzero::model_name::dummy::echo::provider_name::p"})).unwrap();
        assert_eq!(u.model_name.as_deref(), Some("dummy::echo"));

        // Invalid legacy FQN - errors
        assert!(serde_json::from_value::<Unknown>(
            json!({"data": {}, "model_provider_name": "bad"})
        )
        .is_err());
        assert!(serde_json::from_value::<Unknown>(
            json!({"data": {}, "model_provider_name": "tensorzero::model_name::m"})
        )
        .is_err());

        // Conflict: both old and new fields
        assert!(serde_json::from_value::<Unknown>(json!({"data": {}, "model_provider_name": "tensorzero::model_name::m::provider_name::p", "model_name": "x"})).is_err());
    }
}
