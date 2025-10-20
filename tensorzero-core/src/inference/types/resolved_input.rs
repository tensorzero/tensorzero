use std::borrow::Cow;
use std::future::Future;
use std::pin::Pin;

use futures::future::Shared;
use futures::FutureExt;
use mime::MediaType;
use object_store::{PutMode, PutOptions};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use url::Url;

use super::{storage::StoragePath, Base64File, Role, Thought};
use crate::config::{Config, ObjectStoreInfo};
use crate::error::{Error, ErrorDetails};
use crate::inference::types::file::Base64FileMetadata;
use crate::inference::types::stored_input::StoredFile;
use crate::inference::types::stored_input::{
    StoredInput, StoredInputMessage, StoredInputMessageContent,
};
use crate::inference::types::{RequestMessage, ResolvedContentBlock, TemplateInput};
use crate::rate_limiting::RateLimitedInputContent;
use crate::tool::{ToolCall, ToolResult};

#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::{
    resolved_content_block_to_python, resolved_input_message_content_to_python, serialize_to_dict,
};
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[derive(Clone, Debug)]
pub struct LazyResolvedInput {
    pub system: Option<Value>,
    pub messages: Vec<LazyResolvedInputMessage>,
}

#[derive(Clone, Debug)]
pub struct LazyResolvedInputMessage {
    pub role: Role,
    pub content: Vec<LazyResolvedInputMessageContent>,
}

// This gets serialized as part of a `ModelInferenceRequest` when we compute a cache key.
// TODO - decide on the precise caching behavior that we want for file URLs
#[derive(Clone, Debug, Serialize)]
pub enum LazyFile {
    Url {
        file_url: FileUrl,
        #[serde(skip)]
        future: FileFuture,
    },
    FileWithPath(FileWithPath),
}

#[cfg(any(test, feature = "e2e_tests"))]
impl std::cmp::PartialEq for LazyFile {
    // This is only used in tests, so it's fine to panic
    #[expect(clippy::panic)]
    fn eq(&self, _other: &Self) -> bool {
        panic!("Tried to check LazyFile equality")
    }
}

impl LazyFile {
    pub async fn resolve(&self) -> Result<Cow<'_, FileWithPath>, Error> {
        match self {
            LazyFile::Url {
                future,
                file_url: _,
            } => Ok(Cow::Owned(future.clone().await?)),
            LazyFile::FileWithPath(file) => Ok(Cow::Borrowed(file)),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct FileUrl {
    pub url: Url,
    pub mime_type: Option<MediaType>,
}

/// Holds a lazily-resolved file from a `LazyResolvedInputMessageContent::File`.
/// This is constructed as either:
/// 1. An immediately-ready future, when we're converting a `ResolvedInputMessageContent` to a `LazyResolvedInputMessageContent`
/// 2. A network fetch future, when we're resolving an image url in `InputMessageContent::File`.
///
/// This future is `Shared`, so that we can `.await` it from multiple different model providers
/// (if we're not forwarding an image url to the model provider), as well as when writing the
/// file to the object store (if enabled).
pub type FileFuture = Shared<Pin<Box<dyn Future<Output = Result<FileWithPath, Error>> + Send>>>;

#[derive(Clone, Debug)]
pub enum LazyResolvedInputMessageContent {
    Text {
        text: String,
    },
    Template(TemplateInput),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    RawText {
        value: String,
    },
    Thought(Thought),
    // When we add support for forwarding image urls to the model provider,
    // we'll store additional information here
    File(Box<LazyFile>),
    Unknown {
        data: Value,
        model_provider_name: Option<String>,
    },
    // We may extend this in the future to include other types of content
}

/// Like `Input`, but with all network resources resolved.
/// Currently, this is just used to fetch image URLs in the image input,
/// so that we always pass a base64-encoded image to the model provider.
#[derive(Clone, Debug, PartialEq)]
// TODO - should we remove the Serialize impl entirely, rather than rely on it
// for the Pyo3 'str' impl?
#[cfg_attr(any(feature = "pyo3", test), derive(Serialize))]
#[cfg_attr(any(feature = "pyo3", test), serde(deny_unknown_fields))]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct ResolvedInput {
    #[cfg_attr(
        any(feature = "pyo3", test),
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub system: Option<Value>,

    #[cfg_attr(any(feature = "pyo3", test), serde(default))]
    pub messages: Vec<ResolvedInputMessage>,
}

async fn write_file(
    object_store: &Option<ObjectStoreInfo>,
    raw: Base64File,
    storage_path: StoragePath,
) -> Result<(), Error> {
    if let Some(object_store) = object_store {
        // The store might be explicitly disabled
        if let Some(store) = object_store.object_store.as_ref() {
            let data = raw.data()?;
            let bytes = aws_smithy_types::base64::decode(data).map_err(|e| {
                Error::new(ErrorDetails::ObjectStoreWrite {
                    message: format!("Failed to decode file as base64: {e:?}"),
                    path: storage_path.clone(),
                })
            })?;
            let res = store
                .put_opts(
                    &storage_path.path,
                    bytes.into(),
                    PutOptions {
                        mode: PutMode::Create,
                        ..Default::default()
                    },
                )
                .await;
            match res {
                Ok(_) | Err(object_store::Error::AlreadyExists { .. }) => {}
                Err(e) => {
                    return Err(ErrorDetails::ObjectStoreWrite {
                        message: format!("Failed to write file to object store: {e:?}"),
                        path: storage_path.clone(),
                    }
                    .into());
                }
            }
        }
    } else {
        return Err(ErrorDetails::InternalError {
            message: "Called `write_file` with no object store configured".to_string(),
        }
        .into());
    }
    Ok(())
}

/// Produces a `StoredInput` from a `ResolvedInput` by discarding the data for any nested `File`s.
/// The data can be recovered later by re-fetching from the object store using `StoredInput::reresolve`.
impl ResolvedInput {
    pub fn into_stored_input(self) -> StoredInput {
        StoredInput {
            system: self.system,
            messages: self
                .messages
                .into_iter()
                .map(ResolvedInputMessage::into_stored_input_message)
                .collect(),
        }
    }

    pub fn into_lazy_resolved_input(self) -> LazyResolvedInput {
        LazyResolvedInput {
            system: self.system,
            messages: self
                .messages
                .into_iter()
                .map(ResolvedInputMessage::into_lazy_resolved_input_message)
                .collect(),
        }
    }

    /// Writes all the files in the input to the object store,
    /// returning a list of futures (one per file)
    #[must_use]
    pub fn write_all_files<'a>(
        self,
        config: &'a Config,
    ) -> Vec<Pin<Box<dyn Future<Output = ()> + Send + 'a>>> {
        let mut futures = Vec::new();
        if config.gateway.observability.enabled.unwrap_or(true) {
            for message in self.messages {
                for content_block in message.content {
                    if let ResolvedInputMessageContent::File(file) = content_block {
                        let FileWithPath {
                            file: raw,
                            storage_path,
                        } = *file;

                        futures.push(
                            (async {
                                if let Err(e) =
                                    write_file(&config.object_store_info, raw, storage_path).await
                                {
                                    tracing::error!("Failed to write image to object store: {e:?}");
                                }
                            })
                            .boxed(),
                        );
                    }
                }
            }
        }
        futures
    }
}

#[cfg(feature = "pyo3")]
impl std::fmt::Display for ResolvedInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl ResolvedInput {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }

    #[getter]
    pub fn get_system<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(serialize_to_dict(py, self.system.clone())?.into_bound(py))
    }

    #[getter]
    pub fn get_messages(&self) -> Vec<ResolvedInputMessage> {
        self.messages.clone()
    }
}

#[derive(Clone, Debug, PartialEq)]
// TODO - should we remove the Serialize impl entirely, rather than rely on it
// for the Pyo3 'str' impl?
#[cfg_attr(any(feature = "pyo3", test), derive(Serialize))]
#[cfg_attr(any(feature = "pyo3", test), serde(deny_unknown_fields))]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct ResolvedInputMessage {
    pub role: Role,
    pub content: Vec<ResolvedInputMessageContent>,
}

impl ResolvedInputMessage {
    pub fn into_stored_input_message(self) -> StoredInputMessage {
        StoredInputMessage {
            role: self.role,
            content: self
                .content
                .into_iter()
                .map(ResolvedInputMessageContent::into_stored_input_message_content)
                .collect(),
        }
    }

    pub fn into_lazy_resolved_input_message(self) -> LazyResolvedInputMessage {
        LazyResolvedInputMessage {
            role: self.role,
            content: self
                .content
                .into_iter()
                .map(ResolvedInputMessageContent::into_lazy_resolved_input_message_content)
                .collect(),
        }
    }
}

#[cfg(feature = "pyo3")]
impl std::fmt::Display for ResolvedInputMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl ResolvedInputMessage {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }

    #[getter]
    pub fn get_role(&self) -> String {
        self.role.to_string()
    }

    #[getter]
    pub fn get_content<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        self.content
            .iter()
            .map(|content| {
                resolved_input_message_content_to_python(py, content.clone())
                    .map(|pyobj| pyobj.into_bound(py))
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
// TODO - should we remove the Serialize impl entirely, rather than rely on it
// for the Pyo3 'str' impl?
#[cfg_attr(any(feature = "pyo3", test), derive(Serialize))]
#[cfg_attr(
    any(feature = "pyo3", test),
    serde(tag = "type", rename_all = "snake_case")
)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub enum ResolvedInputMessageContent {
    Text {
        text: String,
    },
    Template(TemplateInput),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    RawText {
        value: String,
    },
    Thought(Thought),
    #[cfg_attr(any(feature = "pyo3", test), serde(alias = "image"))]
    File(Box<FileWithPath>),
    Unknown {
        data: Value,
        model_provider_name: Option<String>,
    },
    // We may extend this in the future to include other types of content
}

impl ResolvedInputMessageContent {
    pub fn into_stored_input_message_content(self) -> StoredInputMessageContent {
        match self {
            ResolvedInputMessageContent::Text { text } => StoredInputMessageContent::Text {
                value: Value::String(text),
            },
            ResolvedInputMessageContent::Template(template) => {
                StoredInputMessageContent::Template(template)
            }
            ResolvedInputMessageContent::ToolCall(tool_call) => {
                StoredInputMessageContent::ToolCall(tool_call)
            }
            ResolvedInputMessageContent::ToolResult(tool_result) => {
                StoredInputMessageContent::ToolResult(tool_result)
            }
            ResolvedInputMessageContent::RawText { value } => {
                StoredInputMessageContent::RawText { value }
            }
            ResolvedInputMessageContent::Thought(thought) => {
                StoredInputMessageContent::Thought(thought)
            }
            ResolvedInputMessageContent::File(file) => {
                StoredInputMessageContent::File(Box::new(file.into_stored_file()))
            }
            ResolvedInputMessageContent::Unknown {
                data,
                model_provider_name,
            } => StoredInputMessageContent::Unknown {
                data,
                model_provider_name,
            },
        }
    }

    pub fn into_lazy_resolved_input_message_content(self) -> LazyResolvedInputMessageContent {
        match self {
            ResolvedInputMessageContent::Text { text } => {
                LazyResolvedInputMessageContent::Text { text }
            }
            ResolvedInputMessageContent::Template(template) => {
                LazyResolvedInputMessageContent::Template(template)
            }
            ResolvedInputMessageContent::ToolCall(tool_call) => {
                LazyResolvedInputMessageContent::ToolCall(tool_call)
            }
            ResolvedInputMessageContent::ToolResult(tool_result) => {
                LazyResolvedInputMessageContent::ToolResult(tool_result)
            }

            ResolvedInputMessageContent::RawText { value } => {
                LazyResolvedInputMessageContent::RawText { value }
            }
            ResolvedInputMessageContent::Thought(thought) => {
                LazyResolvedInputMessageContent::Thought(thought)
            }
            ResolvedInputMessageContent::File(file) => {
                LazyResolvedInputMessageContent::File(Box::new(LazyFile::FileWithPath(*file)))
            }
            ResolvedInputMessageContent::Unknown {
                data,
                model_provider_name,
            } => LazyResolvedInputMessageContent::Unknown {
                data,
                model_provider_name,
            },
        }
    }
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(test, ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(get_all, str))]
pub struct FileWithPath {
    #[serde(alias = "image")]
    pub file: Base64File,
    pub storage_path: StoragePath,
}

impl FileWithPath {
    pub fn into_stored_file(self) -> StoredFile {
        let FileWithPath {
            file:
                Base64File {
                    url,
                    mime_type,
                    data: _,
                },
            storage_path,
        } = self;
        StoredFile {
            file: Base64FileMetadata { url, mime_type },
            storage_path,
        }
    }
}

impl std::fmt::Display for FileWithPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl RateLimitedInputContent for LazyFile {
    fn estimated_input_token_usage(&self) -> u64 {
        match self {
            LazyFile::FileWithPath(FileWithPath {
                file: _,
                storage_path: _,
            }) => {}
            // Forwarding a url is inherently incompatible with input token estimation,
            // so we'll need to continue using a hardcoded value here, even if we start
            // estimating tokens LazyFile::FileWithPath
            LazyFile::Url {
                file_url: _,
                future: _,
            } => {}
        }
        10_000 // Hardcoded value for file size estimation, we will improve later
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl FileWithPath {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}

/// Like `RequestMessage`, but holds fully-resolved files instead of `LazyFile`s
#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(test, ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct ResolvedRequestMessage {
    pub role: Role,
    pub content: Vec<ResolvedContentBlock>,
}

impl ResolvedRequestMessage {
    pub fn into_request_message(self) -> RequestMessage {
        RequestMessage {
            role: self.role,
            content: self
                .content
                .into_iter()
                .map(ResolvedContentBlock::into_content_block)
                .collect(),
        }
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl ResolvedRequestMessage {
    #[getter]
    fn get_content<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use pyo3::types::PyList;

        let content = self
            .content
            .iter()
            .map(|c| resolved_content_block_to_python(py, c))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, content).map(Bound::into_any)
    }

    #[getter]
    fn get_role(&self) -> String {
        self.role.to_string()
    }

    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}
