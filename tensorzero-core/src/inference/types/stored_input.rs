use futures::future::try_join_all;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::Config;
use crate::endpoints::object_storage::get_object;
use crate::error::{Error, ErrorDetails};
use crate::inference::types::file::Base64FileMetadata;
use crate::inference::types::storage::StoragePath;
use crate::inference::types::Base64File;
use crate::inference::types::FileWithPath;
use crate::inference::types::ResolvedInput;
use crate::inference::types::ResolvedInputMessage;
use crate::inference::types::ResolvedInputMessageContent;
use crate::inference::types::StoredContentBlock;
use crate::inference::types::TemplateInput;
use crate::inference::types::Text;
use crate::inference::types::{Role, Thought, ToolCall, ToolResult};

#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::serialize_to_dict;
#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::stored_input_message_content_to_python;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

/// The input type that we directly store in ClickHouse.
/// As soon as we retrieve it, we should convert it to a `StoredInput` below.
#[derive(Clone, Debug, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct MaybeLegacyStoredInput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Value>,
    #[serde(default)]
    pub messages: Vec<MaybeLegacyStoredInputMessage>,
}

/// The input type that we directly store in ClickHouse.
/// This is almost identical to `ResolvedInput`, but without `File` data.
/// Only the object-storage path is actually stored in clickhouse
/// (which can be used to re-fetch the file and produce a `ResolvedInput`).
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct StoredInput {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(test, ts(optional))]
    pub system: Option<Value>,
    #[serde(default)]
    pub messages: Vec<StoredInputMessage>,
}

/// Abstracts over a `Config` (without an actual embedded gateway)
/// and an http `Client, so that we can call `reresolve` from `StoredInference`
/// and `evaluations`
pub trait StoragePathResolver {
    async fn resolve(&self, storage_path: StoragePath) -> Result<String, Error>;
}

impl StoragePathResolver for Config {
    async fn resolve(&self, storage_path: StoragePath) -> Result<String, Error> {
        Ok(get_object(self.object_store_info.as_ref(), storage_path)
            .await?
            .data)
    }
}

impl StoredInput {
    /// Converts a `StoredInput` to a `ResolvedInput` by fetching the file data
    /// for any nested `File`s.
    pub async fn reresolve(
        self,
        resolver: &impl StoragePathResolver,
    ) -> Result<ResolvedInput, Error> {
        Ok(ResolvedInput {
            system: self.system,
            messages: try_join_all(
                self.messages
                    .into_iter()
                    .map(|message| message.reresolve(resolver)),
            )
            .await?,
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MaybeLegacyStoredInputMessage {
    pub role: Role,
    pub content: Vec<MaybeLegacyStoredInputMessageContent>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct StoredInputMessage {
    pub role: Role,
    pub content: Vec<StoredInputMessageContent>,
}

impl StoredInputMessage {
    pub async fn reresolve(
        self,
        resolver: &impl StoragePathResolver,
    ) -> Result<ResolvedInputMessage, Error> {
        Ok(ResolvedInputMessage {
            role: self.role,
            content: try_join_all(
                self.content
                    .into_iter()
                    .map(|content| content.reresolve(resolver)),
            )
            .await?,
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MaybeLegacyStoredInputMessageContent {
    Text {
        value: Value,
    },
    Template(TemplateInput),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    RawText {
        value: String,
    },
    Thought(Thought),
    #[serde(alias = "image")]
    File(Box<StoredFile>),
    Unknown {
        data: Value,
        model_provider_name: Option<String>,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub enum StoredInputMessageContent {
    Text(Text),
    Template(TemplateInput),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    RawText {
        value: String,
    },
    Thought(Thought),
    #[serde(alias = "image")]
    File(Box<StoredFile>),
    Unknown {
        data: Value,
        model_provider_name: Option<String>,
    },
    // We may extend this in the future to include other types of content
}

impl StoredInputMessageContent {
    pub async fn reresolve(
        self,
        resolver: &impl StoragePathResolver,
    ) -> Result<ResolvedInputMessageContent, Error> {
        match self {
            StoredInputMessageContent::Text(text) => Ok(ResolvedInputMessageContent::Text(text)),
            StoredInputMessageContent::Template(template) => {
                Ok(ResolvedInputMessageContent::Template(template))
            }
            StoredInputMessageContent::ToolCall(tool_call) => {
                Ok(ResolvedInputMessageContent::ToolCall(tool_call))
            }
            StoredInputMessageContent::ToolResult(tool_result) => {
                Ok(ResolvedInputMessageContent::ToolResult(tool_result))
            }
            StoredInputMessageContent::RawText { value } => {
                Ok(ResolvedInputMessageContent::RawText { value })
            }
            StoredInputMessageContent::Thought(thought) => {
                Ok(ResolvedInputMessageContent::Thought(thought))
            }
            StoredInputMessageContent::File(file) => {
                let data = resolver.resolve(file.storage_path.clone()).await?;
                Ok(ResolvedInputMessageContent::File(Box::new(FileWithPath {
                    file: Base64File {
                        url: file.file.url.clone(),
                        mime_type: file.file.mime_type.clone(),
                        data,
                    },
                    storage_path: file.storage_path.clone(),
                })))
            }
            StoredInputMessageContent::Unknown {
                data,
                model_provider_name,
            } => Ok(ResolvedInputMessageContent::Unknown {
                data,
                model_provider_name,
            }),
        }
    }
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(test, ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(get_all, str))]
pub struct StoredFile {
    #[serde(alias = "image")]
    pub file: Base64FileMetadata,
    pub storage_path: StoragePath,
}

impl std::fmt::Display for StoredInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl std::fmt::Display for StoredInputMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl std::fmt::Display for StoredFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl StoredInputMessage {
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
                stored_input_message_content_to_python(py, content.clone())
                    .map(|pyobj| pyobj.into_bound(py))
            })
            .collect()
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl StoredInput {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }

    #[getter]
    pub fn get_system<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(serialize_to_dict(py, self.system.clone())?.into_bound(py))
    }

    #[getter]
    pub fn get_messages(&self) -> Vec<StoredInputMessage> {
        self.messages.clone()
    }
}

/// The message type that we directly store in ClickHouse.
/// This is almost identical to `RequestMessage`, but without `File` data.
/// Only the object-storage path is actually stored in clickhouse
/// The `RequestMessage/StoredRequestMessage` pair is the model-level equivalent
/// of `ResolvedInput/StoredInput`
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StoredRequestMessage {
    pub role: Role,
    pub content: Vec<StoredContentBlock>,
}

/// Helper function to convert legacy content to modern content format
fn convert_legacy_content(
    content: MaybeLegacyStoredInputMessageContent,
    role: Role,
) -> Result<StoredInputMessageContent, Error> {
    match content {
        // Convert the legacy `value` field to the proper content blocks
        MaybeLegacyStoredInputMessageContent::Text { value } => match value {
            Value::String(text) => Ok(StoredInputMessageContent::Text(Text { text })),
            Value::Object(obj) => Ok(StoredInputMessageContent::Template(TemplateInput {
                name: role.to_string(),
                arguments: obj,
            })),
            _ => Err(Error::new(ErrorDetails::InvalidMessage {
                message: r#"The `value` field in a `{"type": "text", "value": ... }` content block must be a string or object"#.to_string(),
            })),
        },
        MaybeLegacyStoredInputMessageContent::Template(template) => {
            Ok(StoredInputMessageContent::Template(template))
        }
        MaybeLegacyStoredInputMessageContent::ToolCall(tool_call) => {
            Ok(StoredInputMessageContent::ToolCall(tool_call))
        }
        MaybeLegacyStoredInputMessageContent::ToolResult(tool_result) => {
            Ok(StoredInputMessageContent::ToolResult(tool_result))
        }
        MaybeLegacyStoredInputMessageContent::RawText { value } => {
            Ok(StoredInputMessageContent::RawText { value })
        }
        MaybeLegacyStoredInputMessageContent::Thought(thought) => {
            Ok(StoredInputMessageContent::Thought(thought))
        }
        MaybeLegacyStoredInputMessageContent::File(file) => {
            Ok(StoredInputMessageContent::File(file))
        }
        MaybeLegacyStoredInputMessageContent::Unknown {
            data,
            model_provider_name,
        } => Ok(StoredInputMessageContent::Unknown {
            data,
            model_provider_name,
        }),
    }
}

/// Helper function to convert a legacy message to a modern message
fn convert_legacy_message(
    message: MaybeLegacyStoredInputMessage,
) -> Result<StoredInputMessage, Error> {
    let role = message.role;
    let content: Vec<StoredInputMessageContent> = message
        .content
        .into_iter()
        .map(|c| convert_legacy_content(c, role))
        .collect::<Result<Vec<_>, Error>>()?;

    Ok(StoredInputMessage { role, content })
}

/// Convert:
///    `MaybeLegacyStoredInput`: how we store in the database, with potentially legacy data schemas)
///  ->`StoredInput`: a cleaned version that fixed legacy data schemas
impl TryFrom<MaybeLegacyStoredInput> for StoredInput {
    type Error = Error;

    fn try_from(input: MaybeLegacyStoredInput) -> Result<Self, Error> {
        let messages: Vec<StoredInputMessage> = input
            .messages
            .into_iter()
            .map(convert_legacy_message)
            .collect::<Result<Vec<_>, Error>>()?;

        Ok(Self {
            system: input.system,
            messages,
        })
    }
}
