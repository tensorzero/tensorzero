use crate::config_parser::Config;
use crate::endpoints::object_storage::get_object;
use crate::error::Error;
use crate::inference::types::file::Base64FileMetadata;
#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::stored_input_message_content_to_python;
use crate::inference::types::storage::StoragePath;
use crate::inference::types::Base64File;
use crate::inference::types::FileWithPath;
use crate::inference::types::ResolvedInput;
use crate::inference::types::ResolvedInputMessage;
use crate::inference::types::ResolvedInputMessageContent;
use crate::inference::types::{Role, Thought, ToolCall, ToolResult};
use futures::future::try_join_all;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::serialize_to_dict;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

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
    pub system: Option<Value>,
    #[serde(default)]
    pub messages: Vec<StoredInputMessage>,
}

impl StoredInput {
    /// Converts a `StoredInput` to a `ResolvedInput` by fetching the file data
    /// for any nested `File`s.
    pub async fn reresolve(self, config: &Config) -> Result<ResolvedInput, Error> {
        Ok(ResolvedInput {
            system: self.system,
            messages: try_join_all(
                self.messages
                    .into_iter()
                    .map(|message| message.reresolve(config)),
            )
            .await?,
        })
    }
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
    pub async fn reresolve(self, config: &Config) -> Result<ResolvedInputMessage, Error> {
        Ok(ResolvedInputMessage {
            role: self.role,
            content: try_join_all(
                self.content
                    .into_iter()
                    .map(|content| content.reresolve(config)),
            )
            .await?,
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub enum StoredInputMessageContent {
    Text {
        value: Value,
    },
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
    pub async fn reresolve(self, config: &Config) -> Result<ResolvedInputMessageContent, Error> {
        match self {
            StoredInputMessageContent::Text { value } => {
                Ok(ResolvedInputMessageContent::Text { value })
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
                let object = get_object(config, file.storage_path.clone()).await?;
                Ok(ResolvedInputMessageContent::File(Box::new(FileWithPath {
                    file: Base64File {
                        url: file.file.url.clone(),
                        mime_type: file.file.mime_type.clone(),
                        data: object.data,
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
