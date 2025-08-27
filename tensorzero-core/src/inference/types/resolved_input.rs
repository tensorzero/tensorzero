use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{storage::StoragePath, Base64File, Role, Thought};
use crate::inference::types::file::Base64FileMetadata;
use crate::inference::types::stored_input::StoredFile;
use crate::inference::types::stored_input::{
    StoredInput, StoredInputMessage, StoredInputMessageContent,
};
use crate::tool::{ToolCall, ToolResult};

#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::{
    resolved_input_message_content_to_python, serialize_to_dict,
};
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

/// Like `Input`, but with all network resources resolved.
/// Currently, this is just used to fetch image URLs in the image input,
/// so that we always pass a base64-encoded image to the model provider.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct ResolvedInput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Value>,

    #[serde(default)]
    pub messages: Vec<ResolvedInputMessage>,
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
}

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

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
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
}

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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub enum ResolvedInputMessageContent {
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
            ResolvedInputMessageContent::Text { value } => {
                StoredInputMessageContent::Text { value }
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

#[cfg(feature = "pyo3")]
#[pymethods]
impl FileWithPath {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}
