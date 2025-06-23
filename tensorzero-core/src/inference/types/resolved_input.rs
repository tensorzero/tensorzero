use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tool::{ToolCall, ToolResult};

use super::{storage::StoragePath, Base64File, Role, Thought};

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
pub struct ResolvedInput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Value>,

    #[serde(default)]
    pub messages: Vec<ResolvedInputMessage>,
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
pub struct ResolvedInputMessage {
    pub role: Role,
    pub content: Vec<ResolvedInputMessageContent>,
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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(get_all, str))]
pub struct FileWithPath {
    #[serde(alias = "image")]
    pub file: Base64File,
    pub storage_path: StoragePath,
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
