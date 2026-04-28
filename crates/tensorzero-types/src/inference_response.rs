//! Types for inference responses.
//!
//! These types define the wire format for inference responses sent back to clients.

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::types::PyModule;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tensorzero_derive::{TensorZeroDeserialize, export_schema};
use uuid::Uuid;

use crate::content::{Text, Thought, Unknown};
use crate::tool::InferenceResponseToolCall;
use crate::usage::{FinishReason, RawResponseEntry, RawUsageEntry, Usage};

// =============================================================================
// ContentBlockChatOutput
// =============================================================================

/// Defines the types of content block that can come from a `chat` function
#[derive(ts_rs::TS, Clone, Debug, JsonSchema, PartialEq, Serialize, TensorZeroDeserialize)]
#[ts(export, optional_fields)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
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

impl From<String> for ContentBlockChatOutput {
    fn from(text: String) -> Self {
        ContentBlockChatOutput::Text(Text { text })
    }
}

// =============================================================================
// JsonInferenceOutput
// =============================================================================

#[derive(ts_rs::TS, Clone, Debug, Deserialize, Serialize, PartialEq, JsonSchema)]
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
    fn get_parsed<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
        Ok(match &self.parsed {
            Some(value) => {
                let json_str = serde_json::to_string(value).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Failed to serialize parsed value: {e}"
                    ))
                })?;
                let json = PyModule::import(py, "json")?;
                json.call_method1("loads", (json_str,))?
            }
            None => py.None().into_bound(py),
        })
    }
}

// =============================================================================
// InferenceResponse types
// =============================================================================

/// InferenceResponse determines what gets serialized and sent to the client
#[derive(ts_rs::TS, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[ts(export)]
#[serde(untagged, rename_all = "snake_case")]
pub enum InferenceResponse {
    Chat(ChatInferenceResponse),
    Json(JsonInferenceResponse),
}

#[derive(ts_rs::TS, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[ts(export)]
pub struct ChatInferenceResponse {
    pub inference_id: Uuid,
    pub episode_id: Uuid,
    pub variant_name: String,
    pub content: Vec<ContentBlockChatOutput>,
    pub usage: Usage,
    #[ts(optional)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_usage: Option<Vec<RawUsageEntry>>,
    /// DEPRECATED (#5697 / 2026.4+): Use `raw_response` instead.
    #[ts(optional)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_response: Option<String>,
    #[ts(optional)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_response: Option<Vec<RawResponseEntry>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

#[derive(ts_rs::TS, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[ts(export)]
pub struct JsonInferenceResponse {
    pub inference_id: Uuid,
    pub episode_id: Uuid,
    pub variant_name: String,
    pub output: JsonInferenceOutput,
    pub usage: Usage,
    #[ts(optional)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_usage: Option<Vec<RawUsageEntry>>,
    /// DEPRECATED (#5697 / 2026.4+): Use `raw_response` instead.
    #[ts(optional)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_response: Option<String>,
    #[ts(optional)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_response: Option<Vec<RawResponseEntry>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

impl InferenceResponse {
    pub fn usage(&self) -> Usage {
        match self {
            InferenceResponse::Chat(c) => c.usage,
            InferenceResponse::Json(j) => j.usage,
        }
    }

    pub fn raw_usage(&self) -> Option<&Vec<RawUsageEntry>> {
        match self {
            InferenceResponse::Chat(c) => c.raw_usage.as_ref(),
            InferenceResponse::Json(j) => j.raw_usage.as_ref(),
        }
    }

    pub fn raw_response(&self) -> Option<&Vec<RawResponseEntry>> {
        match self {
            InferenceResponse::Chat(c) => c.raw_response.as_ref(),
            InferenceResponse::Json(j) => j.raw_response.as_ref(),
        }
    }

    pub fn finish_reason(&self) -> Option<FinishReason> {
        match self {
            InferenceResponse::Chat(c) => c.finish_reason,
            InferenceResponse::Json(j) => j.finish_reason,
        }
    }

    pub fn variant_name(&self) -> &str {
        match self {
            InferenceResponse::Chat(c) => &c.variant_name,
            InferenceResponse::Json(j) => &j.variant_name,
        }
    }

    pub fn inference_id(&self) -> Uuid {
        match self {
            InferenceResponse::Chat(c) => c.inference_id,
            InferenceResponse::Json(j) => j.inference_id,
        }
    }

    pub fn episode_id(&self) -> Uuid {
        match self {
            InferenceResponse::Chat(c) => c.episode_id,
            InferenceResponse::Json(j) => j.episode_id,
        }
    }
}
