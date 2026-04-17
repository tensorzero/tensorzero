//! Inference response types shared across the TensorZero crate ecosystem.
//!
//! These types represent the response from a TensorZero inference call.
//! They live in this leaf crate so that downstream consumers (e.g. `ts-executor-pool`)
//! can depend on them without pulling in all of `tensorzero-core`.

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tensorzero_derive::{TensorZeroDeserialize, export_schema};
use tensorzero_types::{InferenceResponseToolCall, Text, Thought, Unknown};
use uuid::Uuid;

use crate::{ContentBlock, ContentBlockOutput, FinishReason, RawUsageEntry, Usage};
use tensorzero_types::RawResponseEntry;

// =============================================================================
// ContentBlockChatOutput
// =============================================================================

/// Defines the types of content block that can come from a `chat` function
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, JsonSchema, PartialEq, Serialize, TensorZeroDeserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
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

// =============================================================================
// JsonInferenceOutput
// =============================================================================

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, JsonSchema)]
#[export_schema]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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
            Some(value) => {
                let json_str = serde_json::to_string(value).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Failed to serialize to JSON: {e:?}"
                    ))
                })?;
                let json_module = py.import("json")?;
                json_module.call_method1("loads", (json_str,))?.into_any()
            }
            None => py.None().into_bound(py),
        })
    }
}

// =============================================================================
// InferenceResponse
// =============================================================================

/// InferenceResponse and InferenceResultChunk determine what gets serialized and sent to the client
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(untagged, rename_all = "snake_case")]
pub enum InferenceResponse {
    Chat(ChatInferenceResponse),
    Json(JsonInferenceResponse),
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct ChatInferenceResponse {
    pub inference_id: Uuid,
    pub episode_id: Uuid,
    pub variant_name: String,
    pub content: Vec<ContentBlockChatOutput>,
    pub usage: Usage,
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_usage: Option<Vec<RawUsageEntry>>,
    /// DEPRECATED (#5697 / 2026.4+): Use `raw_response` instead.
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_response: Option<String>,
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_response: Option<Vec<RawResponseEntry>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct JsonInferenceResponse {
    pub inference_id: Uuid,
    pub episode_id: Uuid,
    pub variant_name: String,
    pub output: JsonInferenceOutput,
    pub usage: Usage,
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_usage: Option<Vec<RawUsageEntry>>,
    /// DEPRECATED (#5697 / 2026.4+): Use `raw_response` instead.
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_response: Option<String>,
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
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

// =============================================================================
// From impls
// =============================================================================

#[cfg(any(test, feature = "e2e_tests"))]
impl From<String> for ContentBlockChatOutput {
    fn from(text: String) -> Self {
        ContentBlockChatOutput::Text(Text { text })
    }
}

impl From<ContentBlockChatOutput> for ContentBlock {
    fn from(output: ContentBlockChatOutput) -> Self {
        match output {
            ContentBlockChatOutput::Text(text) => ContentBlock::Text(text),
            ContentBlockChatOutput::ToolCall(inference_response_tool_call) => {
                ContentBlock::ToolCall(inference_response_tool_call.into_tool_call())
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
                ContentBlockOutput::ToolCall(tool_call.into_tool_call())
            }
            ContentBlockChatOutput::Thought(thought) => ContentBlockOutput::Thought(thought),
            ContentBlockChatOutput::Unknown(unknown) => ContentBlockOutput::Unknown(unknown),
        }
    }
}
