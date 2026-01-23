//! Content types for input messages.
//!
//! This module contains the wire format types for various content blocks
//! that can appear in input messages.

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::types::PyModule;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tensorzero_derive::{TensorZeroDeserialize, export_schema};

/// A newtype wrapper around Map<String, Value> for template and system arguments
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(transparent)]
pub struct Arguments(
    // This type cannot be a Python dataclass because it's equivalent to a Map with arbitrary keys,
    // and Python dataclasses need its slots specified. So all references to this type need to be
    // `Map<String, Value>` in JSON schemas.
    pub Map<String, Value>,
);

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(deny_unknown_fields)]
#[export_schema]
pub struct Template {
    pub name: String,
    #[schemars(with = "Map<String, Value>")]
    pub arguments: Arguments,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, JsonSchema)]
#[serde(untagged)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[export_schema]
pub enum System {
    Text(String),
    #[schemars(with = "Map<String, Value>")]
    Template(Arguments),
}

/// InputMessages are validated against the input schema of the Function
/// and then templated and transformed into RequestMessages for a particular Variant.
/// They might contain tool calls or tool results along with text.
/// The abstraction we use to represent this is ContentBlock, which is a union of Text, ToolCall, and ToolResult.
/// ContentBlocks are collected into RequestMessages.
/// These RequestMessages are collected into a ModelInferenceRequest,
/// which should contain all information needed by a ModelProvider to perform the
/// inference that is called for.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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

#[cfg(feature = "pyo3")]
#[pymethods]
impl Text {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}

/// Struct that represents raw text content that should be passed directly to the model
/// without any template processing or validation
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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

#[cfg(feature = "pyo3")]
#[pymethods]
impl RawText {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}

/// Struct that represents an unknown provider-specific content block.
/// We pass this along as-is without any validation or transformation.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
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
/// If parsing the legacy format fails (e.g. the expected prefix/suffix markers are missing),
/// a deserialization error is returned.
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
    pub fn data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
        let json_str = serde_json::to_string(&self.data).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to serialize data: {e}"))
        })?;
        let json = PyModule::import(py, "json")?;
        let value = json.call_method1("loads", (json_str,))?;
        Ok(value)
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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, JsonSchema, PartialEq, Serialize, TensorZeroDeserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(get_all))]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[export_schema]
pub enum ThoughtSummaryBlock {
    #[schemars(title = "ThoughtSummaryBlockSummaryText")]
    SummaryText { text: String },
}

/// Struct that represents a model's reasoning
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
// Note: We don't use `get_all` because `extra_data` is `Value` which doesn't implement `IntoPyObject`.
// The fields are exposed via a manual `#[pymethods]` impl below.
#[cfg_attr(feature = "pyo3", pyclass)]
#[export_schema]
pub struct Thought {
    pub text: Option<String>,
    /// An optional signature - used with Anthropic and OpenRouter for multi-turn
    /// reasoning conversations. Other providers will ignore this field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<Vec<ThoughtSummaryBlock>>,
    /// When set, this `Thought` block will only be used for providers
    /// matching this type (e.g. `anthropic`). Other providers will emit
    /// a warning and discard the block.
    #[serde(
        // This alias is written to the database, so we cannot remove it.
        alias = "_internal_provider_type",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_type: Option<String>,
    /// Provider-specific opaque data for multi-turn reasoning support.
    /// For example, OpenRouter stores encrypted reasoning blocks with `{"format": "...", "encrypted": true}` structure.
    /// Note: Not exposed to Python because `Value` doesn't implement `IntoPyObject`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_data: Option<Value>,
}

#[cfg(feature = "pyo3")]
#[pyo3::pymethods]
impl Thought {
    #[getter]
    fn text(&self) -> Option<String> {
        self.text.clone()
    }

    #[getter]
    fn signature(&self) -> Option<String> {
        self.signature.clone()
    }

    #[getter]
    fn summary(&self) -> Option<Vec<ThoughtSummaryBlock>> {
        self.summary.clone()
    }

    #[getter]
    fn provider_type(&self) -> Option<String> {
        self.provider_type.clone()
    }
}
