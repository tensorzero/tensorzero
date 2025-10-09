use std::collections::HashMap;

#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::{
    content_block_chat_output_to_python, deserialize_from_pyobj, serialize_to_dict, uuid_to_python,
};
use crate::inference::types::stored_input::StoredInput;
use crate::inference::types::{RequestMessage, ResolvedRequestMessage, Text};
use crate::{
    config::Config,
    error::{Error, ErrorDetails},
    inference::types::{ContentBlockChatOutput, JsonInferenceOutput, ModelInput, ResolvedInput},
    tool::ToolCallConfigDatabaseInsert,
    variant::{chat_completion::prepare_model_input, VariantConfig},
};
use chrono::{DateTime, Utc};
#[cfg(feature = "pyo3")]
use pyo3::types::{PyAny, PyList};
#[cfg(feature = "pyo3")]
use pyo3::{exceptions::PyValueError, prelude::*, IntoPyObjectExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// This trait is used to represent a stored sample of data.
/// It should contain all the methods used by `render_samples`
/// from the stored sample of data so that we can abstract over the
/// different places where we could get training samples from, notably
/// datasets and stored inferences.
pub trait StoredSample {
    fn function_name(&self) -> &str;
    fn into_input(self) -> StoredInput;
    fn input(&self) -> &StoredInput;
    fn input_mut(&mut self) -> &mut StoredInput;
    fn owned_simple_info(self) -> SimpleStoredSampleInfo;
}

/// Utility struct that contains the information needed for a RenderedSample
/// that is just copied over from the StoredSample.
pub struct SimpleStoredSampleInfo {
    pub function_name: String,
    pub input: StoredInput,
    pub episode_id: Option<Uuid>,
    pub inference_id: Option<Uuid>,
    pub output: Option<Vec<ContentBlockChatOutput>>,
    pub stored_output: Option<StoredOutput>,
    pub dispreferred_outputs: Vec<Vec<ContentBlockChatOutput>>,
    pub tool_params: Option<ToolCallConfigDatabaseInsert>,
    pub output_schema: Option<Value>,
    pub tags: HashMap<String, String>,
}

/// Represents an stored inference to be used for optimization.
/// These are retrieved from the database in this format.
/// NOTE / TODO: As an incremental step we are deserializing this enum from Python.
/// in the final version we should instead make this a native PyO3 class and
/// avoid deserialization entirely unless given a dict.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub enum StoredInference {
    Chat(StoredChatInference),
    Json(StoredJsonInference),
}

impl std::fmt::Display for StoredInference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl StoredInference {
    #[expect(clippy::too_many_arguments)]
    #[new]
    pub fn new<'py>(
        py: Python<'py>,
        r#type: String,
        function_name: String,
        variant_name: String,
        input: Bound<'py, PyAny>,
        output: Bound<'py, PyAny>,
        episode_id: Bound<'py, PyAny>,
        inference_id: Bound<'py, PyAny>,
        dispreferred_outputs: Option<Bound<'py, PyAny>>,
        tool_params: Option<Bound<'py, PyAny>>,
        output_schema: Option<Bound<'py, PyAny>>,
        tags: Option<Bound<'py, PyAny>>,
        timestamp: Bound<'py, PyAny>,
    ) -> PyResult<Self> {
        let input: StoredInput = deserialize_from_pyobj(py, &input)?;
        let episode_id: Uuid = deserialize_from_pyobj(py, &episode_id)?;
        let inference_id: Uuid = deserialize_from_pyobj(py, &inference_id)?;
        let timestamp: DateTime<Utc> = deserialize_from_pyobj(py, &timestamp)?;
        let tags: HashMap<String, String> = tags
            .as_ref()
            .map(|x| deserialize_from_pyobj(py, x))
            .transpose()?
            .unwrap_or_default();
        match r#type.as_str() {
            "chat" => {
                let output: Vec<ContentBlockChatOutput> = deserialize_from_pyobj(py, &output)?;
                let dispreferred_outputs: Option<Vec<Vec<ContentBlockChatOutput>>> =
                    dispreferred_outputs
                        .map(|x| deserialize_from_pyobj(py, &x))
                        .transpose()?;
                let Some(tool_params) = tool_params.map(|x| deserialize_from_pyobj(py, &x)) else {
                    return Err(PyValueError::new_err(
                        "tool_params is required for chat inferences",
                    ));
                };
                let tool_params: ToolCallConfigDatabaseInsert = tool_params?;
                Ok(Self::Chat(StoredChatInference {
                    function_name,
                    variant_name,
                    input,
                    output,
                    dispreferred_outputs: dispreferred_outputs.unwrap_or_default(),
                    episode_id,
                    inference_id,
                    tool_params,
                    tags,
                    timestamp,
                }))
            }
            "json" => {
                let output: JsonInferenceOutput = deserialize_from_pyobj(py, &output)?;
                let dispreferred_outputs: Option<Vec<JsonInferenceOutput>> = dispreferred_outputs
                    .map(|x| deserialize_from_pyobj(py, &x))
                    .transpose()?;
                let Some(output_schema) = output_schema.map(|x| deserialize_from_pyobj(py, &x))
                else {
                    return Err(PyValueError::new_err(
                        "output_schema is required for json inferences",
                    ));
                };
                let output_schema: Value = output_schema?;
                Ok(Self::Json(StoredJsonInference {
                    function_name,
                    variant_name,
                    input,
                    output,
                    dispreferred_outputs: dispreferred_outputs.unwrap_or_default(),
                    episode_id,
                    inference_id,
                    output_schema,
                    tags,
                    timestamp,
                }))
            }
            _ => Err(PyValueError::new_err(format!("Invalid type: {type}"))),
        }
    }

    pub fn __repr__(&self) -> String {
        self.to_string()
    }

    #[getter]
    pub fn get_function_name(&self) -> String {
        match self {
            StoredInference::Chat(example) => example.function_name.clone(),
            StoredInference::Json(example) => example.function_name.clone(),
        }
    }

    #[getter]
    pub fn get_variant_name(&self) -> String {
        match self {
            StoredInference::Chat(example) => example.variant_name.clone(),
            StoredInference::Json(example) => example.variant_name.clone(),
        }
    }

    #[getter]
    pub fn get_input(&self) -> StoredInput {
        match self {
            StoredInference::Chat(example) => example.input.clone(),
            StoredInference::Json(example) => example.input.clone(),
        }
    }

    /// Returns the output of the inference as PyO3 classes.
    /// This is actually a List of ContentBlockChatOutputs for StoredChatInference
    /// and a JsonInferenceOutput for StoredJsonInference.
    #[getter]
    pub fn get_output<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self {
            StoredInference::Chat(example) => example
                .output
                .iter()
                .map(|x| content_block_chat_output_to_python(py, x.clone()))
                .collect::<PyResult<Vec<_>>>()?
                .into_bound_py_any(py)?,
            StoredInference::Json(example) => example.output.clone().into_bound_py_any(py)?,
        })
    }

    #[getter]
    pub fn get_dispreferred_outputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self {
            StoredInference::Chat(example) => example
                .dispreferred_outputs
                .iter()
                .map(|x| {
                    x.iter()
                        .map(|y| content_block_chat_output_to_python(py, y.clone()))
                        .collect::<PyResult<Vec<_>>>()
                })
                .collect::<PyResult<Vec<Vec<_>>>>()?
                .into_bound_py_any(py)?,
            StoredInference::Json(example) => {
                example.dispreferred_outputs.clone().into_bound_py_any(py)?
            }
        })
    }

    #[getter]
    pub fn get_episode_id<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            StoredInference::Chat(example) => uuid_to_python(py, example.episode_id),
            StoredInference::Json(example) => uuid_to_python(py, example.episode_id),
        }
    }

    #[getter]
    pub fn get_inference_id<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            StoredInference::Chat(example) => uuid_to_python(py, example.inference_id),
            StoredInference::Json(example) => uuid_to_python(py, example.inference_id),
        }
    }

    #[getter]
    pub fn get_tool_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self {
            StoredInference::Chat(example) => {
                example.tool_params.clone().into_py_any(py)?.into_bound(py)
            }
            // Json inferences don't have tool params
            StoredInference::Json(_) => py.None().into_bound(py),
        })
    }

    #[getter]
    pub fn get_output_schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self {
            StoredInference::Chat(_) => py.None().into_bound(py),
            StoredInference::Json(example) => {
                serialize_to_dict(py, example.output_schema.clone())?.into_bound(py)
            }
        })
    }

    #[getter]
    pub fn get_type(&self) -> String {
        match self {
            StoredInference::Chat(_) => "chat".to_string(),
            StoredInference::Json(_) => "json".to_string(),
        }
    }

    #[getter]
    pub fn get_tags(&self) -> HashMap<String, String> {
        match self {
            StoredInference::Chat(example) => example.tags.clone(),
            StoredInference::Json(example) => example.tags.clone(),
        }
    }

    #[getter]
    pub fn get_timestamp(&self) -> String {
        match self {
            StoredInference::Chat(example) => example.timestamp.to_rfc3339(),
            StoredInference::Json(example) => example.timestamp.to_rfc3339(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct StoredChatInference {
    pub function_name: String,
    pub variant_name: String,
    pub input: StoredInput,
    pub output: Vec<ContentBlockChatOutput>,
    #[serde(default)]
    pub dispreferred_outputs: Vec<Vec<ContentBlockChatOutput>>,
    pub timestamp: DateTime<Utc>,
    pub episode_id: Uuid,
    pub inference_id: Uuid,
    #[serde(default)]
    pub tool_params: ToolCallConfigDatabaseInsert,
    #[serde(default)]
    pub tags: HashMap<String, String>,
}

impl std::fmt::Display for StoredChatInference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl StoredChatInference {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct StoredJsonInference {
    pub function_name: String,
    pub variant_name: String,
    pub input: StoredInput,
    pub output: JsonInferenceOutput,
    #[serde(default)]
    pub dispreferred_outputs: Vec<JsonInferenceOutput>,
    pub timestamp: DateTime<Utc>,
    pub episode_id: Uuid,
    pub inference_id: Uuid,
    pub output_schema: Value,
    #[serde(default)]
    pub tags: HashMap<String, String>,
}

impl std::fmt::Display for StoredJsonInference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl StoredJsonInference {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}

impl StoredSample for StoredInference {
    fn input_mut(&mut self) -> &mut StoredInput {
        match self {
            StoredInference::Chat(example) => &mut example.input,
            StoredInference::Json(example) => &mut example.input,
        }
    }
    fn input(&self) -> &StoredInput {
        match self {
            StoredInference::Chat(example) => &example.input,
            StoredInference::Json(example) => &example.input,
        }
    }

    fn into_input(self) -> StoredInput {
        match self {
            StoredInference::Chat(example) => example.input,
            StoredInference::Json(example) => example.input,
        }
    }

    fn function_name(&self) -> &str {
        match self {
            StoredInference::Chat(example) => &example.function_name,
            StoredInference::Json(example) => &example.function_name,
        }
    }

    fn owned_simple_info(self) -> SimpleStoredSampleInfo {
        match self {
            StoredInference::Chat(example) => SimpleStoredSampleInfo {
                function_name: example.function_name,
                input: example.input,
                episode_id: Some(example.episode_id),
                inference_id: Some(example.inference_id),
                output: Some(example.output.clone()),
                stored_output: Some(StoredOutput::Chat(example.output)),
                dispreferred_outputs: example.dispreferred_outputs,
                tool_params: Some(example.tool_params),
                output_schema: None,
                tags: example.tags,
            },
            StoredInference::Json(example) => {
                let output = json_output_to_content_block_chat_output(example.output.clone());
                let dispreferred_outputs = example
                    .dispreferred_outputs
                    .into_iter()
                    .map(json_output_to_content_block_chat_output)
                    .collect();
                SimpleStoredSampleInfo {
                    function_name: example.function_name,
                    input: example.input,
                    episode_id: Some(example.episode_id),
                    inference_id: Some(example.inference_id),
                    output: Some(output),
                    stored_output: Some(StoredOutput::Json(example.output)),
                    dispreferred_outputs,
                    tool_params: None,
                    output_schema: Some(example.output_schema),
                    tags: example.tags,
                }
            }
        }
    }
}

fn json_output_to_content_block_chat_output(
    output: JsonInferenceOutput,
) -> Vec<ContentBlockChatOutput> {
    match output.raw {
        Some(raw) => vec![ContentBlockChatOutput::Text(Text { text: raw })],
        None => vec![],
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[serde(untagged)]
pub enum StoredOutput {
    Chat(Vec<ContentBlockChatOutput>),
    Json(JsonInferenceOutput),
}

/// Represents an inference that has been prepared for fine-tuning.
/// This is constructed by rendering a StoredInference with a variant for messages
/// and by resolving all network resources (e.g. images).
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(any(feature = "e2e_tests", test), derive(PartialEq))]
pub struct RenderedSample {
    pub function_name: String,
    pub input: ModelInput,
    pub stored_input: StoredInput,
    pub output: Option<Vec<ContentBlockChatOutput>>,
    pub stored_output: Option<StoredOutput>,
    pub dispreferred_outputs: Vec<Vec<ContentBlockChatOutput>>,
    pub episode_id: Option<Uuid>,
    pub inference_id: Option<Uuid>,
    pub tool_params: Option<ToolCallConfigDatabaseInsert>,
    pub output_schema: Option<Value>,
    pub tags: HashMap<String, String>,
}

impl RenderedSample {
    pub fn into_lazy_rendered_sample(self) -> LazyRenderedSample {
        LazyRenderedSample {
            function_name: self.function_name,
            system_input: self.input.system,
            messages: self
                .input
                .messages
                .into_iter()
                .map(ResolvedRequestMessage::into_request_message)
                .collect(),
            stored_input: self.stored_input,
            output: self.output,
            stored_output: self.stored_output,
            dispreferred_outputs: self.dispreferred_outputs,
            episode_id: self.episode_id,
            inference_id: self.inference_id,
            tool_params: self.tool_params,
            output_schema: self.output_schema,
            tags: self.tags,
        }
    }
}

/// Like `RenderedSample`, but holds `RequestMessage`s instead of `ResolvedRequestMessage`s
pub struct LazyRenderedSample {
    pub function_name: String,
    pub system_input: Option<String>,
    // This is a a `Vec<ResolvedRequestMessage>` in `RenderedSample`
    pub messages: Vec<RequestMessage>,
    pub stored_input: StoredInput,
    pub output: Option<Vec<ContentBlockChatOutput>>,
    pub stored_output: Option<StoredOutput>,
    pub dispreferred_outputs: Vec<Vec<ContentBlockChatOutput>>,
    pub episode_id: Option<Uuid>,
    pub inference_id: Option<Uuid>,
    pub tool_params: Option<ToolCallConfigDatabaseInsert>,
    pub output_schema: Option<Value>,
    pub tags: HashMap<String, String>,
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl RenderedSample {
    #[getter]
    pub fn get_function_name(&self) -> &str {
        &self.function_name
    }

    #[getter]
    pub fn get_input(&self) -> ModelInput {
        self.input.clone()
    }

    #[getter]
    pub fn get_output<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if let Some(output) = &self.output {
            let output = output
                .iter()
                .map(|x| content_block_chat_output_to_python(py, x.clone()))
                .collect::<PyResult<Vec<_>>>()?;
            PyList::new(py, output).map(Bound::into_any)
        } else {
            Ok(py.None().into_bound(py))
        }
    }

    #[getter]
    pub fn get_stored_output<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if let Some(stored_output) = &self.stored_output {
            match stored_output {
                StoredOutput::Chat(output) => {
                    let output = output
                        .iter()
                        .map(|x| content_block_chat_output_to_python(py, x.clone()))
                        .collect::<PyResult<Vec<_>>>()?;
                    PyList::new(py, output).map(Bound::into_any)
                }
                StoredOutput::Json(output) => Ok(output.clone().into_py_any(py)?.into_bound(py)),
            }
        } else {
            Ok(py.None().into_bound(py))
        }
    }

    #[getter]
    pub fn get_dispreferred_outputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dispreferred_outputs = self
            .dispreferred_outputs
            .iter()
            .map(|x| {
                x.iter()
                    .map(|y| content_block_chat_output_to_python(py, y.clone()))
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, dispreferred_outputs).map(Bound::into_any)
    }

    #[getter]
    pub fn get_tool_params(&self) -> Option<ToolCallConfigDatabaseInsert> {
        self.tool_params.clone()
    }

    #[getter]
    pub fn get_output_schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        serialize_to_dict(py, self.output_schema.clone()).map(|x| x.into_bound(py))
    }

    #[getter]
    pub fn get_episode_id<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.episode_id {
            Some(id) => uuid_to_python(py, id),
            None => Ok(py.None().into_bound(py)),
        }
    }

    #[getter]
    pub fn get_inference_id<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.inference_id {
            Some(id) => uuid_to_python(py, id),
            None => Ok(py.None().into_bound(py)),
        }
    }

    pub fn __repr__(&self) -> String {
        self.to_string()
    }

    #[getter]
    pub fn get_tags(&self) -> HashMap<String, String> {
        self.tags.clone()
    }

    #[getter]
    pub fn get_stored_input(&self) -> StoredInput {
        self.stored_input.clone()
    }
}

impl std::fmt::Display for RenderedSample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Serialize the rendered inference to pretty-printed JSON
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

/// Convert a StoredInference's input to a ModelInput.
/// `variants` should be a map from function name to variant name, i.e. what variant to use for a particular function
/// as the stored inference is being rendered.
/// This does not handle resolving network resources (e.g. images).
async fn render_model_input(
    resolved_input: &ResolvedInput,
    function_name: &str,
    config: &Config,
    variants: &HashMap<String, String>,
) -> Result<ModelInput, Error> {
    let variant_name = variants.get(function_name).ok_or_else(|| {
        Error::new(ErrorDetails::MissingFunctionInVariants {
            function_name: function_name.to_string(),
        })
    })?;
    let function_config = config.get_function(function_name)?;
    let variant_config = function_config
        .variants()
        .get(variant_name)
        .ok_or_else(|| {
            Error::new(ErrorDetails::UnknownVariant {
                name: variant_name.clone(),
            })
        })?;
    let VariantConfig::ChatCompletion(chat_completion_config) = &variant_config.inner else {
        return Err(Error::new(ErrorDetails::InvalidVariantForOptimization {
            function_name: function_name.to_string(),
            variant_name: variant_name.clone(),
        }));
    };
    prepare_model_input(
        resolved_input.system.as_ref(),
        &resolved_input.messages,
        &config.templates,
        chat_completion_config.templates(),
    )
    .await
}

/// Render an impl StoredSample to a RenderedStoredInference.
/// `variants` should be a map from function name to variant name, i.e. what variant to use for a particular function
/// as the inference example is being rendered.
///
/// This does not handle resolving network resources (e.g. images).
pub async fn render_stored_sample<T: StoredSample>(
    stored_sample: T,
    resolved_input: ResolvedInput,
    config: &Config,
    variants: &HashMap<String, String>,
) -> Result<RenderedSample, Error> {
    let SimpleStoredSampleInfo {
        function_name,
        input: _,
        output,
        stored_output,
        dispreferred_outputs,
        tool_params,
        output_schema,
        episode_id,
        inference_id,
        tags,
    } = stored_sample.owned_simple_info();
    let model_input = render_model_input(&resolved_input, &function_name, config, variants).await?;
    Ok(RenderedSample {
        function_name,
        episode_id,
        inference_id,
        input: model_input,
        stored_input: resolved_input.into_stored_input(),
        output,
        stored_output,
        dispreferred_outputs,
        tool_params,
        output_schema,
        tags,
    })
}
