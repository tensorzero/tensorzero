use std::{collections::HashMap, sync::Arc};

use crate::function::FunctionConfig;
#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::{
    content_block_chat_output_to_python, serialize_to_dict, uuid_to_python,
};
use crate::inference::types::stored_input::StoredInput;
use crate::inference::types::{RequestMessage, ResolvedRequestMessage, Text};
use crate::tool::{DynamicToolParams, StaticToolConfig};
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
use pyo3::{prelude::*, IntoPyObjectExt};
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

/// Wire variant of StoredInference for API responses with Python/TypeScript bindings
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "pyo3", pyclass(str, name = "StoredInference"))]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
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

impl StoredInferenceDatabase {
    /// Convert to wire type, properly handling tool params by subtracting static tools
    pub fn into_stored_inference(self, config: &Config) -> Result<StoredInference, Error> {
        match self {
            StoredInferenceDatabase::Chat(chat) => {
                let function_config = config.get_function(&chat.function_name)?;
                Ok(StoredInference::Chat(
                    chat.into_stored_inference(&function_config),
                ))
            }
            StoredInferenceDatabase::Json(json) => Ok(StoredInference::Json(json)),
        }
    }
}

impl StoredInference {
    /// Convert to storage type, properly handling tool params with function config
    pub fn to_storage(self, config: &Config) -> Result<StoredInferenceDatabase, Error> {
        match self {
            StoredInference::Chat(chat) => {
                let function_config = config.get_function(&chat.function_name)?;
                Ok(StoredInferenceDatabase::Chat(
                    chat.to_storage(&function_config, &config.tools)?,
                ))
            }
            StoredInference::Json(json) => Ok(StoredInferenceDatabase::Json(json)),
        }
    }
}

impl StoredChatInference {
    /// Convert to storage type, properly handling tool params with function config
    pub fn to_storage(
        self,
        function_config: &FunctionConfig,
        static_tools: &HashMap<String, Arc<StaticToolConfig>>,
    ) -> Result<StoredChatInferenceDatabase, Error> {
        let tool_params = function_config
            .dynamic_tool_params_to_database_insert(self.tool_params, static_tools)?
            .unwrap_or_default();

        Ok(StoredChatInferenceDatabase {
            function_name: self.function_name,
            variant_name: self.variant_name,
            input: self.input,
            output: self.output,
            dispreferred_outputs: self.dispreferred_outputs,
            timestamp: self.timestamp,
            episode_id: self.episode_id,
            inference_id: self.inference_id,
            tool_params,
            tags: self.tags,
        })
    }
}

/// Storage variant of StoredInference for database operations (no Python/TypeScript bindings)
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StoredInferenceDatabase {
    Chat(StoredChatInferenceDatabase),
    Json(StoredJsonInference),
}

impl std::fmt::Display for StoredInferenceDatabase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

/// Wire variant of StoredChatInference for API responses with Python/TypeScript bindings
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
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
    pub tool_params: DynamicToolParams,
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

impl StoredChatInferenceDatabase {
    /// Convert to wire type, properly handling tool params by subtracting static tools
    pub fn into_stored_inference(self, function_config: &FunctionConfig) -> StoredChatInference {
        let tool_params = function_config.database_insert_to_dynamic_tool_params(self.tool_params);

        StoredChatInference {
            function_name: self.function_name,
            variant_name: self.variant_name,
            input: self.input,
            output: self.output,
            dispreferred_outputs: self.dispreferred_outputs,
            timestamp: self.timestamp,
            episode_id: self.episode_id,
            inference_id: self.inference_id,
            tool_params,
            tags: self.tags,
        }
    }
}

/// Storage variant of StoredChatInference for database operations (no Python/TypeScript bindings)
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StoredChatInferenceDatabase {
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

impl std::fmt::Display for StoredChatInferenceDatabase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
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

impl StoredSample for StoredInferenceDatabase {
    fn input_mut(&mut self) -> &mut StoredInput {
        match self {
            StoredInferenceDatabase::Chat(example) => &mut example.input,
            StoredInferenceDatabase::Json(example) => &mut example.input,
        }
    }
    fn input(&self) -> &StoredInput {
        match self {
            StoredInferenceDatabase::Chat(example) => &example.input,
            StoredInferenceDatabase::Json(example) => &example.input,
        }
    }

    fn into_input(self) -> StoredInput {
        match self {
            StoredInferenceDatabase::Chat(example) => example.input,
            StoredInferenceDatabase::Json(example) => example.input,
        }
    }

    fn function_name(&self) -> &str {
        match self {
            StoredInferenceDatabase::Chat(example) => &example.function_name,
            StoredInferenceDatabase::Json(example) => &example.function_name,
        }
    }

    fn owned_simple_info(self) -> SimpleStoredSampleInfo {
        match self {
            StoredInferenceDatabase::Chat(example) => SimpleStoredSampleInfo {
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
            StoredInferenceDatabase::Json(example) => {
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
/// This is a wire type - it uses ToolCallConfigWire and has Python/TypeScript bindings.
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
    pub tool_params: Option<DynamicToolParams>,
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
    pub tool_params: Option<DynamicToolParams>,
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
    pub fn get_tool_params(&self) -> Option<DynamicToolParams> {
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

    // Convert tool_params from storage format to wire format, subtracting static tools
    let function_config = config.get_function(&function_name)?;
    let dynamic_tool_params =
        tool_params.map(|tp| function_config.database_insert_to_dynamic_tool_params(tp));

    Ok(RenderedSample {
        function_name,
        episode_id,
        inference_id,
        input: model_input,
        stored_input: resolved_input.into_stored_input()?,
        output,
        stored_output,
        dispreferred_outputs,
        tool_params: dynamic_tool_params,
        output_schema,
        tags,
    })
}
